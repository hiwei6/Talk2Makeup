import sys

# 在 job script（非 TTY）環境下 stdout 是全緩衝，強制改為行緩衝
# 讓 print 訊息即時寫入 log 檔（等同 python -u 的效果）
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import gradio as gr
import torch
import numpy as np
import cv2
import os
import gc
import re
import json
from PIL import Image, ImageColor
from diffusers import DDIMScheduler
from pipeline_dreammakeup import DreamMakeupPipeline
from openai import OpenAI

# 切換到腳本所在目錄，確保相對路徑正確
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 引用工具函式
from utils.color import apply_makeup_natural
from utils.mask import gradation_smoothing
from models.bisenet import BiSeNet
from utils.segmentation import load_bisenet, get_face_masks, get_advanced_masks, generate_realistic_eyeshadow, visualize_parsing_maps

# 引入 Hugging Face 工具來關閉進度條
try:
    from huggingface_hub.utils import disable_progress_bars
except ImportError:
    from huggingface_hub import disable_progress_bars

# ==========================================
# 1. 基礎設定與模型載入 (Diffusion & BiSeNet)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "assets/BKG1.safetensors"
WEIGHTS_PATH = "assets/79999_iter.pth"

global_models = {"pipe": None, "net": None}

def load_models():
    """載入 Diffusion Pipeline 與 BiSeNet (FP16 記憶體優化版)"""
    if global_models["pipe"] is not None:
        return global_models["pipe"], global_models["net"]

    print(f"[{DEVICE}] 正在載入模型 (FP16 Compact Mode)...")
    disable_progress_bars()

    # 1. 載入 BiSeNet
    net = load_bisenet(WEIGHTS_PATH, device=DEVICE)
    print("✅ BiSeNet loaded.")

    # 2. 載入 Pipeline
    print("Loading pipeline (FP16)...")
    try:
        pipe = DreamMakeupPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=None,
            variant="fp16",
            torch_dtype=torch.float16,
        )
    except Exception as e:
        print(f"FP16 variant load failed, trying standard FP32 load: {e}")
        pipe = DreamMakeupPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=None,
            torch_dtype=torch.float32,
        )

    # 3. 移至 GPU
    if DEVICE == "cuda":
        print("Moving pipeline to CUDA...")
        pipe = pipe.to("cuda")

    # 4. 設定 Scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # 5. 載入 LoRA
    if os.path.exists(LORA_PATH):
        try:
            pipe.load_lora_weights(".", weight_name=LORA_PATH)
            pipe.fuse_lora(lora_scale=0.2)
            print("✅ LoRA Loaded.")
        except Exception as e:
            print(f"⚠️ LoRA 載入失敗 (已忽略): {e}")

    global_models["pipe"] = pipe
    global_models["net"] = net

    return pipe, net

def preprocess_image(image_np: np.ndarray) -> Image.Image:
    """Letterbox to square with reflect padding, then resize to 512×512.
    Preserves the full original image without any cropping.
    Reflect padding avoids hard black/white edges that could confuse BiSeNet or diffusion.
    """
    arr = np.array(Image.fromarray(image_np).convert("RGB"))
    h, w = arr.shape[:2]
    long = max(w, h)
    pad_top    = (long - h) // 2
    pad_bottom = long - h - pad_top
    pad_left   = (long - w) // 2
    pad_right  = long - w - pad_left
    arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=255)
    return Image.fromarray(arr).resize((512, 512), Image.LANCZOS)

def hex_to_rgb(color_str):
    if not color_str: return [0, 0, 0]
    if color_str.startswith("#"):
        try: return list(ImageColor.getcolor(color_str, "RGB"))
        except: return [0, 0, 0]
    if "rgb" in color_str:
        try:
            nums = re.findall(r"[\d\.]+", color_str)
            if len(nums) >= 3: return [int(float(n)) for n in nums[:3]]
        except: pass
    return [0, 0, 0]

# ==========================================
# 2. LLM 魔法解析模組 (Ollama)
# ==========================================
# 設定本地端 Ollama 伺服器
LLM_CLIENT = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
    timeout=400.0,
)
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

SYSTEM_PROMPT = """You are a professional makeup artist AI. Convert the user's makeup description into a JSON object.

REASONING FIRST (this is the key step — do NOT skip it or rush to fill in default numbers):
Before deciding any color/alpha/size, think through it like a real makeup artist would for ANY description you're
given — whether it names a known style, blends several, or is purely descriptive with no name at all. There is no
fixed list of "known styles" to match against; reason from first principles every time:
  1. What is the overall mood/reference this evokes (era, occasion, culture, intensity, realism vs. theatrical)?
  2. Given that mood, which face parts would a real makeup artist actually touch — including parts the user never
     named, if the mood clearly implies them (e.g. a described mood that's inherently about a dramatically altered
     base complexion implies foundation should be on, even if "foundation" was never said)?
  3. For each part you're touching, how opaque/saturated and how large would it realistically need to be to actually
     read as that mood in a photo — not "a safe middle value", but what the mood specifically calls for?
Write this reasoning concisely into the "style_analysis" field. Let it genuinely drive the numbers you output next —
if style_analysis says something is dramatic/full-coverage, the matching alpha must actually be high, not moderate.

STRICT RULES:
- Output ONLY valid JSON. No explanation, no markdown, no code blocks.
- "color" must be a 6-digit hex code like "#D9334D".
- "alpha" controls opacity. Use the FULL 0.0–2.0 range based on your style_analysis, not a cautious middle value:
  * 0.0 = this part plays no role in the look (neither stated nor implied)
  * 0.2-0.5 = sheer/barely-there
  * 0.6-1.0 = clearly visible, bold/saturated
  * 1.1-2.0 = near/full opaque coverage — for whenever the analysis concluded a part must completely override the
    natural skin/lip tone to read correctly. Don't be conservative here if the mood calls for it.
- "prompt" is an English Stable Diffusion keyword (e.g. "glossy red lips"). Empty string if alpha=0.
- "global_prompt" is English, describes overall style.
- "reason" explains WHY you chose this color/alpha for this part, in Traditional Chinese (繁體中文), 1-2 sentences,
  consistent with your style_analysis.
- "global_reason" explains the overall design intent in Traditional Chinese, 1-2 sentences.
- "blush" and "eyeshadow" additionally need a "size" field (float, 0.5–2.0, default 1.0) for how LARGE the painted
  area is (independent of "alpha", which is intensity, not area). Derive it from style_analysis the same way as
  alpha — small/precise vs. wide/dramatic — rather than a list of trigger phrases.

OUTPUT FORMAT (fill every field; the same reasoning-driven judgment applies no matter what the input looks like):
{"style_analysis":"...","global_prompt":"...","global_reason":"...","foundation":{"color":"#HEX","alpha":0.0,"prompt":"","reason":"..."},"lips":{"color":"#HEX","alpha":0.5,"prompt":"...","reason":"..."},"blush":{"color":"#HEX","alpha":0.3,"size":1.0,"prompt":"...","reason":"..."},"eyeshadow":{"color":"#HEX","alpha":0.4,"size":1.0,"prompt":"...","reason":"..."},"brows":{"color":"#HEX","alpha":0.3,"prompt":"...","reason":"..."}}

EXAMPLE — input "natural Korean look, pink lips, soft blush" (a mild, low-coverage mood → low alphas, small sizes):
{"style_analysis":"這是清透自然的韓系日常妝，強調透膚感而非遮蓋力，所有部位都應維持低到中濃度、貼近原膚色與小範圍暈染，底妝不需額外處理。","global_prompt":"Korean natural makeup, soft and dewy","global_reason":"整體走韓系清透路線，以低飽和度色系和輕薄妝感呈現自然水潤的氛圍。","foundation":{"color":"#FFE4C4","alpha":0.0,"prompt":"","reason":"韓系裸妝本就強調保留原膚色透氣感，不需額外遮蓋。"},"lips":{"color":"#FFB6C1","alpha":0.5,"prompt":"soft pink glossy lips","reason":"粉嫩玫瑰色呼應韓系少女感，中等濃度保持水潤透明質感。"},"blush":{"color":"#F4A0A0","alpha":0.35,"size":0.8,"prompt":"soft pink blush","reason":"淡粉色腮紅以低濃度、小範圍暈染於蘋果肌，製造自然的血色感與水光肌效果。"},"eyeshadow":{"color":"#C8A882","alpha":0.3,"size":0.9,"prompt":"neutral eyeshadow","reason":"米棕色眼影小範圍淡掃加深眼神層次，不搶眼但增加立體感。"},"brows":{"color":"#5C3D1E","alpha":0.3,"prompt":"natural soft brows","reason":"深棕色描繪平直眉，符合韓系妝容的標誌性眉型。"}}

EXAMPLE — input "整臉慘白、誇張放大的正紅嘴唇，像馬戲團表演者" (a described mood implying full-coverage base + bold lips,
even though the user only explicitly named two things — style_analysis is what surfaces the rest):
{"style_analysis":"使用者描述的是舞台/戲劇化的表演妝感：『整臉慘白』直接要求底妝要接近全遮蓋才能蓋掉原膚色，『誇張放大的正紅嘴唇』表示唇妝需要高飽和度且視覺範圍大，這類妝容通常伴隨誇張的五官輪廓（粗眉、明顯眼線），因此眉毛也拉高濃度呼應整體戲劇感，但使用者未強調眼影，故維持低調。","global_prompt":"theatrical performer makeup, chalk white face, oversized bold red lips","global_reason":"以近乎全白的底妝與誇大的正紅唇為核心，呈現馬戲團表演者的戲劇化舞台感。","foundation":{"color":"#FAF6F0","alpha":1.4,"prompt":"chalk white opaque stage makeup base","reason":"『整臉慘白』要求底妝完全覆蓋原膚色，需要接近全遮蓋的高濃度粉白色。"},"lips":{"color":"#D6001C","alpha":1.3,"prompt":"oversized bold saturated red lips","reason":"『誇張放大的正紅嘴唇』需要高飽和度與高濃度，才能呈現舞台感的鮮豔實色而非淡妝的粉色。"},"blush":{"color":"#E8394A","alpha":0.4,"size":1.0,"prompt":"subtle red cheek tint","reason":"使用者未特別描述腮紅，給予中等濃度的紅色點綴以呼應整體戲劇色調，不搶過唇妝的焦點。"},"eyeshadow":{"color":"#8A8580","alpha":0.2,"size":0.9,"prompt":"minimal neutral eye definition","reason":"使用者未提及眼妝需求，僅給予淡淡的輪廓感，讓焦點集中在底妝與唇妝。"},"brows":{"color":"#1A1A1A","alpha":0.6,"prompt":"bold defined eyebrows","reason":"戲劇化妝容通常伴隨明顯的眉型輪廓，中高濃度呼應整體誇張感。"}}
"""

def _llm_result_to_markdown(data: dict) -> str:
    """將 LLM 解析結果轉為易讀的 Markdown 摘要"""
    PART_LABELS = {
        "foundation": "底妝", "lips": "口紅",
        "blush": "腮紅", "eyeshadow": "眼影", "brows": "眉毛",
    }
    SIZE_PARTS = {"blush", "eyeshadow"}
    style_analysis = data.get("style_analysis", "")
    global_reason = data.get("global_reason", "")
    lines = [
        f"**妝容分析**：{style_analysis}" if style_analysis else "",
        f"**整體風格**：{data.get('global_prompt', '')}",
        f"*{global_reason}*" if global_reason else "",
        "",
        "| 部位 | 顏色 | 濃度 | 範圍 | 質感描述 | 設計說明 |",
        "|:----:|:----:|:----:|:----:|:---------|:---------|",
    ]
    for key, label in PART_LABELS.items():
        p = data.get(key, {})
        alpha = float(p.get("alpha", 0))
        color = p.get("color", "#000000")
        prompt = p.get("prompt", "") or "—"
        reason = p.get("reason", "") or "—"
        bar = "█" * int(alpha * 5) + "░" * (10 - int(alpha * 5))
        if alpha == 0:
            color_cell = "—"
            alpha_cell = "不套用"
        else:
            color_cell = f"`{color}`"
            alpha_cell = f"{alpha:.1f} {bar}"
        if key in SIZE_PARTS and alpha > 0:
            size = float(p.get("size", 1.0))
            size_cell = f"{size:.1f}x"
        else:
            size_cell = "—"
        lines.append(f"| **{label}** | {color_cell} | {alpha_cell} | {size_cell} | {prompt} | {reason} |")
    return "\n".join(lines)

def parse_makeup_with_llm(user_input):
    """接收使用者的自然語言，呼叫 LLM 轉換為 UI 參數，並回傳分析摘要"""
    if not user_input.strip():
        raise gr.Error("請輸入你想要的妝容描述！")

    print(f"[LLM] 正在解析: {user_input}")
    try:
        response = LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"請幫我設計這個妝容：{user_input}"},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json\n?(.*?)\n?```", r"\1", content, flags=re.DOTALL)
        data = json.loads(content)
        print("[LLM] 解析成功")
    except Exception as e:
        print(f"[LLM] 解析失敗: {e}")
        raise gr.Error(f"LLM 解析失敗，請確認 Ollama 正在運行。錯誤：{e}")

    def safe(part, key, default):
        return data.get(part, {}).get(key, default)

    summary_md = _llm_result_to_markdown(data)
    raw_json   = json.dumps(data, ensure_ascii=False, indent=2)

    return (
        data.get("global_prompt", "natural makeup, highly detailed"),
        safe("foundation", "color", "#FFE4C4"), float(safe("foundation", "alpha", 0.0)), safe("foundation", "prompt", ""),
        safe("lips",       "color", "#D9334D"), float(safe("lips",       "alpha", 0.0)), safe("lips",       "prompt", ""),
        safe("blush",      "color", "#E69999"), float(safe("blush",      "alpha", 0.0)), float(safe("blush", "size", 1.0)), safe("blush", "prompt", ""),
        safe("eyeshadow",  "color", "#CC7359"), float(safe("eyeshadow",  "alpha", 0.0)), float(safe("eyeshadow", "size", 1.0)), safe("eyeshadow", "prompt", ""),
        safe("brows",      "color", "#4D331A"), float(safe("brows",      "alpha", 0.0)), safe("brows",      "prompt", ""),
        summary_md,
        raw_json,
    )

# ==========================================
# 3. 核心生成函式
# ==========================================
def generate_custom_makeup(
    input_image,
    global_prompt, lambda_interp, steps,
    found_color_hex, found_alpha, found_prompt,
    lip_color_hex, lip_alpha, lip_prompt,
    blush_color_hex, blush_alpha, blush_size, blush_prompt,
    eye_color_hex, eye_alpha, eye_size, eye_prompt,
    brow_color_hex, brow_alpha, brow_prompt
):
    pipe, net = load_models()

    if input_image is None: raise gr.Error("請上傳圖片")

    # 1. 圖片前處理（center crop → 512×512，避免非正方形輸入造成臉部變形）
    source_pil = preprocess_image(input_image)
    source_np = np.array(source_pil) / 255.0

    # 2. 取得分割並視覺化
    print("Generating masks...")
    masks = get_face_masks(source_pil, net, device=DEVICE)
    parsing = masks['full_parse']
    vis_parsing_np = visualize_parsing_maps(np.array(source_pil), parsing)
    vis_parsing_pil = Image.fromarray(vis_parsing_np)

    # 3. 取得進階遮罩（腮紅範圍隨 blush_size 彈性縮放）
    adv_masks = get_advanced_masks(parsing, 512, 512, blush_size=blush_size)

    print("Applying Makeup...")
    current_image = source_np.copy()

    # 顏色轉換
    found_rgb = hex_to_rgb(found_color_hex)
    lip_rgb = hex_to_rgb(lip_color_hex)
    blush_rgb = hex_to_rgb(blush_color_hex)
    eye_rgb = hex_to_rgb(eye_color_hex)
    brow_rgb = hex_to_rgb(brow_color_hex)

    # --- Makeup Application Steps ---
    if found_alpha > 0:
        current_image = apply_makeup_natural(
            current_image, adv_masks['skin'], found_rgb, found_alpha, update_l_channel=True
        )

    if brow_alpha > 0:
        current_image = apply_makeup_natural(
            current_image, gradation_smoothing(adv_masks['brows'], (9, 9)),
            brow_rgb, brow_alpha, update_l_channel=False
        )

    if eye_alpha > 0:
        eyeshadow_mask_gradient = generate_realistic_eyeshadow(parsing, 512, 512, size_scale=eye_size)
        current_image = apply_makeup_natural(
            current_image, eyeshadow_mask_gradient, eye_rgb, eye_alpha, update_l_channel=False
        )

    if blush_alpha > 0:
        current_image = apply_makeup_natural(
            current_image, adv_masks['blush'], blush_rgb, blush_alpha, update_l_channel=False
        )

    if lip_alpha > 0:
        current_image = apply_makeup_natural(
            current_image, gradation_smoothing(adv_masks['lips'], (5, 5), iterations=0),
            lip_rgb, lip_alpha, update_l_channel=False
        )

    guide_pil = Image.fromarray((current_image * 255).astype(np.uint8))

    # 4. 生成
    parts_prompts = []
    if found_prompt: parts_prompts.append(found_prompt)
    if lip_prompt: parts_prompts.append(lip_prompt)
    if blush_prompt: parts_prompts.append(blush_prompt)
    if eye_prompt: parts_prompts.append(eye_prompt)
    if brow_prompt: parts_prompts.append(brow_prompt)
    
    full_prompt = f"{global_prompt}, {', '.join(parts_prompts)}"
    print(f"Final Prompt: {full_prompt}")

    print("Running diffusion inference...")
    with torch.no_grad():
        target_device = pipe.device
        pipeline_dtype = pipe.unet.dtype if hasattr(pipe, 'unet') else torch.float32
        
        src_tensor = pipe.image_processor.preprocess(source_pil).to(target_device, dtype=pipeline_dtype)
        makeup_tensor = pipe.image_processor.preprocess(guide_pil).to(target_device, dtype=pipeline_dtype)
        
        z_prime_0 = pipe.vae.encode(makeup_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
        
        output = pipe(
            source_image=src_tensor,
            masked_image_latents=z_prime_0,
            prompt=full_prompt,
            negative_prompt="blurry, ugly, bad anatomy",
            num_inference_steps=steps,
            early_stop_steps=250,
            lambda_interp=lambda_interp
        )

    result_image = output.images[0]
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vis_parsing_pil, guide_pil, result_image

# ==========================================
# 4. Gradio UI 介面
# ==========================================
custom_css = """#col-container {max-width: 1200px; margin: auto;}"""

with gr.Blocks(css=custom_css, title="DreamMakeup Visualizer") as demo:
    gr.Markdown("# 🎨 DreamMakeup + LLM Agent")
    
    # ---------------- LLM 指令區塊 ----------------
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🪄 一鍵 AI 魔法妝容")
            gr.Markdown("輸入妝容描述，LLM 自動解析並填入下方所有參數。")
            with gr.Row():
                magic_prompt = gr.Textbox(
                    label="",
                    show_label=False,
                    placeholder="例如：幫我畫一個微醺感的日系桃花妝，口紅要水潤的蜜桃色，眼影帶點珠光粉...",
                    lines=2,
                    scale=4,
                )
                magic_btn = gr.Button("✨ LLM 解析", variant="secondary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### 📊 LLM 分析結果")
            llm_summary = gr.Markdown(
                value="*點擊「LLM 解析」後，這裡會顯示解析摘要。*",
                label="",
            )

    with gr.Accordion("🔍 原始 JSON（完整參數）", open=False):
        llm_json_out = gr.Code(language="json", label="LLM 回傳 JSON", lines=15)

    gr.HTML("<hr>")
    
    # ---------------- 主控台區塊 ----------------
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="原始圖片", type="numpy", height=300)
            global_prompt = gr.Textbox(label="全域風格", value="Korean style, natural makeup")
            
            with gr.Row():
                lambda_slider = gr.Slider(0.0, 1.0, value=0.15, label="引導權重")
                steps_slider = gr.Slider(10, 50, value=30, step=1, label="步數")
            
            with gr.Accordion("💄 手動微調 (由 LLM 自動填入或手動修改)", open=True):
                with gr.Accordion("✨ 底妝 (Foundation)", open=False):
                    with gr.Row():
                        found_color = gr.ColorPicker(label="膚色調整 (RGB)", value="#FFE4C4")
                        found_alpha = gr.Slider(0.0, 2.0, value=0.0, label="濃度")
                        found_prompt = gr.Textbox(label="膚質描述", value="fair skin, flawless skin, matte finish")

                with gr.Accordion("👄 口紅 (Lips)", open=True):
                    with gr.Row():
                        lip_color = gr.ColorPicker(label="顏色 (RGB)", value="#D9334D")
                        lip_alpha = gr.Slider(0.0, 2.0, value=0.6, label="濃度")
                        lip_prompt = gr.Textbox(label="質感描述", value="gradient lips, bitten lips")

                with gr.Accordion("😊 腮紅 (Blush)", open=False):
                    with gr.Row():
                        blush_color = gr.ColorPicker(label="顏色 (RGB)", value="#E69999")
                        blush_alpha = gr.Slider(0.0, 2.0, value=0.3, label="濃度")
                        blush_size = gr.Slider(0.4, 2.5, value=1.0, step=0.1, label="範圍 (0.4=小範圍, 2.5=大範圍打亮)")
                        blush_prompt = gr.Textbox(label="質感描述", value="soft pink blush")

                with gr.Accordion("👁️ 眼影 (Eyeshadow)", open=False):
                    with gr.Row():
                        eye_color = gr.ColorPicker(label="顏色 (RGB)", value="#CC7359")
                        eye_alpha = gr.Slider(0.0, 2.0, value=0.4, label="濃度")
                        eye_size = gr.Slider(0.4, 2.5, value=1.0, step=0.1, label="範圍 (0.4=眼頭提亮, 2.5=大範圍煙燻)")
                        eye_prompt = gr.Textbox(label="質感描述", value="soft coral eyeshadow")
                
                with gr.Accordion("🤨 眉毛 (Brows)", open=False):
                    with gr.Row():
                        brow_color = gr.ColorPicker(label="顏色 (RGB)", value="#4D331A")
                        brow_alpha = gr.Slider(0.0, 2.0, value=0.3, label="濃度")
                        brow_prompt = gr.Textbox(label="質感描述", value="natural straight brows")

            run_btn = gr.Button("🎨 開始分析與生成", variant="primary")

        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("1. 最終結果"):
                    result_img = gr.Image(label="Diffusion 生成結果", type="pil")
                with gr.TabItem("2. 妝容引導圖"):
                    guide_img = gr.Image(label="上妝引導", type="pil")
                with gr.TabItem("3. 語意分割圖"):
                    seg_img = gr.Image(label="BiSeNet 解析結果", type="pil")

    # ---------------- 事件綁定 ----------------
    
    # 1. 綁定 LLM 解析按鈕
    magic_btn.click(
        fn=parse_makeup_with_llm,
        inputs=[magic_prompt],
        outputs=[
            global_prompt,
            found_color, found_alpha, found_prompt,
            lip_color, lip_alpha, lip_prompt,
            blush_color, blush_alpha, blush_size, blush_prompt,
            eye_color, eye_alpha, eye_size, eye_prompt,
            brow_color, brow_alpha, brow_prompt,
            llm_summary,
            llm_json_out,
        ]
    )

    # 2. 綁定 Diffusion 生成按鈕
    run_btn.click(
        fn=generate_custom_makeup,
        inputs=[
            input_img, global_prompt, lambda_slider, steps_slider,
            found_color, found_alpha, found_prompt,
            lip_color, lip_alpha, lip_prompt,
            blush_color, blush_alpha, blush_size, blush_prompt,
            eye_color, eye_alpha, eye_size, eye_prompt,
            brow_color, brow_alpha, brow_prompt
        ],
        outputs=[seg_img, guide_img, result_img]
    )
    
if __name__ == "__main__":
    print("🚀 正在啟動 Gradio 伺服器...")
    demo.queue().launch(
        share=True,
        server_name="0.0.0.0",           # 綁定所有介面，方便從計算節點外經 SSH tunnel 連入
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,                 # 錯誤訊息直接顯示在 UI 上
        prevent_thread_lock=False,       # 阻塞主執行緒，讓 job script 保持存活直到 job 被砍
    )