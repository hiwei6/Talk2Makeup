# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import cv2
import os
import gc
import re
import json
import urllib.request
from PIL import Image, ImageColor
from diffusers import DDIMScheduler
from pipeline_dreammakeup import DreamMakeupPipeline
from openai import OpenAI

# 切換到腳本所在目錄，確保相對路徑正確
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils.color import apply_makeup_natural
from utils.mask import gradation_smoothing
from models.bisenet import BiSeNet
from utils.segmentation import (
    load_bisenet, get_face_masks, get_advanced_masks,
    generate_realistic_eyeshadow, visualize_parsing_maps
)

try:
    from huggingface_hub.utils import disable_progress_bars
except ImportError:
    from huggingface_hub import disable_progress_bars

# ==========================================
# 設定
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "assets/BKG1.safetensors"
WEIGHTS_PATH = "assets/79999_iter.pth"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")

# 模型從 NFS 冷載約需 3 分鐘，timeout 設 400 秒
LLM_CLIENT = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
    timeout=400.0,
)

# ==========================================
# LLM 解析
# ==========================================
SYSTEM_PROMPT = """You are a professional makeup artist AI. Convert the user's makeup description into a JSON object.

STRICT RULES:
- Output ONLY valid JSON. No explanation, no markdown, no code blocks.
- "color" must be a 6-digit hex code like "#D9334D".
- "alpha" controls opacity: 0.0=skip, 0.3-0.6=natural, 0.8-1.5=heavy. Set alpha>0 for any mentioned part.
- "prompt" is an English Stable Diffusion keyword (e.g. "glossy red lips"). Empty string if not mentioned.
- "global_prompt" is English, describes overall style.
- "reason" explains WHY you chose this color/alpha for this part, in Traditional Chinese (繁體中文), 1-2 sentences.
- "global_reason" explains the overall design intent in Traditional Chinese, 1-2 sentences.

OUTPUT FORMAT (fill every field, use alpha=0.0 only for truly unmentioned parts):
{"global_prompt":"...","global_reason":"...","foundation":{"color":"#HEX","alpha":0.0,"prompt":"","reason":"..."},"lips":{"color":"#HEX","alpha":0.5,"prompt":"...","reason":"..."},"blush":{"color":"#HEX","alpha":0.3,"prompt":"...","reason":"..."},"eyeshadow":{"color":"#HEX","alpha":0.4,"prompt":"...","reason":"..."},"brows":{"color":"#HEX","alpha":0.3,"prompt":"...","reason":"..."}}

EXAMPLE — input "natural Korean look, pink lips, soft blush":
{"global_prompt":"Korean natural makeup, soft and dewy","global_reason":"整體走韓系清透路線，以低飽和度色系和輕薄妝感呈現自然水潤的氛圍。","foundation":{"color":"#FFE4C4","alpha":0.0,"prompt":"","reason":"使用者未提到底妝，維持原膚色以保留自然感。"},"lips":{"color":"#FFB6C1","alpha":0.5,"prompt":"soft pink glossy lips","reason":"粉嫩玫瑰色呼應韓系少女感，中等濃度保持水潤透明質感。"},"blush":{"color":"#F4A0A0","alpha":0.35,"prompt":"soft pink blush","reason":"淡粉色腮紅以低濃度暈染，製造自然的血色感與水光肌效果。"},"eyeshadow":{"color":"#C8A882","alpha":0.3,"prompt":"neutral eyeshadow","reason":"米棕色眼影淡掃加深眼神層次，不搶眼但增加立體感。"},"brows":{"color":"#5C3D1E","alpha":0.3,"prompt":"natural soft brows","reason":"深棕色描繪平直眉，符合韓系妝容的標誌性眉型。"}}
"""

def _check_ollama():
    """確認 Ollama 可連線，若不行就提前報錯。"""
    url = OLLAMA_BASE_URL.replace("/v1", "")
    try:
        urllib.request.urlopen(url, timeout=5)
    except Exception as e:
        raise RuntimeError(
            f"無法連線到 Ollama ({url})。\n"
            f"請先執行：OLLAMA_MODELS=~/.ollama/models ~/bin/ollama serve\n"
            f"原始錯誤：{e}"
        )

def parse_makeup_with_llm(user_input: str) -> dict:
    if not user_input.strip():
        raise ValueError("請輸入妝容描述！")

    _check_ollama()

    print(f"[LLM] 正在解析: {user_input}")
    print("[LLM] 若模型尚未載入，首次呼叫需等待約 3 分鐘（NFS 冷載），請耐心等候...")
    try:
        response = LLM_CLIENT.chat.completions.create(
            model=OLLAMA_MODEL,
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
        raise ValueError(f"LLM 解析失敗：{e}")

    def safe(part, key, default):
        return data.get(part, {}).get(key, default)

    return {
        "global_prompt": data.get("global_prompt", "natural makeup, highly detailed"),
        "foundation": {
            "color":  safe("foundation", "color",  "#FFE4C4"),
            "alpha":  float(safe("foundation", "alpha",  0.0)),
            "prompt": safe("foundation", "prompt", ""),
        },
        "lips": {
            "color":  safe("lips", "color",  "#D9334D"),
            "alpha":  float(safe("lips", "alpha",  0.0)),
            "prompt": safe("lips", "prompt", ""),
        },
        "blush": {
            "color":  safe("blush", "color",  "#E69999"),
            "alpha":  float(safe("blush", "alpha",  0.0)),
            "prompt": safe("blush", "prompt", ""),
        },
        "eyeshadow": {
            "color":  safe("eyeshadow", "color",  "#CC7359"),
            "alpha":  float(safe("eyeshadow", "alpha",  0.0)),
            "prompt": safe("eyeshadow", "prompt", ""),
        },
        "brows": {
            "color":  safe("brows", "color",  "#4D331A"),
            "alpha":  float(safe("brows", "alpha",  0.0)),
            "prompt": safe("brows", "prompt", ""),
        },
    }

# ==========================================
# 工具函式
# ==========================================
def preprocess_image(path: str) -> Image.Image:
    """Letterbox to square with reflect padding, then resize to 512×512.
    Preserves the full original image without any cropping.
    Reflect padding avoids hard black/white edges that could confuse BiSeNet or diffusion.
    """
    arr = np.array(Image.open(path).convert("RGB"))
    h, w = arr.shape[:2]
    long = max(w, h)
    pad_top    = (long - h) // 2
    pad_bottom = long - h - pad_top
    pad_left   = (long - w) // 2
    pad_right  = long - w - pad_left
    arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=255)
    return Image.fromarray(arr).resize((512, 512), Image.LANCZOS)

def hex_to_rgb(color_str):
    if not color_str:
        return [0, 0, 0]
    if color_str.startswith("#"):
        try:
            return list(ImageColor.getcolor(color_str, "RGB"))
        except Exception:
            return [0, 0, 0]
    if "rgb" in color_str:
        try:
            nums = re.findall(r"[\d\.]+", color_str)
            if len(nums) >= 3:
                return [int(float(n)) for n in nums[:3]]
        except Exception:
            pass
    return [0, 0, 0]

# ==========================================
# 模型載入
# ==========================================
def load_models():
    print(f"[{DEVICE}] 正在載入模型...")
    disable_progress_bars()

    net = load_bisenet(WEIGHTS_PATH, device=DEVICE)
    print("✅ BiSeNet loaded.")

    print("Loading Diffusion pipeline (FP16)...")
    try:
        pipe = DreamMakeupPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=None,
            variant="fp16",
            torch_dtype=torch.float16,
        )
    except Exception as e:
        print(f"FP16 variant load failed, trying FP32: {e}")
        pipe = DreamMakeupPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=None,
            torch_dtype=torch.float32,
        )

    if DEVICE == "cuda":
        pipe = pipe.to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if os.path.exists(LORA_PATH):
        try:
            pipe.load_lora_weights(".", weight_name=LORA_PATH)
            pipe.fuse_lora(lora_scale=0.2)
            print("✅ LoRA loaded.")
        except Exception as e:
            print(f"⚠️ LoRA 載入失敗 (已忽略): {e}")

    return pipe, net

# ==========================================
# 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="DreamMakeup CLI Inference")
    parser.add_argument("-i", "--image",       required=True,  help="輸入圖片路徑")
    parser.add_argument("-p", "--prompt",      default="",     help="妝容自然語言描述 (呼叫 LLM)")
    parser.add_argument("-j", "--json",        default="",     help="直接讀取 JSON 參數檔 (跳過 LLM)")
    parser.add_argument("-o", "--output_dir",  default="results", help="輸出資料夾")
    parser.add_argument("--steps",             type=int,   default=30,   help="Diffusion 步數")
    parser.add_argument("--lambda_interp",     type=float, default=0.15, help="引導權重")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 取得妝容參數（LLM 或 JSON）
    if args.json:
        print(f"[JSON] 讀取: {args.json}")
        with open(args.json, "r", encoding="utf-8") as f:
            makeup_params = json.load(f)
    elif args.prompt:
        makeup_params = parse_makeup_with_llm(args.prompt)
        out_path = os.path.join(args.output_dir, "0_makeup_params.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(makeup_params, f, ensure_ascii=False, indent=2)
        print(f"[JSON] 已儲存至 {out_path}")
    else:
        parser.error("請提供 -p (文字描述) 或 -j (JSON 檔案) 其中一種")

    # 2. 載入模型
    pipe, net = load_models()

    # 3. 圖片前處理（center crop → 512×512，避免非正方形輸入造成臉部變形）
    print(f"[Image] 載入: {args.image}")
    source_pil = preprocess_image(args.image)
    source_np  = np.array(source_pil) / 255.0

    # 4. 人臉分割
    print("[Seg] 產生遮罩...")
    masks   = get_face_masks(source_pil, net, device=DEVICE)
    parsing = masks["full_parse"]
    vis_parsing_pil = Image.fromarray(visualize_parsing_maps(np.array(source_pil), parsing))
    vis_parsing_pil.save(os.path.join(args.output_dir, "1_segmentation_map.png"))

    # 5. 上妝引導圖
    print("[Makeup] 套用妝容...")
    adv_masks     = get_advanced_masks(parsing, 512, 512)
    current_image = source_np.copy()

    p = makeup_params
    if p["foundation"]["alpha"] > 0:
        current_image = apply_makeup_natural(
            current_image, adv_masks["skin"],
            hex_to_rgb(p["foundation"]["color"]),
            p["foundation"]["alpha"], update_l_channel=True,
        )
    if p["brows"]["alpha"] > 0:
        current_image = apply_makeup_natural(
            current_image, gradation_smoothing(adv_masks["brows"], (9, 9)),
            hex_to_rgb(p["brows"]["color"]),
            p["brows"]["alpha"], update_l_channel=False,
        )
    if p["eyeshadow"]["alpha"] > 0:
        current_image = apply_makeup_natural(
            current_image, generate_realistic_eyeshadow(parsing, 512, 512),
            hex_to_rgb(p["eyeshadow"]["color"]),
            p["eyeshadow"]["alpha"], update_l_channel=False,
        )
    if p["blush"]["alpha"] > 0:
        current_image = apply_makeup_natural(
            current_image, adv_masks["blush"],
            hex_to_rgb(p["blush"]["color"]),
            p["blush"]["alpha"], update_l_channel=False,
        )
    if p["lips"]["alpha"] > 0:
        current_image = apply_makeup_natural(
            current_image, gradation_smoothing(adv_masks["lips"], (5, 5), iterations=0),
            hex_to_rgb(p["lips"]["color"]),
            p["lips"]["alpha"], update_l_channel=False,
        )

    guide_pil = Image.fromarray((current_image * 255).astype(np.uint8))
    guide_pil.save(os.path.join(args.output_dir, "2_makeup_guide.png"))

    # 6. 組裝 prompt
    parts = [p[part]["prompt"] for part in ["foundation", "lips", "blush", "eyeshadow", "brows"] if p[part]["prompt"]]
    full_prompt = p["global_prompt"]
    if parts:
        full_prompt += ", " + ", ".join(parts)
    print(f"[Prompt] {full_prompt}")

    # 7. Diffusion 生成
    print("[Diffusion] 生成中...")
    with torch.no_grad():
        dtype  = pipe.unet.dtype if hasattr(pipe, "unet") else torch.float32
        device = pipe.device
        src_tensor    = pipe.image_processor.preprocess(source_pil).to(device, dtype=dtype)
        makeup_tensor = pipe.image_processor.preprocess(guide_pil).to(device, dtype=dtype)
        z_prime_0     = pipe.vae.encode(makeup_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
        output = pipe(
            source_image=src_tensor,
            masked_image_latents=z_prime_0,
            prompt=full_prompt,
            negative_prompt="blurry, ugly, bad anatomy",
            num_inference_steps=args.steps,
            early_stop_steps=250,
            lambda_interp=args.lambda_interp,
        )

    output.images[0].save(os.path.join(args.output_dir, "3_final_result.png"))
    print(f"✅ 完成！結果在 {args.output_dir}/")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
