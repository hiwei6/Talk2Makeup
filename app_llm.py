import gradio as gr
import torch
import numpy as np
import json
import os
from PIL import Image
from diffusers import DDIMScheduler
from pipeline_dreammakeup import DreamMakeupPipeline

from utils.color import apply_makeup_natural
from utils.mask import gradation_smoothing
from utils.segmentation import load_bisenet, get_face_masks, get_advanced_masks, get_hair_mask, get_messy_mask, generate_dynamic_blush
from llm_processor import get_makeup_plan_from_llm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
global_models = {"pipe": None, "net": None}

def load_models():
    if global_models["pipe"] is not None: return global_models["pipe"], global_models["net"]
    print("Loading models...")
    net = load_bisenet("assets/79999_iter.pth", device=DEVICE)
    pipe = DreamMakeupPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    global_models["pipe"], global_models["net"] = pipe, net
    return pipe, net

def process_smart_makeup(input_image, user_prompt):
    pipe, net = load_models()
    if input_image is None or not user_prompt: raise gr.Error("請提供圖片與文字指令！")

    # 1. 呼叫 LLM 獲取計畫
    print(f"詢問 LLM: {user_prompt}")
    plan = get_makeup_plan_from_llm(user_prompt)
    
    # 2. 圖片前處理與遮罩解析
    source_pil = Image.fromarray(input_image).convert("RGB").resize((512, 512))
    source_np = np.array(source_pil) / 255.0
    h, w = 512, 512
    masks = get_face_masks(source_pil, net, device=DEVICE)
    parsing = masks['full_parse']
    adv_masks = get_advanced_masks(parsing, h, w)

    # 3. 根據 LLM 計畫動態繪製導引圖
    current_image = source_np.copy()
    
    # 粉底 (Foundation)
    if "foundation" in plan and plan["foundation"]["alpha"] > 0:
        skin_mask = adv_masks.get('skin', np.where(parsing == 1, 1.0, 0.0))
        current_image = apply_makeup_natural(
            current_image, skin_mask, plan["foundation"]["color"], plan["foundation"]["alpha"], 
            update_l_channel=plan["foundation"].get("force_lightness", False)
        )
        
    # 頭髮 (Hair)
    if "hair" in plan and plan["hair"]["alpha"] > 0:
        current_image = apply_makeup_natural(
            current_image, get_hair_mask(parsing, h, w), plan["hair"]["color"], plan["hair"]["alpha"]
        )

    # 眼妝 (Eyes)
    if "eyes" in plan and plan["eyes"]["alpha"] > 0:
        dilation = plan["eyes"].get("dilation", 0)
        raw_eye = masks['eyes']
        eye_mask = get_messy_mask(raw_eye, intensity=dilation) if dilation > 0 else gradation_smoothing(raw_eye, (15, 15))
        current_image = apply_makeup_natural(
            current_image, eye_mask, plan["eyes"]["color"], plan["eyes"]["alpha"], update_l_channel=(dilation > 0)
        )

    # 腮紅 (Blush) - 呼叫動態引擎
    if "blush" in plan and plan["blush"]["alpha"] > 0:
        blush_mask = generate_dynamic_blush(
            parsing, h, w, 
            placement=plan["blush"].get("placement", "apple"), 
            scale=plan["blush"].get("scale", 1.0)
        )
        current_image = apply_makeup_natural(
            current_image, blush_mask, plan["blush"]["color"], plan["blush"]["alpha"], use_soft_light=True
        )

    # 嘴唇 (Lips)
    if "lips" in plan and plan["lips"]["alpha"] > 0:
        dilation = plan["lips"].get("dilation", 0)
        lip_mask = get_messy_mask(masks['lips'], intensity=dilation) if dilation > 0 else adv_masks['lips']
        current_image = apply_makeup_natural(
            current_image, lip_mask, plan["lips"]["color"], plan["lips"]["alpha"]
        )

    guide_pil = Image.fromarray((current_image * 255).astype(np.uint8))

    # 4. 擴散模型生成
    sd_prompt = plan.get("description", user_prompt)
    print(f"SD Prompt: {sd_prompt}")
    
    with torch.no_grad():
        src_tensor = pipe.image_processor.preprocess(source_pil).to(DEVICE)
        makeup_tensor = pipe.image_processor.preprocess(guide_pil).to(DEVICE)
        if DEVICE == "cuda": src_tensor, makeup_tensor = src_tensor.half(), makeup_tensor.half()
        z_prime_0 = pipe.vae.encode(makeup_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor

    output = pipe(
        source_image=src_tensor, masked_image_latents=z_prime_0,
        prompt=sd_prompt, negative_prompt="blurry, ugly, distortion",
        num_inference_steps=30, early_stop_steps=250, lambda_interp=0.15
    )
    
    # 格式化 JSON 以利顯示
    json_str = json.dumps(plan, indent=2, ensure_ascii=False)
    return guide_pil, output.images[0], json_str

# --- Gradio UI ---
with gr.Blocks(title="LLM DreamMakeup") as demo:
    gr.Markdown("# 🧠 LLM-Driven DreamMakeup\n輸入自然語言指令，AI 自動為您訂製專屬妝容！")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="上傳人臉照片", type="numpy")
            user_prompt = gr.Textbox(label="你想要什麼妝容？", placeholder="例如：萬聖節的小丑妝 / 日系曬傷微醺妝 / 阿凡達藍色皮膚", lines=2)
            run_btn = gr.Button("✨ AI 一鍵試妝", variant="primary")
        
        with gr.Column():
            result_img = gr.Image(label="最終生成結果", type="pil")
            
    with gr.Row():
        with gr.Accordion("🔍 系統運作細節 (給教授看的)", open=False):
            with gr.Row():
                guide_img = gr.Image(label="OpenCV 動態導引圖", type="pil")
                json_out = gr.Code(label="LLM 解析參數 (Semantic-to-Spatial)", language="json")

    run_btn.click(process_smart_makeup, inputs=[input_img, user_prompt], outputs=[guide_img, result_img, json_out])
    demo.load(load_models)

if __name__ == "__main__":
    demo.queue().launch(share=True)