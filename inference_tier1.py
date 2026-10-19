# -*- coding: utf-8 -*-
"""
Tier1 CLI 推論:LLM hidden states + Projector 條件化的 DreamMakeup。

與 inference.py 的差異:
- 顏色精準路徑完全相同:Ollama JSON → BiSeNet 分割 → LAB 混色 → guide 圖 → z'_0。
- 風格/質感路徑改走 Tier1:使用者的「原始描述文字」直接經本地 LLM 取 hidden
  states,交給訓練好的 Projector 產生條件向量,不再壓縮成英文關鍵字給 CLIP。
  這正是 Tier1 的核心:條件訊號不經過「JSON prompt 欄位」這一手壓縮。

用法範例:
    python inference_tier1.py -i assets/input.jpg \
        -p "幫我畫微醺感的日系桃花妝,口紅要水潤的蜜桃色" \
        --projector checkpoints/tier1/projector_step20000.pt \
        --unet_lora checkpoints/tier1/unet_lora_step20000.pt
"""

import argparse
import gc
import json
import os

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pipeline_dreammakeup_tier1 import Tier1DreamMakeupPipeline
from models.projector import MakeupProjector
from utils.color import apply_makeup_natural
from utils.mask import gradation_smoothing
from utils.segmentation import (
    load_bisenet, get_face_masks, get_advanced_masks,
    generate_realistic_eyeshadow, visualize_parsing_maps,
)
# 顏色路徑沿用 inference.py 的既有實作
from inference import parse_makeup_with_llm, preprocess_image, hex_to_rgb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
WEIGHTS_PATH = "assets/79999_iter.pth"


def encode_style_text(style_text: str, llm_path: str, max_length: int):
    """本地 LLM 取 hidden states(條件 + 空字串無條件),完成後立即釋放 LLM。"""
    from models.llm_encoder import LLMTextEncoder

    encoder = LLMTextEncoder(model_path=llm_path, device=DEVICE, max_length=max_length)
    cond_hidden, cond_mask = encoder.encode(style_text)
    uncond_hidden, uncond_mask = encoder.encode("")
    encoder.free()
    gc.collect()
    return cond_hidden, cond_mask, uncond_hidden, uncond_mask


def load_tier1_pipeline(projector_ckpt: str, unet_lora: str = None):
    print("Loading Tier1 Diffusion pipeline (FP16)...")
    try:
        pipe = Tier1DreamMakeupPipeline.from_pretrained(
            MODEL_ID, safety_checker=None, variant="fp16", torch_dtype=torch.float16,
        )
    except Exception as e:
        print(f"FP16 variant load failed, trying FP32: {e}")
        pipe = Tier1DreamMakeupPipeline.from_pretrained(
            MODEL_ID, safety_checker=None, torch_dtype=torch.float32,
        )

    if DEVICE == "cuda":
        pipe = pipe.to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # 載入訓練好的 Projector(fp32 運算)
    ckpt = torch.load(projector_ckpt, map_location=DEVICE)
    projector = MakeupProjector(**ckpt["config"]).to(DEVICE)
    projector.load_state_dict(ckpt["state_dict"])
    projector.eval()
    pipe.set_projector(projector)
    print(f"✅ Projector loaded ({projector_ckpt}, step {ckpt.get('step', '?')})")

    # 載入配套訓練的 UNet LoRA(train_projector.py --unet_lora 的產物)
    if unet_lora and os.path.exists(unet_lora):
        from peft import LoraConfig

        lora_sd = torch.load(unet_lora, map_location=DEVICE)
        rank = next(v.shape[0] for k, v in lora_sd.items() if "lora_A" in k)
        pipe.unet.add_adapter(LoraConfig(
            r=rank, lora_alpha=rank,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        ))
        missing, unexpected = pipe.unet.load_state_dict(lora_sd, strict=False)
        pipe.unet.to(dtype=pipe.unet.dtype)
        print(f"✅ UNet LoRA loaded ({unet_lora})")

    return pipe


def main():
    parser = argparse.ArgumentParser(description="Tier1 DreamMakeup CLI Inference")
    parser.add_argument("-i", "--image",      required=True, help="輸入圖片路徑")
    parser.add_argument("-p", "--prompt",     default="",    help="妝容自然語言描述(同時供 Ollama JSON 與 Tier1 條件化)")
    parser.add_argument("-j", "--json",       default="",    help="JSON 參數檔(跳過 Ollama;需搭配 --style_text)")
    parser.add_argument("--style_text",       default="",    help="覆寫 Tier1 條件文字(預設用 -p 原文)")
    parser.add_argument("--projector",        required=True, help="訓練好的 projector checkpoint (.pt)")
    parser.add_argument("--unet_lora",        default="",    help="配套的 UNet LoRA checkpoint (.pt)")
    parser.add_argument("--llm_path",         default=None,  help="LLM 權重路徑(預設環境變數 LLM_ENCODER_PATH)")
    parser.add_argument("--max_length",       type=int, default=128)
    parser.add_argument("-o", "--output_dir", default="results_tier1")
    parser.add_argument("--steps",            type=int,   default=30)
    parser.add_argument("--lambda_interp",    type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 妝容參數(顏色路徑用,與 inference.py 完全相同)
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

    # 2. Tier1 條件文字:優先用使用者原始描述,不經 JSON 壓縮
    style_text = args.style_text or args.prompt
    if not style_text:
        parts = [makeup_params[k]["prompt"] for k in
                 ["foundation", "lips", "blush", "eyeshadow", "brows"]
                 if makeup_params[k]["prompt"]]
        style_text = makeup_params["global_prompt"] + ", " + ", ".join(parts)
    print(f"[Tier1] 條件文字: {style_text}")

    # 3. LLM hidden states(取完即釋放 LLM,再載 SD)
    cond_hidden, cond_mask, uncond_hidden, uncond_mask = encode_style_text(
        style_text, args.llm_path, args.max_length
    )

    # 4. 載入 BiSeNet 與 Tier1 pipeline
    net = load_bisenet(WEIGHTS_PATH, device=DEVICE)
    print("✅ BiSeNet loaded.")
    pipe = load_tier1_pipeline(args.projector, args.unet_lora)

    # 5. 前處理 + 分割(與 inference.py 相同)
    print(f"[Image] 載入: {args.image}")
    source_pil = preprocess_image(args.image)
    source_np = np.array(source_pil) / 255.0

    print("[Seg] 產生遮罩...")
    masks = get_face_masks(source_pil, net, device=DEVICE)
    parsing = masks["full_parse"]
    Image.fromarray(visualize_parsing_maps(np.array(source_pil), parsing)).save(
        os.path.join(args.output_dir, "1_segmentation_map.png"))

    # 6. LAB 上妝引導圖(顏色精準路徑,與 inference.py 完全相同)
    print("[Makeup] 套用妝容...")
    adv_masks = get_advanced_masks(parsing, 512, 512)
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

    # 7. Tier1 Diffusion 生成
    print("[Diffusion] Tier1 生成中...")
    with torch.no_grad():
        dtype = pipe.unet.dtype
        src_tensor    = pipe.image_processor.preprocess(source_pil).to(DEVICE, dtype=dtype)
        makeup_tensor = pipe.image_processor.preprocess(guide_pil).to(DEVICE, dtype=dtype)
        z_prime_0     = pipe.vae.encode(makeup_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor

        output = pipe(
            source_image=src_tensor,
            masked_image_latents=z_prime_0,
            cond_hidden=cond_hidden, cond_mask=cond_mask,
            uncond_hidden=uncond_hidden, uncond_mask=uncond_mask,
            num_inference_steps=args.steps,
            early_stop_steps=250,
            lambda_interp=args.lambda_interp,
        )

    output.images[0].save(os.path.join(args.output_dir, "3_final_result.png"))
    print(f"✅ 完成!結果在 {args.output_dir}/")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
