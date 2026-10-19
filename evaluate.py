#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py — Talk2Makeup 量化評估工具
========================================

計算以下七項指標：
  ① CAI         色彩準確性（LLM 指定 Hex vs. 生成區域實際色，ΔE₀₀ 越小越好）
  ② ID-CLIP     語意身份一致性（source vs. generated，CLIP 餘弦相似度，越高越好）
  ③ Text-CLIP   全域文字對齊（global_prompt vs. generated，越高越好）
  ④ Region-CLIP 逐區域文字對齊（region_prompt vs. 裁剪區域，越高越好）
  ⑤ SSIM        像素結構相似性（source vs. generated，越高越好）
  ⑥ LPIPS       深層感知相似性（source vs. generated，越低越好）
  ⑦ ID-ArcFace  嚴格身份辨識相似性（source vs. generated，越高越好）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
快速開始（不需安裝額外套件，先跑 SSIM）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python evaluate.py \\
    --source    assets/input.jpg \\
    --generated results/3_final_result.png \\
    --params    results/0_makeup_params.json \\
    --metrics   cai ssim

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
完整評估（需安裝所有依賴）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install scikit-image lpips tabulate
  pip install git+https://github.com/openai/CLIP.git
  pip install insightface onnxruntime-gpu   # ArcFace 用，可用 onnxruntime 替代

  python evaluate.py \\
    --source    assets/input.jpg \\
    --generated results/3_final_result.png \\
    --params    results/0_makeup_params.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
批次評估（一次跑多張）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 建立 CSV（每行：source,generated,params）：
  # assets/img1.jpg,results_img1/3_final_result.png,results_img1/0_makeup_params.json

  python evaluate.py \\
    --batch_csv eval_list.csv \\
    --output_json results/eval_batch.json
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageColor

# 切到腳本目錄，確保 utils/models 的 import 路徑正確
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.segmentation import (
    load_bisenet,
    get_face_masks,
    get_advanced_masks,
    generate_realistic_eyeshadow,
)

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALL_METRICS = ["cai", "id_clip", "text_clip", "region_clip", "ssim", "lpips", "arcface"]


# ══════════════════════════════════════════════════════════════════════════════
# 共用工具函式
# ══════════════════════════════════════════════════════════════════════════════

def hex_to_rgb(hex_str: str) -> tuple:
    """'#RRGGBB' -> (R, G, B) uint8 三元組"""
    hex_str = hex_str.strip()
    if not hex_str.startswith("#"):
        hex_str = "#" + hex_str
    try:
        return ImageColor.getcolor(hex_str, "RGB")
    except Exception:
        return (0, 0, 0)


def load_image_rgb(path: str, size: tuple = (512, 512)) -> np.ndarray:
    """
    讀取圖片並 resize 為 (H, W, 3) uint8 RGB。
    使用 LANCZOS 品質縮放，與 inference.py 的輸出尺寸保持一致。
    """
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    return np.array(img)


def get_region_masks(img_pil: Image.Image, bisenet, device: str) -> dict:
    """
    對給定圖片執行 BiSeNet 分割，回傳各妝容區域的 float32 遮罩。

    Returns:
        dict with keys: lips, brows, blush, skin (底妝), eyeshadow, parsing
    """
    raw = get_face_masks(img_pil, bisenet, device=device)
    parsing = raw["full_parse"]
    masks = get_advanced_masks(parsing, 512, 512)
    masks["eyeshadow"] = generate_realistic_eyeshadow(parsing, 512, 512)
    return masks


# ══════════════════════════════════════════════════════════════════════════════
# Metric ①  CAI — Color Accuracy Index（色彩準確性）
# ══════════════════════════════════════════════════════════════════════════════

def compute_cai(generated_np: np.ndarray, masks: dict, params: dict) -> dict:
    """
    色彩準確性指標（CAI，以 CIEDE2000 色差 ΔE₀₀ 衡量）。

    核心邏輯：
      1. 對每個 alpha > 0 的啟用區域，從「生成影像」的 BiSeNet 遮罩內提取像素
      2. 計算該區域的平均 LAB 色彩
      3. 與 LLM 指定的 Hex 色（也轉換為 LAB）計算 ΔE₀₀

    ΔE₀₀ 解讀：
        < 1.0  — 人眼幾乎無法分辨（極佳）
        1–2    — 極輕微差異（業界可接受標準）
        2–5    — 輕微可見差異（良好）
        5–10   — 明顯偏差（需改善）
        > 10   — 嚴重偏差

    說明：
        CAI 是 Talk2Makeup 最核心的指標，直接驗證系統從「文字描述
        → LLM 解析 Hex → 像素上色 → 擴散生成」整個流程的色彩保真度。
        λ=0.15 的插值係數理論上會讓最終色彩略偏向文字語意而非像素引導，
        CAI 可量化此偏差程度。

    Returns:
        dict: {
            "lips":       {"delta_e": float, "target_hex": str, "actual_hex": str},
            "eyeshadow":  {...},
            ...
            "mean_delta_e": float  # 所有啟用區域的平均 ΔE₀₀
        }
    """
    try:
        from skimage.color import rgb2lab, deltaE_ciede2000
    except ImportError:
        print("  [CAI] 跳過：請先安裝 pip install scikit-image")
        return {}

    # params 的 key（foundation/lips/...）對應到 masks 的 key（skin/lips/...）
    region_to_mask = {
        "lips":       masks.get("lips"),
        "eyeshadow":  masks.get("eyeshadow"),
        "blush":      masks.get("blush"),
        "brows":      masks.get("brows"),
        "foundation": masks.get("skin"),
    }

    results = {}
    for region, mask in region_to_mask.items():
        rp = params.get(region, {})
        alpha = float(rp.get("alpha", 0.0))
        if alpha <= 0 or mask is None:
            continue

        binary = (mask > 0.15).astype(bool)
        if binary.sum() < 10:
            continue

        target_rgb = hex_to_rgb(rp.get("color", "#000000"))
        pixels = generated_np[binary]                        # (N, 3) uint8
        mean_rgb = pixels.mean(axis=0).astype(np.uint8)

        # 轉 LAB 計算 ΔE₀₀（skimage 的 rgb2lab 期望輸入為 [0, 1] float）
        target_lab = rgb2lab(np.array([[target_rgb]], dtype=np.float32) / 255.0)[0, 0]
        actual_lab = rgb2lab(np.array([[mean_rgb]],  dtype=np.float32) / 255.0)[0, 0]
        delta_e = float(deltaE_ciede2000(target_lab, actual_lab))

        actual_hex = "#{:02X}{:02X}{:02X}".format(*mean_rgb.tolist())
        results[region] = {
            "delta_e":    round(delta_e, 3),
            "target_hex": rp.get("color", ""),
            "actual_hex": actual_hex,
        }

    if results:
        results["mean_delta_e"] = round(
            np.mean([v["delta_e"] for v in results.values()]), 3
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLIP 模型載入（②③④ 共用）
# ══════════════════════════════════════════════════════════════════════════════

def _load_clip(model_name: str = "ViT-B/32"):
    """
    載入 CLIP 模型。
    需安裝：pip install git+https://github.com/openai/CLIP.git
    """
    try:
        import clip
        model, preprocess = clip.load(model_name, device=DEVICE)
        model.eval()
        return model, preprocess, clip
    except ImportError:
        print("  [CLIP] 未安裝，跳過所有 CLIP 指標。")
        print("  安裝方法：pip install git+https://github.com/openai/CLIP.git")
        return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# Metric ②  ID-CLIP — 語意身份一致性
# ══════════════════════════════════════════════════════════════════════════════

def compute_id_clip(source_np: np.ndarray,
                    generated_np: np.ndarray,
                    clip_model, preprocess, clip_module) -> float:
    """
    語意身份一致性（ID-CLIP）。

    使用 CLIP Vision Encoder 提取來源與生成影像的特徵向量（512-dim），
    計算兩者的餘弦相似度，評估語意層面上「是否看起來像同一個人」。

    說明：
        CLIP 偏向高層語意（「人臉」「微笑」「亞洲女性」），
        對細粒度身份特徵（眼角紋理、皮膚細節）不夠敏感。
        Stable-Makeup 基線：ID-CLIP = 0.7211。

    Returns:
        float: 餘弦相似度，越高越相似
    """
    src_pil = Image.fromarray(source_np)
    gen_pil = Image.fromarray(generated_np)
    with torch.no_grad():
        f1 = clip_model.encode_image(preprocess(src_pil).unsqueeze(0).to(DEVICE))
        f2 = clip_model.encode_image(preprocess(gen_pil).unsqueeze(0).to(DEVICE))
        f1 = f1 / f1.norm(dim=-1, keepdim=True)
        f2 = f2 / f2.norm(dim=-1, keepdim=True)
    return round((f1 * f2).sum().item(), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Metric ③  Text-CLIP — 全域文字對齊
# ══════════════════════════════════════════════════════════════════════════════

def compute_text_clip(generated_np: np.ndarray,
                      global_prompt: str,
                      clip_model, preprocess, clip_module) -> float:
    """
    全域文字對齊分數（Text-CLIP）。

    衡量生成影像是否符合 LLM 解析輸出的全域風格描述
    （如 "Korean natural makeup, soft and dewy"）。
    使用 CLIP 跨模態對齊能力計算圖文餘弦相似度。

    說明：
        Text-CLIP 是 Talk2Makeup 特有的指標（基於參考影像的方法無法計算）。
        分數越高代表生成結果越忠實呈現使用者輸入的整體妝容風格。

    Returns:
        float: 圖文餘弦相似度，越高越好
    """
    gen_pil = Image.fromarray(generated_np)
    with torch.no_grad():
        img_feat  = clip_model.encode_image(preprocess(gen_pil).unsqueeze(0).to(DEVICE))
        text_feat = clip_model.encode_text(clip_module.tokenize([global_prompt], truncate=True).to(DEVICE))
        img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return round((img_feat * text_feat).sum().item(), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Metric ④  Region-CLIP — 逐區域文字對齊
# ══════════════════════════════════════════════════════════════════════════════

def compute_region_clip(generated_np: np.ndarray,
                        masks: dict,
                        params: dict,
                        clip_model, preprocess, clip_module) -> dict:
    """
    逐區域文字對齊分數（Region-CLIP）。

    對每個啟用區域（alpha > 0 且有 SD prompt），用 BiSeNet 遮罩的
    Bounding Box 裁剪出對應的影像 patch，計算 CLIP(patch) vs. CLIP(region_prompt)
    的餘弦相似度。

    例如：
        唇部裁圖  vs. "glossy peach lips, water tint"
        眼影裁圖  vs. "shimmery pink eyeshadow, glitter"

    說明：
        補充全域 Text-CLIP 無法捕捉的局部資訊，是評估五區域獨立控制
        效果的細粒度指標。foundation 區域通常涵蓋全臉，語意較模糊，
        可參考但解讀需謹慎。

    Returns:
        dict: { region_name: float, "mean": float }
    """
    region_to_mask = {
        "lips":       masks.get("lips"),
        "eyeshadow":  masks.get("eyeshadow"),
        "blush":      masks.get("blush"),
        "brows":      masks.get("brows"),
        "foundation": masks.get("skin"),
    }

    results = {}
    for region, mask in region_to_mask.items():
        rp = params.get(region, {})
        alpha  = float(rp.get("alpha", 0.0))
        prompt = rp.get("prompt", "").strip()
        if alpha <= 0 or not prompt or mask is None:
            continue

        binary = (mask > 0.15).astype(np.uint8)
        coords = cv2.findNonZero(binary)
        if coords is None:
            continue
        x, y, bw, bh = cv2.boundingRect(coords)
        patch = generated_np[y:y + bh, x:x + bw]
        if patch.size == 0:
            continue
        patch_pil = Image.fromarray(patch)

        with torch.no_grad():
            img_feat  = clip_model.encode_image(preprocess(patch_pil).unsqueeze(0).to(DEVICE))
            text_feat = clip_model.encode_text(clip_module.tokenize([prompt], truncate=True).to(DEVICE))
            img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        results[region] = round((img_feat * text_feat).sum().item(), 4)

    if results:
        results["mean"] = round(np.mean(list(results.values())), 4)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Metric ⑤  SSIM — 像素結構相似性
# ══════════════════════════════════════════════════════════════════════════════

def compute_ssim(source_np: np.ndarray, generated_np: np.ndarray) -> float:
    """
    結構相似性指標（SSIM）。

    從亮度、對比度、局部結構三個維度綜合評估來源圖與生成圖的相似程度，
    主要反映 Diffusion 過程中對原始臉部幾何結構（臉型、五官位置）的保留能力。

    說明：
        SSIM 高（≈1.0）代表幾何結構穩定，但 SSIM ≠ 身份保持。
        Stable-Makeup 評估基線：SSIM = 0.9020。
        Talk2Makeup 理論上應接近或高於此值（Early-stopped Inversion 設計）。

    需安裝：pip install scikit-image

    Returns:
        float: SSIM 值，範圍 [0, 1]，越高越相似
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("  [SSIM] 跳過：請先安裝 pip install scikit-image")
        return None

    src_gray = cv2.cvtColor(source_np, cv2.COLOR_RGB2GRAY)
    gen_gray = cv2.cvtColor(generated_np, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(src_gray, gen_gray, full=True, data_range=255)
    return round(float(score), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Metric ⑥  LPIPS — 感知相似性
# ══════════════════════════════════════════════════════════════════════════════

def compute_lpips(source_np: np.ndarray, generated_np: np.ndarray) -> float:
    """
    感知相似性指標（LPIPS，Learned Perceptual Image Patch Similarity）。

    使用 VGG-16 的深層特徵空間計算兩張影像的感知距離，比 SSIM 更符合
    人眼對「質感細節差異」的感知，能夠偵測霧面 vs. 光澤感的差異、
    皮膚紋理的細微變化，以及妝容造成的局部特徵改變。

    說明：
        LPIPS 越低越好（0 = 完全相同）。
        一般人像生成任務約在 0.05–0.20 之間。
        若 Talk2Makeup 的 LPIPS 顯著低於 Stable-Makeup，
        代表 Early-stopped Inversion 確實更好地保留了感知細節。

    需安裝：pip install lpips

    Returns:
        float: LPIPS 距離，越低越好
    """
    try:
        import lpips
    except ImportError:
        print("  [LPIPS] 跳過：請先安裝 pip install lpips")
        return None

    loss_fn = lpips.LPIPS(net="vgg").to(DEVICE)
    loss_fn.eval()

    def to_lpips_tensor(arr: np.ndarray):
        # lpips 期望輸入範圍 [-1, 1]
        t = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0)
        return (t / 127.5 - 1.0).to(DEVICE)

    with torch.no_grad():
        dist = loss_fn(to_lpips_tensor(source_np), to_lpips_tensor(generated_np))
    return round(float(dist.item()), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Metric ⑦  ID-ArcFace — 嚴格身份辨識相似性
# ══════════════════════════════════════════════════════════════════════════════

def compute_arcface(source_np: np.ndarray, generated_np: np.ndarray) -> float:
    """
    嚴格身份辨識相似性（ID-ArcFace）。

    使用 ArcFace（Additive Angular Margin Loss）人臉辨識模型提取高辨識性
    的臉部特徵嵌入向量，計算來源影像與生成影像的餘弦相似度。
    ArcFace 的訓練目標是最大化類間距離，因此對眼角紋理、皮膚細節、
    五官比例等「決定個人身份」的微細特徵高度敏感。

    說明：
        > 0.7   — 人臉辨識系統的同人判斷門檻
        0.5–0.7 — 可能為同一人，但特徵有偏移
        < 0.5   — 身份嚴重漂移（ID 漂移現象）
        Stable-Makeup 評估基線：ID-ArcFace = 0.3656（強妝條件下偏低）

    需安裝（二選一）：
        GPU：pip install insightface onnxruntime-gpu
        CPU：pip install insightface onnxruntime

    Returns:
        float: 餘弦相似度，越高代表身份保持越完整
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("  [ArcFace] 跳過：請先安裝 pip install insightface onnxruntime-gpu")
        return None

    ctx_id = 0 if DEVICE == "cuda" else -1
    app = FaceAnalysis(
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=ctx_id, det_size=(512, 512))

    def get_embedding(rgb_np: np.ndarray):
        bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)
        if not faces:
            return None
        # 若偵測到多張臉，取面積最大者（最靠近鏡頭的主體）
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        emb = face.embedding.astype(np.float32)
        return emb / np.linalg.norm(emb)

    src_emb = get_embedding(source_np)
    gen_emb = get_embedding(generated_np)

    if src_emb is None or gen_emb is None:
        print("  [ArcFace] 警告：影像中未偵測到人臉，返回 None")
        return None

    return round(float(np.dot(src_emb, gen_emb)), 4)


# ══════════════════════════════════════════════════════════════════════════════
# 核心評估流程
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_single(
    source_path: str,
    generated_path: str,
    params_path: str,
    bisenet_weights: str,
    metrics: list,
    clip_model_name: str = "ViT-B/32",
) -> dict:
    """對單一組（source, generated, params）計算所有選定指標，回傳結果 dict。"""

    print(f"\n{'─'*60}")
    print(f"  來源圖  : {source_path}")
    print(f"  生成圖  : {generated_path}")
    print(f"  參數檔  : {params_path}")
    print(f"{'─'*60}")

    source_np    = load_image_rgb(source_path)
    generated_np = load_image_rgb(generated_path)

    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    results = {}
    failures = {}

    # 依賴預檢
    try:
        import clip as _c; _has_clip = True
    except ImportError:
        _has_clip = False
    try:
        import skimage as _sk; _has_skimage = True
    except ImportError:
        _has_skimage = False
    try:
        import lpips as _lp; _has_lpips = True
    except ImportError:
        _has_lpips = False
    try:
        from insightface.app import FaceAnalysis as _fa; _has_insightface = True
    except ImportError:
        _has_insightface = False

    if not _has_clip:
        for m in ("id_clip", "text_clip", "region_clip"):
            if m in metrics:
                failures[m] = "CLIP 未安裝，請執行：pip install git+https://github.com/openai/CLIP.git"
    if not _has_skimage:
        if "ssim" in metrics:
            failures["ssim"] = "scikit-image 未安裝，請執行：pip install scikit-image"
        if "cai" in metrics:
            failures["cai"] = "scikit-image 未安裝，請執行：pip install scikit-image"
    if not _has_lpips and "lpips" in metrics:
        failures["lpips"] = "lpips 未安裝，請執行：pip install lpips"
    if not _has_insightface and "arcface" in metrics:
        failures["arcface"] = "insightface 未安裝，請執行：pip install insightface onnxruntime-gpu"

    # ── BiSeNet 分割（CAI / Region-CLIP 需要）────────────────────────────
    need_bisenet = any(m in metrics for m in ["cai", "region_clip"])
    masks = None
    if need_bisenet:
        print("\n[BiSeNet] 執行臉部分割（使用生成圖）...")
        if not os.path.exists(bisenet_weights):
            print(f"  找不到 BiSeNet 權重：{bisenet_weights}")
            print("  跳過 cai / region_clip")
            for m in ("cai", "region_clip"):
                if m in metrics and m not in failures:
                    failures[m] = f"找不到 BiSeNet 權重：{bisenet_weights}"
            metrics = [m for m in metrics if m not in ("cai", "region_clip")]
        else:
            bisenet = load_bisenet(bisenet_weights, device=DEVICE)
            gen_pil = Image.fromarray(generated_np)
            masks = get_region_masks(gen_pil, bisenet, DEVICE)
            del bisenet
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # ── CLIP 模型（id_clip / text_clip / region_clip 需要）───────────────
    need_clip = _has_clip and any(m in metrics for m in ["id_clip", "text_clip", "region_clip"])
    clip_model = preprocess = clip_module = None
    if need_clip:
        print(f"\n[CLIP] 載入模型 {clip_model_name}...")
        clip_model, preprocess, clip_module = _load_clip(clip_model_name)
        if clip_model is None:
            for m in ("id_clip", "text_clip", "region_clip"):
                if m in metrics and m not in failures:
                    failures[m] = "CLIP 載入失敗，請確認安裝正確"
            metrics = [m for m in metrics if m not in ("id_clip", "text_clip", "region_clip")]

    # ── 計算各指標 ────────────────────────────────────────────────────────

    if "cai" in metrics and masks is not None:
        print("\n[①] 計算 CAI（色彩準確性 ΔE₀₀）...")
        results["cai"] = compute_cai(generated_np, masks, params)

    if "id_clip" in metrics and clip_model is not None:
        print("\n[②] 計算 ID-CLIP（語意身份相似度）...")
        results["id_clip"] = compute_id_clip(
            source_np, generated_np, clip_model, preprocess, clip_module
        )

    if "text_clip" in metrics and clip_model is not None:
        print("\n[③] 計算 Text-CLIP（全域文字對齊）...")
        global_prompt = params.get("global_prompt", "")
        results["text_clip"] = compute_text_clip(
            generated_np, global_prompt, clip_model, preprocess, clip_module
        )
        results["_global_prompt"] = global_prompt

    if "region_clip" in metrics and clip_model is not None and masks is not None:
        print("\n[④] 計算 Region-CLIP（逐區域文字對齊）...")
        results["region_clip"] = compute_region_clip(
            generated_np, masks, params, clip_model, preprocess, clip_module
        )

    if "ssim" in metrics:
        print("\n[⑤] 計算 SSIM（結構相似性）...")
        v = compute_ssim(source_np, generated_np)
        if v is not None:
            results["ssim"] = v

    if "lpips" in metrics:
        print("\n[⑥] 計算 LPIPS（感知相似性）...")
        v = compute_lpips(source_np, generated_np)
        if v is not None:
            results["lpips"] = v

    if "arcface" in metrics:
        print("\n[⑦] 計算 ID-ArcFace（嚴格身份辨識）...")
        v = compute_arcface(source_np, generated_np)
        if v is None:
            if "arcface" not in failures:
                failures["arcface"] = "未偵測到人臉，請確認影像包含正面人臉"
        else:
            results["arcface"] = v

    results["_failures"] = failures
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 結果列印
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results: dict, source: str = "", generated: str = ""):
    """列印評估結果，缺失的指標顯示缺失原因。"""
    failures = results.get("_failures", {})
    if source and generated:
        print(f"\n來源：{Path(source).name}  生成：{Path(generated).name}")
    print()

    def show(label, key, note=""):
        v = results.get(key)
        if v is not None:
            print(f"  {label:<18} {v:.4f}    {note}")
        elif key in failures:
            print(f"  {label:<18} 缺失    {failures[key]}")

    # CAI
    cai = results.get("cai", {})
    print("色彩準確性 (CAI)")
    if cai:
        for region, v in cai.items():
            if region == "mean_delta_e":
                continue
            label = f"CAI-{region}"
            print(f"  {label:<18} ΔE={v['delta_e']:.2f}   {v['target_hex']} -> {v['actual_hex']}")
        mean = cai.get("mean_delta_e")
        if mean is not None:
            print(f"  {'CAI 平均 ΔE':<18} {mean}")
    elif "cai" in failures:
        print(f"  {'CAI':<18} 缺失    {failures['cai']}")
    print()

    print("身份與文字對齊")
    show("ID-CLIP",   "id_clip",   "語意身份一致性（基線 0.7211），越高越好")
    show("Text-CLIP", "text_clip", "全域文字對齊，越高越好")
    rc = results.get("region_clip", {})
    if rc:
        mean = rc.get("mean")
        if mean is not None:
            print(f"  {'Region-CLIP':<18} {mean:.4f}    逐區域文字對齊，越高越好")
    elif "region_clip" in failures:
        print(f"  {'Region-CLIP':<18} 缺失    {failures['region_clip']}")
    print()

    print("結構與品質")
    show("SSIM",       "ssim",    "結構相似（基線 0.9020），越高越好")
    show("LPIPS",      "lpips",   "感知距離（越低越好）")
    show("ID-ArcFace", "arcface", "嚴格身份辨識（基線 0.3656），越高越好")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Talk2Makeup 量化評估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 最快速（只需 scikit-image）
  python evaluate.py \\
    --source assets/input.jpg \\
    --generated results/3_final_result.png \\
    --params results/0_makeup_params.json \\
    --metrics cai ssim

  # 完整評估
  python evaluate.py \\
    --source assets/input.jpg \\
    --generated results/3_final_result.png \\
    --params results/0_makeup_params.json

  # 儲存結果
  python evaluate.py \\
    --source assets/input.jpg \\
    --generated results/3_final_result.png \\
    --params results/0_makeup_params.json \\
    --output_json results/eval_result.json

  # 批次評估（CSV 每行：source,generated,params）
  python evaluate.py \\
    --batch_csv eval_list.csv \\
    --output_json results/eval_batch.json
        """,
    )

    # ── 單張模式
    parser.add_argument("--source",    type=str, help="來源（未上妝）圖片路徑")
    parser.add_argument("--generated", type=str, help="Talk2Makeup 生成結果圖片路徑")
    parser.add_argument(
        "--params",
        type=str,
        help="LLM 解析參數 JSON 檔（inference.py 輸出的 0_makeup_params.json）",
    )

    # ── 批次模式
    parser.add_argument(
        "--batch_csv",
        type=str,
        help="批次 CSV 檔，每行格式：source_path,generated_path,params_path",
    )

    # ── 通用設定
    parser.add_argument(
        "--bisenet_weights",
        type=str,
        default="assets/79999_iter.pth",
        help="BiSeNet 權重路徑（CAI / Region-CLIP 需要，預設：assets/79999_iter.pth）",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=ALL_METRICS,
        choices=ALL_METRICS,
        metavar="METRIC",
        help=(
            "要計算的指標（空白分隔，預設：全部）。\n"
            f"可選：{' | '.join(ALL_METRICS)}"
        ),
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        help="CLIP 模型名稱（預設：ViT-B/32；更精準但更慢：ViT-L/14）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="將結果儲存至 JSON 檔（選用）",
    )

    args = parser.parse_args()

    if args.batch_csv:
        # 批次模式
        all_results = []
        with open(args.batch_csv, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 3 or not row[0] or row[0].startswith("#"):
                    continue
                src, gen, prm = row[0].strip(), row[1].strip(), row[2].strip()
                res = evaluate_single(src, gen, prm,
                                      args.bisenet_weights, list(args.metrics),
                                      args.clip_model)
                res["_source"] = src
                res["_generated"] = gen
                print_results(res, src, gen)
                all_results.append(res)

        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"✅  批次結果已儲存至 {args.output_json}")

    elif args.source and args.generated and args.params:
        # 單張模式
        res = evaluate_single(
            args.source, args.generated, args.params,
            args.bisenet_weights, list(args.metrics), args.clip_model,
        )
        print_results(res, args.source, args.generated)

        if args.output_json:
            res["_source"] = args.source
            res["_generated"] = args.generated
            out_dir = str(Path(args.output_json).parent)
            if out_dir:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"✅  結果已儲存至 {args.output_json}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
