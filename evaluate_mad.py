#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_mad.py — 依照 MAD 論文評估框架的 Talk2Makeup 評估工具
=================================================================

依據 MAD: Makeup All-in-One with Cross-Domain Diffusion Model
(CVPRW 2025) 的評估設計，計算以下指標：

  Text-to-Makeup（Talk2Makeup 主任務）：
    CLIP_T_user — 圖文相似度（generated vs 使用者原始輸入）
                  = cos( CLIP_img(generated), CLIP_txt(user_input) )
    CLIP_T_llm  — 圖文相似度（generated vs LLM 組合提示）
                  = cos( CLIP_img(generated), CLIP_txt("makeup with {prompts}") )
    CLIP_I  — CLIP 身份保持（source vs generated）
              = cos( CLIP_img(source), CLIP_img(generated) )
    CLIP_S  — CLIP 妝容風格一致性（makeup_guide vs generated）
              = cos( CLIP_img(makeup_guide), CLIP_img(generated) )
              ※ MAD 原版以「參考妝容圖」計算；Talk2Makeup 無參考圖，
                改用 LAB 像素引導圖（2_makeup_guide.png）作為風格錨點。

  LLM 解析品質：
    LLM_align — LLM 解析合理性（user_input vs global_prompt，純文字對文字）
                = cos( CLIP_txt(user_input), CLIP_txt(global_prompt) )
                衡量 LLM 將使用者輸入轉化為 SD 提示時的語意保真度。

  Makeup Quality：
    IDS     — Identity Score：ArcFace 餘弦相似度（source vs generated）
              ※ MAD 以 Inception 分類準確率計算；此處採更通用的 ArcFace。
    PSNR    — Peak Signal-to-Noise Ratio（source vs generated，結構品質）
    SSIM    — Structural Similarity（source vs generated）

  Batch-only（需 ≥50 張）：
    KID     — Kernel Inception Distance（生成分佈 vs 參考分佈）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
快速開始（只需 pip install scikit-image + CLIP）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install scikit-image tabulate
  pip install git+https://github.com/openai/CLIP.git

  python evaluate_mad.py \\
    --source    assets/input.jpg \\
    --generated results/3_final_result.png \\
    --guide     results/2_makeup_guide.png \\
    --params    results/0_makeup_params.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
批次評估（CSV 每行：source,generated,guide,params）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python evaluate_mad.py \\
    --batch_csv eval_list.csv \\
    --output_json results/eval_mad.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
完整依賴安裝：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install scikit-image lpips tabulate
  pip install git+https://github.com/openai/CLIP.git
  pip install insightface onnxruntime-gpu   # IDS 用，可選
  pip install torch-fidelity               # KID 用，批次才需要
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
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALL_METRICS = ["clip_t_user", "clip_t_llm", "llm_align", "clip_i", "clip_s", "ids", "psnr", "ssim", "kid"]


# ══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════════════

def load_image_rgb(path: str, size: tuple = (512, 512)) -> np.ndarray:
    """讀取圖片，resize 為 (H, W, 3) uint8 RGB"""
    return np.array(Image.open(path).convert("RGB").resize(size, Image.LANCZOS))


def build_text_from_params(params: dict) -> str:
    """
    從 LLM 解析參數組裝 MAD 格式的文字描述。

    MAD 原始格式：「makeup with {keyword1}, {keyword2}, ...」
    Talk2Makeup 對應：將 global_prompt 與各啟用區域的 SD prompt 合併，
    與 MAD 的 `f"makeup with {', '.join(val['style'])}"` 結構對齊。
    """
    parts = []
    global_p = params.get("global_prompt", "").strip()
    if global_p:
        parts.append(global_p)

    for region in ("lips", "eyeshadow", "blush", "brows", "foundation"):
        rp = params.get(region, {})
        if float(rp.get("alpha", 0.0)) > 0:
            prompt = rp.get("prompt", "").strip()
            if prompt:
                parts.append(prompt)

    combined = ", ".join(parts) if parts else "makeup"
    return f"makeup with {combined}"


# ══════════════════════════════════════════════════════════════════════════════
# CLIP 模型載入（CLIP_T / CLIP_I / CLIP_S 共用）
# ══════════════════════════════════════════════════════════════════════════════

def _load_clip(model_name: str = "ViT-B/32"):
    """
    載入 CLIP 模型。
    安裝：pip install git+https://github.com/openai/CLIP.git
    """
    try:
        import clip
        model, preprocess = clip.load(model_name, device=DEVICE)
        model.eval()
        return model, preprocess, clip
    except ImportError:
        print("  [CLIP] 未安裝。安裝：pip install git+https://github.com/openai/CLIP.git")
        return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# Metric: CLIP_T — 文字對齊分數
# ══════════════════════════════════════════════════════════════════════════════

def compute_clip_t(generated_np: np.ndarray,
                   text: str,
                   clip_model, preprocess, clip_module) -> float:
    """
    CLIP_T（Text-Makeup Alignment）。

    MAD 原始定義：
        cos( CLIP_img(generated), CLIP_txt("makeup with {style_keywords}") )
        衡量生成影像是否符合文字描述的妝容風格。

    Talk2Makeup 對應：
        text = "makeup with {global_prompt}, {region_prompt1}, {region_prompt2}, ..."
        由 build_text_from_params() 組裝，與 MAD 的 style keyword 格式一致。

    分數解讀：越高代表生成結果越符合文字描述的妝容語意。

    Returns:
        float: CLIP 圖文餘弦相似度
    """
    gen_pil = Image.fromarray(generated_np)
    with torch.no_grad():
        img_feat  = clip_model.encode_image(preprocess(gen_pil).unsqueeze(0).to(DEVICE))
        text_feat = clip_model.encode_text(clip_module.tokenize([text], truncate=True).to(DEVICE))
        img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return round((img_feat * text_feat).sum().item(), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Metric: CLIP_I — 身份保持分數
# ══════════════════════════════════════════════════════════════════════════════

def compute_clip_i(source_np: np.ndarray,
                   generated_np: np.ndarray,
                   clip_model, preprocess, clip_module) -> float:
    """
    CLIP_I（Identity Preservation）。

    MAD 原始定義：
        cos( CLIP_img(source_nonmakeup), CLIP_img(generated) )
        衡量生成影像是否在語意層面保持了原始素顏人物的身份特徵。

    Talk2Makeup 對應：
        source_nonmakeup → source（未上妝的原始輸入圖片）

    說明：
        CLIP_I 在語意空間（「人臉」「性別」「表情」層次）評估身份一致性，
        對細粒度辨識特徵不夠敏感（需搭配 IDS/ArcFace 補充）。
        MAD 論文以此指標與其他方法比較（compute_text_metrics.py 第 40–61 行）。

    Returns:
        float: CLIP 圖像餘弦相似度，越高代表身份保持越好
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
# Metric: CLIP_S — 妝容風格一致性
# ══════════════════════════════════════════════════════════════════════════════

def compute_clip_s(guide_np: np.ndarray,
                   generated_np: np.ndarray,
                   clip_model, preprocess, clip_module) -> float:
    """
    CLIP_S（Makeup Style Similarity）。

    MAD 原始定義：
        cos( CLIP_img(reference_makeup), CLIP_img(generated) )
        衡量生成影像的妝容風格是否與參考妝容圖一致
        （compute_text_metrics.py 第 64–85 行）。

    Talk2Makeup 的適配：
        MAD 以「參考妝容圖」作為風格錨點；Talk2Makeup 為文字驅動系統，
        不使用參考圖。因此改以「像素引導圖」（2_makeup_guide.png）取代，
        該圖是 LAB 色彩空間上色後的直接上妝結果，完整編碼了 LLM 指定的
        色彩與區域資訊，是最接近 Talk2Makeup「目標風格」的像素級錨點。

        CLIP_S = cos( CLIP_img(makeup_guide), CLIP_img(generated) )

        此指標衡量：DreamMakeup 擴散模型的最終輸出在視覺風格上是否
        忠實地呈現了 LAB 像素引導的妝容設計（λ=0.15 插值的效果驗證）。

    分數解讀：
        高 CLIP_S → 擴散模型忠實跟隨像素引導色彩風格
        低 CLIP_S → 文字語意引導（85%）主導了最終生成，顏色偏離像素設計

    Returns:
        float: 引導圖與生成圖的 CLIP 餘弦相似度
    """
    guide_pil = Image.fromarray(guide_np)
    gen_pil   = Image.fromarray(generated_np)
    with torch.no_grad():
        f1 = clip_model.encode_image(preprocess(guide_pil).unsqueeze(0).to(DEVICE))
        f2 = clip_model.encode_image(preprocess(gen_pil).unsqueeze(0).to(DEVICE))
        f1 = f1 / f1.norm(dim=-1, keepdim=True)
        f2 = f2 / f2.norm(dim=-1, keepdim=True)
    return round((f1 * f2).sum().item(), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Metric: LLM_align — LLM 解析合理性
# ══════════════════════════════════════════════════════════════════════════════

def compute_llm_align(user_input: str,
                      global_prompt: str,
                      clip_model, clip_module) -> float:
    """
    LLM 解析合理性（LLM_align）。

    在 CLIP 文字嵌入空間中衡量 LLM 對使用者輸入的解析是否語意合理：
        LLM_align = cos( CLIP_txt(user_input), CLIP_txt(global_prompt) )

    分數解讀：
        高分 → LLM 解析後的全域風格描述與使用者原始意圖高度對齊
        低分 → 有兩種可能：
               (a) LLM 過度重新詮釋（hallucination），偏離使用者意圖
               (b) 使用者以簡短詞彙輸入（如「韓系妝」），LLM 展開為豐富的 SD 提示
                   （如 "Korean dewy natural makeup, soft gradient..."），
                   此時低分是正常的擴充行為，並非解析錯誤

        建議搭配 CLIP_T_user vs CLIP_T_llm 的差值一起解讀：
            若 CLIP_T_llm > CLIP_T_user → LLM 展開後更貼近視覺結果，擴充有效
            若 CLIP_T_user > CLIP_T_llm → 使用者原始描述比 LLM 翻譯更準確

    Returns:
        float: CLIP 文字-文字餘弦相似度
    """
    with torch.no_grad():
        t1 = clip_module.tokenize([user_input],    truncate=True).to(DEVICE)
        t2 = clip_module.tokenize([global_prompt], truncate=True).to(DEVICE)
        f1 = clip_model.encode_text(t1)
        f2 = clip_model.encode_text(t2)
        f1 = f1 / f1.norm(dim=-1, keepdim=True)
        f2 = f2 / f2.norm(dim=-1, keepdim=True)
    return round((f1 * f2).sum().item(), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Metric: IDS — Identity Score
# ══════════════════════════════════════════════════════════════════════════════

def compute_ids(source_np: np.ndarray, generated_np: np.ndarray) -> float:
    """
    Identity Score（IDS）。

    MAD 原始定義（compute_classification.py）：
        使用 Inception 模型預測生成圖屬於正確身份的機率，作為身份辨識分數。
        需在其特定資料集上訓練，不易直接套用。

    Talk2Makeup 的適配：
        改以 ArcFace 人臉辨識模型的餘弦相似度作為 IDS，
        物理意義更直觀（0 = 完全不同人，1 = 完全相同），
        與主流身份保持研究的評估標準一致。

        IDS = cos( ArcFace(source), ArcFace(generated) )

    分數解讀：
        > 0.7   — 人臉辨識系統通常採用此門檻判定同一人
        0.5–0.7 — 身份有偏移但仍可辨認
        < 0.5   — 身份嚴重漂移（Stable-Makeup 基線：0.3656）

    需安裝：pip install insightface onnxruntime-gpu
            （僅 CPU：pip install insightface onnxruntime）

    Returns:
        float: ArcFace 餘弦相似度；None 若未安裝或未偵測到人臉
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("  [IDS] 跳過：pip install insightface onnxruntime-gpu")
        return None

    ctx_id = 0 if DEVICE == "cuda" else -1
    app = FaceAnalysis(
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=ctx_id, det_size=(512, 512))

    def embed(rgb_np: np.ndarray):
        bgr   = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        e = face.embedding.astype(np.float32)
        return e / np.linalg.norm(e)

    src_emb = embed(source_np)
    gen_emb = embed(generated_np)
    if src_emb is None or gen_emb is None:
        print("  [IDS] 警告：未偵測到人臉，返回 None")
        return None
    return round(float(np.dot(src_emb, gen_emb)), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Metric: PSNR — Peak Signal-to-Noise Ratio
# ══════════════════════════════════════════════════════════════════════════════

def compute_psnr(source_np: np.ndarray, generated_np: np.ndarray) -> float:
    """
    峰值信噪比（PSNR）。

    MAD 原始用途：Makeup Removal 任務，比較「去妝生成圖」與「真實素顏圖」。

    Talk2Makeup 的適配：
        比較「來源圖」與「上妝生成圖」，評估結構失真程度。
        PSNR 越高代表像素層面失真越小，但上妝本身會改變像素，
        因此 Talk2Makeup 的 PSNR 預期低於去妝任務。
        此指標主要作為結構保持的輔助參考。

        PSNR = 20 × log₁₀(255 / √MSE(source, generated))

    Returns:
        float: PSNR（dB），越高越好；MSE=0 時返回 inf
    """
    src  = source_np.astype(np.float64)
    gen  = generated_np.astype(np.float64)
    mse  = np.mean((src - gen) ** 2)
    if mse == 0:
        return float("inf")
    return round(20.0 * np.log10(255.0 / np.sqrt(mse)), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Metric: SSIM — Structural Similarity
# ══════════════════════════════════════════════════════════════════════════════

def compute_ssim(source_np: np.ndarray, generated_np: np.ndarray) -> float:
    """
    結構相似性（SSIM）。

    MAD 原始用途：Makeup Removal 任務（同 PSNR）。
    Talk2Makeup 的適配：來源圖 vs. 生成圖，評估臉部幾何結構保持。
    Stable-Makeup 評估基線：SSIM = 0.9020。

    Returns:
        float: SSIM，範圍 [0, 1]，越高越相似
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        print("  [SSIM] 跳過：pip install scikit-image")
        return None

    src_gray = cv2.cvtColor(source_np, cv2.COLOR_RGB2GRAY)
    gen_gray = cv2.cvtColor(generated_np, cv2.COLOR_RGB2GRAY)
    score, _ = ssim_fn(src_gray, gen_gray, full=True, data_range=255)
    return round(float(score), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Metric: KID — Kernel Inception Distance（批次限定）
# ══════════════════════════════════════════════════════════════════════════════

def compute_kid_batch(generated_dir: str, reference_dir: str, subset_size: int = 50) -> float:
    """
    Kernel Inception Distance（KID）— 批次模式限定。

    MAD 原始定義（compute_metrics.py）：
        使用 Inception-v3 特徵計算生成影像集合與真實妝容影像集合之間
        的 KID，衡量整體生成分佈品質（比 FID 對小樣本更穩健）。

    Talk2Makeup 的適配：
        generated_dir — 多個 Talk2Makeup 生成結果目錄（3_final_result.png × N）
        reference_dir — 參考妝容圖目錄（作為「真實妝容分佈」）

    需安裝：pip install torch-fidelity

    Args:
        generated_dir: 含多張生成圖的目錄（支援 .png/.jpg）
        reference_dir: 含多張參考圖的目錄
        subset_size:   KID 計算的子集大小（建議 ≥ 50）

    Returns:
        float: KID mean 值，越低代表生成分佈越接近真實妝容分佈
    """
    try:
        import torch_fidelity
    except ImportError:
        print("  [KID] 跳過：pip install torch-fidelity")
        return None

    result = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=reference_dir,
        kid=True,
        kid_subset_size=subset_size,
        device=DEVICE,
        verbose=False,
    )
    return round(result["kernel_inception_distance_mean"], 6)


# ══════════════════════════════════════════════════════════════════════════════
# 核心評估流程
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_single(
    source_path: str,
    generated_path: str,
    guide_path: str,
    params_path: str,
    metrics: list,
    clip_model_name: str = "ViT-B/32",
    user_input: str = "",
) -> dict:
    """對單一組（source, generated, guide, params）計算所有選定指標。"""

    print(f"\n{'─'*60}")
    print(f"  來源圖  : {source_path}")
    print(f"  生成圖  : {generated_path}")
    print(f"  引導圖  : {guide_path}")
    print(f"  參數檔  : {params_path}")
    if user_input:
        print(f"  使用者輸入: {user_input}")
    print(f"{'─'*60}")

    source_np    = load_image_rgb(source_path)
    generated_np = load_image_rgb(generated_path)
    guide_np     = load_image_rgb(guide_path)

    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    llm_text      = build_text_from_params(params)
    global_prompt = params.get("global_prompt", "")
    print(f"  LLM 全域風格：{global_prompt[:80]}{'...' if len(global_prompt) > 80 else ''}")
    print(f"  CLIP 組合文字：{llm_text[:80]}{'...' if len(llm_text) > 80 else ''}")

    results = {
        "_llm_text":      llm_text,
        "_global_prompt": global_prompt,
        "_user_input":    user_input,
    }
    failures = {}

    # 依賴預檢
    try:
        import clip as _c; _has_clip = True
    except ImportError:
        _has_clip = False
    try:
        from skimage.metrics import structural_similarity as _ss; _has_skimage = True
    except ImportError:
        _has_skimage = False
    try:
        from insightface.app import FaceAnalysis as _fa; _has_insightface = True
    except ImportError:
        _has_insightface = False

    if not _has_clip:
        for m in ("clip_t_user", "clip_t_llm", "llm_align", "clip_i", "clip_s"):
            if m in metrics:
                failures[m] = "CLIP 未安裝，請執行：pip install git+https://github.com/openai/CLIP.git"
    if not _has_skimage and "ssim" in metrics:
        failures["ssim"] = "scikit-image 未安裝，請執行：pip install scikit-image"
    if not _has_insightface and "ids" in metrics:
        failures["ids"] = "insightface 未安裝，請執行：pip install insightface onnxruntime-gpu"
    if not user_input:
        for m in ("clip_t_user", "llm_align"):
            if m in metrics:
                failures[m] = "未提供使用者原始輸入，請加上 --user_input 參數"

    # ── CLIP 模型（所有 CLIP 指標共用）
    need_clip = _has_clip and any(
        m in metrics for m in ["clip_t_user", "clip_t_llm", "llm_align", "clip_i", "clip_s"]
    )
    clip_model = preprocess = clip_module = None
    if need_clip:
        print(f"\n[CLIP] 載入 {clip_model_name}...")
        clip_model, preprocess, clip_module = _load_clip(clip_model_name)
        if clip_model is None:
            for m in ("clip_t_user", "clip_t_llm", "llm_align", "clip_i", "clip_s"):
                if m in metrics and m not in failures:
                    failures[m] = "CLIP 載入失敗，請確認安裝正確"

    # ── 計算各指標
    if "clip_t_llm" in metrics and clip_model is not None:
        print("\n[CLIP_T_LLM] 圖像 vs LLM 組合提示...")
        results["clip_t_llm"] = compute_clip_t(
            generated_np, llm_text, clip_model, preprocess, clip_module
        )

    if "clip_t_user" in metrics and clip_model is not None and user_input:
        print("\n[CLIP_T_USER] 圖像 vs 使用者原始輸入...")
        results["clip_t_user"] = compute_clip_t(
            generated_np, user_input, clip_model, preprocess, clip_module
        )

    if "llm_align" in metrics and clip_model is not None and user_input:
        print("\n[LLM_ALIGN] 使用者輸入 vs LLM 全域風格一致性...")
        results["llm_align"] = compute_llm_align(
            user_input, global_prompt, clip_model, clip_module
        )

    if "clip_i" in metrics and clip_model is not None:
        print("\n[CLIP_I] 身份保持分數...")
        results["clip_i"] = compute_clip_i(
            source_np, generated_np, clip_model, preprocess, clip_module
        )

    if "clip_s" in metrics and clip_model is not None:
        print("\n[CLIP_S] 妝容風格一致性（引導圖 vs 生成圖）...")
        results["clip_s"] = compute_clip_s(
            guide_np, generated_np, clip_model, preprocess, clip_module
        )

    if "ids" in metrics and _has_insightface:
        print("\n[IDS] Identity Score（ArcFace）...")
        v = compute_ids(source_np, generated_np)
        if v is None:
            failures["ids"] = "未偵測到人臉，請確認影像包含正面人臉"
        else:
            results["ids"] = v

    if "psnr" in metrics:
        print("\n[PSNR] 結構品質...")
        results["psnr"] = compute_psnr(source_np, generated_np)

    if "ssim" in metrics and _has_skimage:
        print("\n[SSIM] 結構相似性...")
        results["ssim"] = compute_ssim(source_np, generated_np)

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
            print(f"  {label:<26} {v:.4f}    {note}")
        elif key in failures:
            print(f"  {label:<26} 缺失    {failures[key]}")

    print("Text-to-Makeup")
    show("CLIP_T (使用者輸入)", "clip_t_user", "圖像 vs 使用者原始文字，越高越好")
    show("CLIP_T (LLM 解析)",  "clip_t_llm",  "圖像 vs LLM 組合提示，越高越好")
    print()

    print("LLM 解析品質")
    show("LLM_align",          "llm_align",   "使用者輸入 vs LLM 全域風格語意對齊，越高越好")
    user_in = results.get("_user_input", "")
    gp      = results.get("_global_prompt", "")
    if user_in:
        print(f"  使用者輸入：{user_in}")
    if gp:
        print(f"  LLM 全域風格：{gp}")

    clip_t_u = results.get("clip_t_user")
    clip_t_l = results.get("clip_t_llm")
    if clip_t_u is not None and clip_t_l is not None:
        diff = round(clip_t_l - clip_t_u, 4)
        direction = "LLM 展開有效（貼近視覺）" if diff > 0 else "使用者輸入更直接貼近視覺"
        print(f"  CLIP_T 差值（LLM - 使用者）：{diff:+.4f}    {direction}")
    print()

    print("視覺一致性")
    show("CLIP_I",             "clip_i",      "身份保持，越高越好")
    show("CLIP_S",             "clip_s",      "妝容風格一致性，越高越好")
    print()

    print("Makeup Quality")
    show("IDS",                "ids",         "嚴格身份（基線 0.3656），越高越好")
    show("PSNR",               "psnr",        "結構品質（dB），越高越好")
    show("SSIM",               "ssim",        "結構相似（基線 0.9020），越高越好")
    llm_text = results.get("_llm_text", "")
    if llm_text:
        print(f"\n  CLIP 組合文字：{llm_text}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Talk2Makeup 量化評估工具（依 MAD CVPRW 2025 框架）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
指標對應表（MAD 論文 → Talk2Makeup 適配）：
  CLIP_T  cos(CLIP_img(gen),   CLIP_txt("makeup with {prompts}"))  ← 文字對齊
  CLIP_I  cos(CLIP_img(src),   CLIP_img(gen))                      ← 身份保持
  CLIP_S  cos(CLIP_img(guide), CLIP_img(gen))                      ← 風格一致（引導圖取代參考圖）
  IDS     cos(ArcFace(src),    ArcFace(gen))                       ← 嚴格身份（ArcFace 取代 Inception）
  PSNR    20·log₁₀(255/√MSE(src, gen))                            ← 結構品質
  SSIM    structural_similarity(src, gen)                           ← 結構相似

使用範例：
  # 完整評估（含 LLM 解析品質分析）
  python evaluate_mad.py \\
    --source     assets/input.jpg \\
    --generated  results/3_final_result.png \\
    --guide      results/2_makeup_guide.png \\
    --params     results/0_makeup_params.json \\
    --user_input "韓系自然妝，粉色唇彩"

  # 不提供 user_input（跳過 clip_t_user / llm_align）
  python evaluate_mad.py \\
    --source    assets/input.jpg \\
    --generated results/3_final_result.png \\
    --guide     results/2_makeup_guide.png \\
    --params    results/0_makeup_params.json

  # 只跑 CLIP 指標
  python evaluate_mad.py \\
    --source     assets/input.jpg \\
    --generated  results/3_final_result.png \\
    --guide      results/2_makeup_guide.png \\
    --params     results/0_makeup_params.json \\
    --user_input "煙燻妝" \\
    --metrics    clip_t_user clip_t_llm llm_align clip_i clip_s

  # 批次 CSV（每行：source,generated,guide,params[,user_input]）
  python evaluate_mad.py \\
    --batch_csv  eval_list.csv \\
    --user_input "default prompt if not in CSV" \\
    --output_json results/eval_mad.json

  # 批次評估 + 輸出 JSON
  python evaluate_mad.py \\
    --batch_csv eval_list.csv \\
    --output_json results/eval_mad.json

  # 批次 + KID（需 torch-fidelity 且 ≥50 張）
  python evaluate_mad.py \\
    --batch_csv eval_list.csv \\
    --kid_generated_dir results/all_generated/ \\
    --kid_reference_dir data/beauty_makeup/ \\
    --output_json results/eval_mad.json
        """,
    )

    # ── 單張模式
    parser.add_argument("--source",     type=str, help="來源（未上妝）圖片路徑")
    parser.add_argument("--generated",  type=str, help="Talk2Makeup 生成結果圖片路徑（3_final_result.png）")
    parser.add_argument("--guide",      type=str, help="LAB 像素引導圖路徑（2_makeup_guide.png）")
    parser.add_argument("--params",     type=str, help="LLM 解析參數 JSON（0_makeup_params.json）")
    parser.add_argument("--user_input", type=str, default="",
                        help="使用者原始輸入文字（啟用 CLIP_T_user 及 LLM_align 指標）")

    # ── 批次模式
    parser.add_argument(
        "--batch_csv",
        type=str,
        help="批次 CSV 檔（每行：source,generated,guide,params）",
    )

    # ── KID 批次
    parser.add_argument("--kid_generated_dir", type=str, default="",
                        help="KID 用：含所有生成圖的目錄")
    parser.add_argument("--kid_reference_dir", type=str, default="",
                        help="KID 用：含參考妝容圖的目錄（作為真實分佈）")
    parser.add_argument("--kid_subset_size",   type=int, default=50,
                        help="KID 子集大小（預設 50）")

    # ── 通用設定
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[m for m in ALL_METRICS if m != "kid"],
        choices=ALL_METRICS,
        metavar="METRIC",
        help=f"要計算的指標（預設：除 kid 外全部）。可選：{' | '.join(ALL_METRICS)}",
    )
    parser.add_argument("--clip_model",   type=str, default="ViT-B/32",
                        help="CLIP 模型名稱（預設：ViT-B/32；更精準：ViT-L/14）")
    parser.add_argument("--output_json", type=str, default="",
                        help="將結果儲存至 JSON（選用）")

    args = parser.parse_args()

    all_results = []

    # ── 批次模式
    if args.batch_csv:
        with open(args.batch_csv, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 4 or not row[0] or row[0].startswith("#"):
                    continue
                src, gen, guide, prm = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                # 第 5 欄為可選 user_input；若無則使用 --user_input 全域值
                ui = row[4].strip() if len(row) >= 5 else args.user_input
                res = evaluate_single(src, gen, guide, prm,
                                      list(args.metrics), args.clip_model,
                                      user_input=ui)
                res["_source"] = src
                res["_generated"] = gen
                print_results(res, src, gen)
                all_results.append(res)

        # KID（batch-only）
        if "kid" in args.metrics and args.kid_generated_dir and args.kid_reference_dir:
            print(f"\n[KID] 計算分佈距離...")
            kid_score = compute_kid_batch(
                args.kid_generated_dir, args.kid_reference_dir, args.kid_subset_size
            )
            if kid_score is not None:
                print(f"  KID = {kid_score:.6f}")
                for r in all_results:
                    r["kid"] = kid_score

        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\n✅  批次結果已儲存至 {args.output_json}")

    # ── 單張模式
    elif args.source and args.generated and args.guide and args.params:
        res = evaluate_single(
            args.source, args.generated, args.guide, args.params,
            [m for m in args.metrics if m != "kid"],
            args.clip_model,
            user_input=args.user_input,
        )
        print_results(res, args.source, args.generated)

        if args.output_json:
            res["_source"] = args.source
            res["_generated"] = args.generated
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"✅  結果已儲存至 {args.output_json}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
