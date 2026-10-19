#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_batch.py — Talk2Makeup 批次評估工具
================================================

一次評估多組生成結果，整合 evaluate.py（七指標）與 evaluate_mad.py（MAD 框架）
的所有指標，並解決逐張評估的三個問題：

  1. 模型只載入一次：CLIP / BiSeNet / LPIPS / ArcFace 全程共用，
     批次速度比重複呼叫 evaluate.py 快一個數量級
  2. 自動配對：直接掃描 inference.py 的輸出資料夾結構，不需手編 CSV
  3. 彙總統計：輸出每張圖的明細 CSV + 全體 mean/std 摘要，可直接貼進論文表格

支援指標（--metrics 選擇，預設全跑）：
  cai         色彩準確性 ΔE₀₀（越低越好）        [需 BiSeNet + scikit-image]
  region_clip 逐區域文字對齊（越高越好）          [需 BiSeNet + CLIP]
  text_clip   全域文字對齊 global_prompt（越高越好）[需 CLIP]
  clip_t_llm  MAD CLIP_T：圖 vs LLM 組合提示      [需 CLIP]
  clip_t_user MAD CLIP_T：圖 vs 使用者原始輸入    [需 CLIP + user_input]
  llm_align   使用者輸入 vs global_prompt 語意對齊 [需 CLIP + user_input]
  clip_i      CLIP 身份保持（= evaluate.py 的 id_clip）[需 CLIP]
  clip_s      妝容風格一致性 guide vs generated    [需 CLIP + 引導圖]
  ssim        結構相似性（越高越好）              [需 scikit-image]
  psnr        峰值信噪比 dB（越高越好）
  lpips       感知距離（越低越好）                [需 lpips]
  arcface     ArcFace 嚴格身份（= MAD 的 IDS）    [需 insightface]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
用法一：自動掃描模式（推薦）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  目錄結構約定（inference.py -o results_batch/<名稱> 的輸出）：
    sources/img01.jpg                  ← 來源圖，檔名 stem 與結果資料夾同名
    results_batch/img01/0_makeup_params.json
    results_batch/img01/2_makeup_guide.png
    results_batch/img01/3_final_result.png
    results_batch/img02/...

  python evaluate_batch.py \\
    --results_root results_batch \\
    --sources_dir  sources \\
    --output_dir   eval_out

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
用法二：CSV 清單模式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CSV 每行：source_path,results_dir[,user_input]（# 開頭為註解）
    sources/img01.jpg,results_batch/img01,韓系自然妝 粉色唇彩
    sources/img02.jpg,results_batch/img02

  python evaluate_batch.py --batch_csv eval_list.csv --output_dir eval_out

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
輸出（--output_dir，預設 eval_out/）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  eval_per_image.csv   每張圖一行的指標明細（可用 Excel / pandas 開）
  eval_full.json       完整結果（含 CAI 逐區域細節、失敗原因）
  eval_summary.json    全體 mean / std / min / max 摘要
  終端機同時列印摘要表格
"""

import sys

# job script（非 TTY）環境下強制行緩衝，讓進度訊息即時寫入 log
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
import csv
import json
import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")

# 重用既有評估程式的指標實作，避免重複維護
from evaluate import (
    load_image_rgb,
    get_region_masks,
    compute_cai,
    compute_region_clip,
    compute_ssim,
    _load_clip,
)
from evaluate_mad import (
    build_text_from_params,
    compute_clip_t,
    compute_clip_i,
    compute_clip_s,
    compute_llm_align,
    compute_psnr,
)
from utils.segmentation import load_bisenet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALL_METRICS = [
    "cai", "region_clip", "text_clip", "clip_t_llm", "clip_t_user",
    "llm_align", "clip_i", "clip_s", "ssim", "psnr", "lpips", "arcface",
]

# 結果資料夾內的固定檔名（inference.py 的輸出約定）
F_PARAMS    = "0_makeup_params.json"
F_GUIDE     = "2_makeup_guide.png"
F_GENERATED = "3_final_result.png"

SOURCE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


# ══════════════════════════════════════════════════════════════════════════════
# 共用模型快取：整個批次只載入一次
# ══════════════════════════════════════════════════════════════════════════════

class ModelCache:
    """依需求 lazy-load，載入後全批次共用；缺依賴時記錄原因並跳過對應指標。"""

    def __init__(self, clip_model_name: str, bisenet_weights: str):
        self.clip_model_name = clip_model_name
        self.bisenet_weights = bisenet_weights
        self._clip = None       # (model, preprocess, module) 或 "failed"
        self._bisenet = None
        self._lpips = None
        self._arcface = None

    def clip(self):
        if self._clip is None:
            print(f"[模型] 載入 CLIP {self.clip_model_name}...")
            model, preprocess, module = _load_clip(self.clip_model_name)
            self._clip = (model, preprocess, module) if model is not None else "failed"
        return None if self._clip == "failed" else self._clip

    def bisenet(self):
        if self._bisenet is None:
            if not os.path.exists(self.bisenet_weights):
                print(f"[模型] 找不到 BiSeNet 權重：{self.bisenet_weights}")
                self._bisenet = "failed"
            else:
                print("[模型] 載入 BiSeNet...")
                self._bisenet = load_bisenet(self.bisenet_weights, device=DEVICE)
        return None if self._bisenet == "failed" else self._bisenet

    def lpips(self):
        if self._lpips is None:
            try:
                import lpips as lpips_pkg
                print("[模型] 載入 LPIPS (VGG)...")
                fn = lpips_pkg.LPIPS(net="vgg").to(DEVICE)
                fn.eval()
                self._lpips = fn
            except ImportError:
                print("[模型] lpips 未安裝（pip install lpips），跳過 LPIPS")
                self._lpips = "failed"
        return None if self._lpips == "failed" else self._lpips

    def arcface(self):
        if self._arcface is None:
            try:
                from insightface.app import FaceAnalysis
                print("[模型] 載入 ArcFace (insightface)...")
                app = FaceAnalysis(
                    allowed_modules=["detection", "recognition"],
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(512, 512))
                self._arcface = app
            except ImportError:
                print("[模型] insightface 未安裝（pip install insightface onnxruntime-gpu），跳過 ArcFace")
                self._arcface = "failed"
        return None if self._arcface == "failed" else self._arcface


def compute_lpips_cached(loss_fn, source_np, generated_np) -> float:
    def to_tensor(arr):
        t = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0)
        return (t / 127.5 - 1.0).to(DEVICE)
    with torch.no_grad():
        dist = loss_fn(to_tensor(source_np), to_tensor(generated_np))
    return round(float(dist.item()), 4)


def compute_arcface_cached(app, source_np, generated_np) -> float:
    def embed(rgb_np):
        faces = app.get(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        e = face.embedding.astype(np.float32)
        return e / np.linalg.norm(e)
    src_emb, gen_emb = embed(source_np), embed(generated_np)
    if src_emb is None or gen_emb is None:
        return None
    return round(float(np.dot(src_emb, gen_emb)), 4)


# ══════════════════════════════════════════════════════════════════════════════
# 評估項目收集
# ══════════════════════════════════════════════════════════════════════════════

def find_source_for(name: str, sources_dir: Path):
    """依名稱 stem 在 sources_dir 中尋找來源圖"""
    for ext in SOURCE_EXTS:
        p = sources_dir / f"{name}{ext}"
        if p.exists():
            return str(p)
    return None


def collect_items(args) -> list:
    """回傳 [{name, source, generated, guide, params, user_input}, ...]"""
    items = []

    if args.batch_csv:
        with open(args.batch_csv, "r", encoding="utf-8") as f:
            for i, row in enumerate(csv.reader(f), 1):
                if not row or not row[0].strip() or row[0].strip().startswith("#"):
                    continue
                if len(row) < 2:
                    print(f"  [跳過] CSV 第 {i} 行格式錯誤（需要 source,results_dir）：{row}")
                    continue
                src = row[0].strip()
                rdir = Path(row[1].strip())
                ui = row[2].strip() if len(row) >= 3 else args.user_input
                items.append({
                    "name":       rdir.name,
                    "source":     src,
                    "generated":  str(rdir / F_GENERATED),
                    "guide":      str(rdir / F_GUIDE),
                    "params":     str(rdir / F_PARAMS),
                    "user_input": ui,
                })

    elif args.results_root and args.sources_dir:
        root = Path(args.results_root)
        sources_dir = Path(args.sources_dir)
        if not root.is_dir():
            sys.exit(f"錯誤：results_root 不存在：{root}")
        if not sources_dir.is_dir():
            sys.exit(f"錯誤：sources_dir 不存在：{sources_dir}")
        for rdir in sorted(p for p in root.iterdir() if p.is_dir()):
            if not (rdir / F_GENERATED).exists():
                continue
            src = find_source_for(rdir.name, sources_dir)
            if src is None:
                print(f"  [跳過] {rdir.name}：sources_dir 中找不到同名來源圖")
                continue
            items.append({
                "name":       rdir.name,
                "source":     src,
                "generated":  str(rdir / F_GENERATED),
                "guide":      str(rdir / F_GUIDE),
                "params":     str(rdir / F_PARAMS),
                "user_input": args.user_input,
            })
    else:
        return None

    # 檔案存在性檢查（generated / source 必要；guide / params 缺少只跳過相關指標）
    valid = []
    for it in items:
        missing = [p for p in (it["source"], it["generated"]) if not os.path.exists(p)]
        if missing:
            print(f"  [跳過] {it['name']}：找不到 {missing}")
            continue
        valid.append(it)
    return valid


# ══════════════════════════════════════════════════════════════════════════════
# 單項評估（使用共用模型）
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_item(item: dict, metrics: list, cache: ModelCache) -> dict:
    res = {"name": item["name"], "_source": item["source"], "_generated": item["generated"]}
    skips = {}

    source_np    = load_image_rgb(item["source"])
    generated_np = load_image_rgb(item["generated"])

    params = None
    if os.path.exists(item["params"]):
        with open(item["params"], "r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        for m in ("cai", "region_clip", "text_clip", "clip_t_llm", "llm_align"):
            if m in metrics:
                skips[m] = f"缺少 {F_PARAMS}"

    guide_np = None
    if os.path.exists(item["guide"]):
        guide_np = load_image_rgb(item["guide"])
    elif "clip_s" in metrics:
        skips["clip_s"] = f"缺少 {F_GUIDE}"

    user_input = item.get("user_input", "")
    if not user_input:
        for m in ("clip_t_user", "llm_align"):
            if m in metrics and m not in skips:
                skips[m] = "未提供 user_input"

    # ── BiSeNet 遮罩（cai / region_clip 需要，對生成圖分割）
    masks = None
    if params is not None and any(m in metrics and m not in skips for m in ("cai", "region_clip")):
        net = cache.bisenet()
        if net is not None:
            masks = get_region_masks(Image.fromarray(generated_np), net, DEVICE)
        else:
            for m in ("cai", "region_clip"):
                if m in metrics and m not in skips:
                    skips[m] = "BiSeNet 權重載入失敗"

    # ── CLIP
    clip_metrics = {"region_clip", "text_clip", "clip_t_llm", "clip_t_user", "llm_align", "clip_i", "clip_s"}
    clip_pack = None
    if any(m in metrics and m not in skips for m in clip_metrics):
        clip_pack = cache.clip()
        if clip_pack is None:
            for m in clip_metrics:
                if m in metrics and m not in skips:
                    skips[m] = "CLIP 未安裝"

    def ok(m):
        return m in metrics and m not in skips

    if ok("cai") and masks is not None:
        cai = compute_cai(generated_np, masks, params)
        res["cai_detail"] = cai
        res["cai"] = cai.get("mean_delta_e")

    if ok("region_clip") and masks is not None and clip_pack:
        rc = compute_region_clip(generated_np, masks, params, *clip_pack)
        res["region_clip_detail"] = rc
        res["region_clip"] = rc.get("mean")

    if clip_pack:
        model, preprocess, module = clip_pack
        if ok("text_clip"):
            res["text_clip"] = compute_clip_t(
                generated_np, params.get("global_prompt", ""), model, preprocess, module)
        if ok("clip_t_llm"):
            res["clip_t_llm"] = compute_clip_t(
                generated_np, build_text_from_params(params), model, preprocess, module)
        if ok("clip_t_user"):
            res["clip_t_user"] = compute_clip_t(
                generated_np, user_input, model, preprocess, module)
        if ok("llm_align"):
            res["llm_align"] = compute_llm_align(
                user_input, params.get("global_prompt", ""), model, module)
        if ok("clip_i"):
            res["clip_i"] = compute_clip_i(source_np, generated_np, model, preprocess, module)
        if ok("clip_s") and guide_np is not None:
            res["clip_s"] = compute_clip_s(guide_np, generated_np, model, preprocess, module)

    if ok("ssim"):
        v = compute_ssim(source_np, generated_np)
        if v is not None:
            res["ssim"] = v
        else:
            skips["ssim"] = "scikit-image 未安裝"

    if ok("psnr"):
        res["psnr"] = compute_psnr(source_np, generated_np)

    if ok("lpips"):
        fn = cache.lpips()
        if fn is not None:
            res["lpips"] = compute_lpips_cached(fn, source_np, generated_np)
        else:
            skips["lpips"] = "lpips 未安裝"

    if ok("arcface"):
        app = cache.arcface()
        if app is not None:
            v = compute_arcface_cached(app, source_np, generated_np)
            if v is not None:
                res["arcface"] = v
            else:
                skips["arcface"] = "未偵測到人臉"
        else:
            skips["arcface"] = "insightface 未安裝"

    res["_skips"] = skips
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 彙總與輸出
# ══════════════════════════════════════════════════════════════════════════════

def summarize(all_results: list, metrics: list) -> dict:
    """對每個指標計算 N / mean / std / min / max"""
    summary = {}
    for m in metrics:
        vals = [r[m] for r in all_results if isinstance(r.get(m), (int, float))]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        summary[m] = {
            "n":    len(vals),
            "mean": round(float(arr.mean()), 4),
            "std":  round(float(arr.std(ddof=1)), 4) if len(vals) > 1 else 0.0,
            "min":  round(float(arr.min()), 4),
            "max":  round(float(arr.max()), 4),
        }
    return summary


METRIC_NOTES = {
    "cai":         "色彩準確 ΔE₀₀，越低越好",
    "region_clip": "逐區域文字對齊，越高越好",
    "text_clip":   "全域文字對齊，越高越好",
    "clip_t_llm":  "MAD CLIP_T (LLM 提示)，越高越好",
    "clip_t_user": "MAD CLIP_T (使用者輸入)，越高越好",
    "llm_align":   "LLM 解析語意對齊，越高越好",
    "clip_i":      "CLIP 身份保持（基線 0.7211），越高越好",
    "clip_s":      "妝容風格一致性，越高越好",
    "ssim":        "結構相似（基線 0.9020），越高越好",
    "psnr":        "結構品質 dB，越高越好",
    "lpips":       "感知距離，越低越好",
    "arcface":     "嚴格身份（基線 0.3656），越高越好",
}


def print_summary(summary: dict, n_items: int, n_failed: int):
    print(f"\n{'═'*74}")
    print(f"  批次評估摘要：成功 {n_items - n_failed} / {n_items} 張")
    print(f"{'═'*74}")
    print(f"  {'指標':<14}{'N':>4}{'mean':>10}{'std':>9}{'min':>9}{'max':>9}   說明")
    print(f"  {'─'*70}")
    for m, s in summary.items():
        note = METRIC_NOTES.get(m, "")
        print(f"  {m:<14}{s['n']:>4}{s['mean']:>10.4f}{s['std']:>9.4f}"
              f"{s['min']:>9.4f}{s['max']:>9.4f}   {note}")
    print(f"{'═'*74}\n")


def write_outputs(all_results, summary, metrics, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 每張圖一行的明細 CSV
    csv_path = out / "eval_per_image.csv"
    cols = ["name"] + metrics + ["skips", "error"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in all_results:
            row = [r.get("name", "")]
            for m in metrics:
                v = r.get(m)
                row.append(v if isinstance(v, (int, float)) else "")
            row.append("; ".join(f"{k}:{v}" for k, v in r.get("_skips", {}).items()))
            row.append(r.get("_error", ""))
            w.writerow(row)

    with open(out / "eval_full.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(out / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ 明細：{csv_path}")
    print(f"✅ 完整結果：{out / 'eval_full.json'}")
    print(f"✅ 摘要：{out / 'eval_summary.json'}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Talk2Makeup 批次評估工具（模型共用、自動配對、彙總統計）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 自動掃描：results_batch/<name>/ 配對 sources/<name>.jpg
  python evaluate_batch.py --results_root results_batch --sources_dir sources

  # CSV 模式（每行：source,results_dir[,user_input]）
  python evaluate_batch.py --batch_csv eval_list.csv

  # 只跑不需額外依賴的快速指標
  python evaluate_batch.py --results_root results_batch --sources_dir sources \\
      --metrics cai ssim psnr
        """,
    )
    parser.add_argument("--results_root", type=str, default="",
                        help="含多個結果子資料夾的根目錄（每個子資料夾為 inference.py 的一次輸出）")
    parser.add_argument("--sources_dir",  type=str, default="",
                        help="來源圖目錄，檔名 stem 需與結果子資料夾同名")
    parser.add_argument("--batch_csv",    type=str, default="",
                        help="CSV 清單（每行：source,results_dir[,user_input]），與自動掃描二選一")
    parser.add_argument("--user_input",   type=str, default="",
                        help="全域 user_input（CSV 第三欄可逐行覆蓋）")
    parser.add_argument("--metrics", nargs="+", default=ALL_METRICS, choices=ALL_METRICS,
                        metavar="METRIC",
                        help=f"要計算的指標（預設全部）。可選：{' | '.join(ALL_METRICS)}")
    parser.add_argument("--bisenet_weights", type=str, default="assets/79999_iter.pth",
                        help="BiSeNet 權重（cai / region_clip 需要）")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        help="CLIP 模型（預設 ViT-B/32）")
    parser.add_argument("--output_dir", type=str, default="eval_out",
                        help="輸出目錄（預設 eval_out/）")

    args = parser.parse_args()

    items = collect_items(args)
    if items is None:
        parser.print_help()
        sys.exit("\n錯誤：請指定 --batch_csv 或 --results_root + --sources_dir")
    if not items:
        sys.exit("錯誤：找不到任何可評估的項目")

    metrics = list(args.metrics)
    print(f"共 {len(items)} 個評估項目，指標：{', '.join(metrics)}")

    cache = ModelCache(args.clip_model, args.bisenet_weights)
    all_results = []
    n_failed = 0
    t0 = time.time()

    for i, item in enumerate(items, 1):
        print(f"\n[{i}/{len(items)}] {item['name']}")
        try:
            res = evaluate_item(item, metrics, cache)
            shown = ", ".join(
                f"{m}={res[m]:.4f}" for m in metrics if isinstance(res.get(m), (int, float))
            )
            print(f"  {shown if shown else '（無可計算指標）'}")
            if res["_skips"]:
                print(f"  跳過：{'; '.join(f'{k}（{v}）' for k, v in res['_skips'].items())}")
        except Exception as e:
            n_failed += 1
            res = {"name": item["name"], "_source": item["source"],
                   "_generated": item["generated"], "_error": str(e), "_skips": {}}
            print(f"  ❌ 評估失敗：{e}")
        all_results.append(res)

    elapsed = time.time() - t0
    print(f"\n總耗時 {elapsed:.1f} 秒（平均 {elapsed / len(items):.1f} 秒/張）")

    summary = summarize(all_results, metrics)
    print_summary(summary, len(items), n_failed)
    write_outputs(all_results, summary, metrics, args.output_dir)


if __name__ == "__main__":
    main()
