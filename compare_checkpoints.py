#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_checkpoints.py — untrained baseline / projector_best.pt(step 3000)/
projector_step6000.pt 三方比較

train_beautyface.sh(job 333319)訓練出的兩個 checkpoint 在 validation MSE 打平
(best 0.1133 @ step3000,step6000 是 0.1256),MSE 分不出高下,只能靠 held-out
的語意指標(CLIP)判斷哪個 checkpoint 翻譯出來的條件向量更貼近文字描述。額外加入
一個隨機初始化、完全沒訓練過的 Projector(不套 LoRA)當下限基準,用來確認訓練
本身到底有沒有讓 Projector 學到東西,而不只是兩個訓練完的 checkpoint 互相比較。

流程:
    1. 從 beautyface_val.jsonl(訓練沒看過的樣本)挑 4 張、橫跨 4 種主要
       overall_style,對每張圖分別用 untrained / best / step6000 三個 checkpoint
       各跑一次 inference_tier1.py(顏色路徑固定用同一份 params JSON,只有
       Tier1 conditioning 這個變因不同,才是乾淨的對照實驗)。
    2. 用 evaluate_batch.py 對三組結果各自算 clip_t_user / clip_t_llm / clip_i /
       clip_s / ssim(user_input 用該樣本真正的 held-out caption)。
    3. 把三份 eval_summary.json 並排印出來,並算出 best 相對 untrained 的差距,
       確認訓練是否真的有效。
"""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 橫跨 4 種主要 overall_style,訓練資料裡完全沒看過的 caption/圖片配對
SAMPLE_IDS = ["21_46", "669_358_04", "9_23", "13_119"]

PARAMS_JSON = ROOT / "results" / "0_makeup_params.json"  # 顏色路徑固定錨點,兩組共用,隔離 Tier1 這個變因
SOURCES_DIR = ROOT / "phototest" / "data" / "beautyface" / "images"

CHECKPOINTS = {
    # 隨機初始化、完全沒訓練過的 Projector,不套 LoRA(LoRA 也是訓練出來的東西)。
    # 當作下限基準:訓練出來的 checkpoint 至少要贏過這組,才能證明訓練真的有學到東西。
    "untrained": {
        "projector": ROOT / "checkpoints/tier1_beautyface/projector_untrained.pt",
        "unet_lora": None,
    },
    "best": {
        "projector": ROOT / "checkpoints/tier1_beautyface/projector_best.pt",
        "unet_lora": ROOT / "checkpoints/tier1_beautyface/unet_lora_best.pt",
    },
    "step6000": {
        "projector": ROOT / "checkpoints/tier1_beautyface/projector_step6000.pt",
        "unet_lora": ROOT / "checkpoints/tier1_beautyface/unet_lora_step6000.pt",
    },
}

METRICS = ["clip_t_user", "clip_t_llm", "clip_i", "clip_s", "ssim"]


def load_samples():
    by_id = {}
    with open(ROOT / "phototest" / "beautyface_val.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_id[row["image_id"]] = row
    return [by_id[sid] for sid in SAMPLE_IDS]


def run_inference(variant: str, sample: dict):
    out_dir = ROOT / "results_compare" / variant / sample["image_id"]
    already_done = (out_dir / "3_final_result.png").exists()
    if already_done:
        print(f"[Skip] {variant}/{sample['image_id']} 已存在,不重跑")
    else:
        ckpt = CHECKPOINTS[variant]
        cmd = [
            sys.executable, "inference_tier1.py",
            "-i", str(SOURCES_DIR / sample["file_name"]),
            "-j", str(PARAMS_JSON),
            "--style_text", sample["caption"],
            "--projector", str(ckpt["projector"]),
            "-o", str(out_dir),
        ]
        if ckpt["unet_lora"]:  # untrained 這組沒有 LoRA,不傳這個參數(inference_tier1.py 會用純凍結 UNet)
            cmd += ["--unet_lora", str(ckpt["unet_lora"])]
        print(f"\n=== [{variant}] {sample['image_id']} ===")
        subprocess.run(cmd, cwd=ROOT, check=True)

    # inference_tier1.py 用 -j 模式不會把 params 存進 output_dir,補一份讓
    # evaluate_batch.py 的 clip_t_llm 找得到(不管是不是剛跑完都補,涵蓋舊結果)
    params_dst = out_dir / "0_makeup_params.json"
    if not params_dst.exists():
        shutil.copy2(PARAMS_JSON, params_dst)


def write_csv(variant: str, samples: list) -> Path:
    csv_path = ROOT / f"eval_compare_{variant}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for s in samples:
            results_dir = ROOT / "results_compare" / variant / s["image_id"]
            w.writerow([str(SOURCES_DIR / s["file_name"]), str(results_dir), s["caption"]])
    return csv_path


def run_eval(variant: str, csv_path: Path) -> Path:
    out_dir = ROOT / f"eval_out_{variant}"
    cmd = [
        sys.executable, "evaluate_batch.py",
        "--batch_csv", str(csv_path),
        "--metrics", *METRICS,
        "--output_dir", str(out_dir),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out_dir / "eval_summary.json"


VARIANT_ORDER = ["untrained", "best", "step6000"]


def print_comparison(summaries: dict):
    variants = [v for v in VARIANT_ORDER if v in summaries]
    header = "".join(f"{v:>14}" for v in variants)
    print(f"\n{'='*(20 + 14 * len(variants))}")
    print("  untrained baseline vs best(step3000) vs step6000 — held-out 語意對齊比較")
    print(f"{'='*(20 + 14 * len(variants))}")
    print(f"  {'指標':<14}{header}   best 相對 untrained")
    print(f"  {'-'*(18 + 14 * len(variants) + 24)}")
    for m in METRICS:
        vals = {v: summaries[v].get(m, {}).get("mean") for v in variants}
        if any(vals[v] is None for v in variants):
            continue
        row = "".join(f"{vals[v]:>14.4f}" for v in variants)
        note = ""
        if "untrained" in vals and "best" in vals:
            gap = vals["best"] - vals["untrained"]
            note = f"{'+' if gap >= 0 else ''}{gap:.4f}" + ("  (訓練有幫助)" if abs(gap) > 0.01 else "  (幾乎沒差)")
        print(f"  {m:<14}{row}   {note}")
    print(f"{'='*(20 + 14 * len(variants))}\n")


def main():
    samples = load_samples()
    print(f"[樣本] {len(samples)} 筆 held-out 樣本:{[s['image_id'] for s in samples]}")

    for variant in CHECKPOINTS:
        for sample in samples:
            run_inference(variant, sample)

    summaries = {}
    for variant in CHECKPOINTS:
        csv_path = write_csv(variant, samples)
        summary_path = run_eval(variant, csv_path)
        with open(summary_path, encoding="utf-8") as f:
            summaries[variant] = json.load(f)

    print_comparison(summaries)


if __name__ == "__main__":
    main()
