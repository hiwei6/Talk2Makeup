#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crosscheck_projector.py — 檢驗 Projector 是否真的依照文字條件產生不同妝容
(conditioning collapse 檢測)

背景:
compare_checkpoints.py 顯示訓練後 clip_i/ssim/clip_s 大幅提升,但
clip_t_user/clip_t_llm(語意對齊)幾乎沒變。肉眼比對後發現 untrained 的
問題主要是畫面失真(distortion),不是「妝容套錯」——這代表訓練後的提升
可能只是「projector 不再輸出雜訊、讓 UNet 正常運作」,而不是「projector
學會依照不同文字內容產生不同妝容」。

驗證方式:
固定同一張來源圖(顏色路徑也固定,隔離 Tier1 文字條件這一個變因),分別用
4 個 held-out caption(跨 4 種風格)當 Tier1 條件文字各生成一張結果圖
(同一個 checkpoint)。再用 evaluate_batch.py 的 clip_t_user,把「每張生成
圖」對「每一句 caption」都算一次相似度,組成 4x4 矩陣:
    - 對角線 = 生成時用的 caption 對到自己的分數,理應明顯高於同一列的
      其他欄(用錯的 caption)。
    - 若對角線跟非對角線差不多 → conditioning collapse:不管餵哪句
      caption,輸出幾乎不受影響,訓練只學到「別把畫面搞壞」而非「聽話」。

用法:
    python crosscheck_projector.py --anchor 21_46 --variant best
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 跟 compare_checkpoints.py 用同一組 held-out 樣本,避免引入新變因
SAMPLE_IDS = ["21_46", "669_358_04", "9_23", "13_119"]

PARAMS_JSON = ROOT / "results" / "0_makeup_params.json"  # 顏色路徑固定錨點,跟 compare_checkpoints.py 共用
SOURCES_DIR = ROOT / "phototest" / "data" / "beautyface" / "images"

CHECKPOINTS = {
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


def load_samples():
    by_id = {}
    with open(ROOT / "phototest" / "beautyface_val.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_id[row["image_id"]] = row
    return [by_id[sid] for sid in SAMPLE_IDS]


def run_inference(variant: str, anchor: dict, caption_owner_id: str, caption_text: str, out_dir: Path):
    already_done = (out_dir / "3_final_result.png").exists()
    if already_done:
        print(f"[Skip] {out_dir} 已存在,不重跑")
    else:
        ckpt = CHECKPOINTS[variant]
        cmd = [
            sys.executable, "inference_tier1.py",
            "-i", str(SOURCES_DIR / anchor["file_name"]),
            "-j", str(PARAMS_JSON),
            "--style_text", caption_text,
            "--projector", str(ckpt["projector"]),
            "-o", str(out_dir),
        ]
        if ckpt["unet_lora"]:
            cmd += ["--unet_lora", str(ckpt["unet_lora"])]
        print(f"\n=== [{variant}] anchor={anchor['image_id']} caption_from={caption_owner_id} ===")
        subprocess.run(cmd, cwd=ROOT, check=True)

    params_dst = out_dir / "0_makeup_params.json"
    if not params_dst.exists():
        shutil.copy2(PARAMS_JSON, params_dst)


def print_matrix(ids: list, pairs: list, rows: list):
    matrix = {g: {} for g in ids}
    for (gen_owner, score_owner), row in zip(pairs, rows):
        v = row["clip_t_user"]
        matrix[gen_owner][score_owner] = float(v) if v not in ("", None) else float("nan")

    print("\n每一列 = 用哪個樣本的 caption 生成;每一欄 = 拿哪個樣本的 caption 去算 clip_t_user")
    print("對角線(標 *)是「生成結果對應自己 caption」的分數,理應明顯高於同列其他欄\n")

    header = "生成用caption\\計分用caption".ljust(22) + "".join(c.ljust(14) for c in ids)
    print(header)
    diag_vals, offdiag_vals = [], []
    for g in ids:
        cells = []
        for c in ids:
            v = matrix[g][c]
            mark = "*" if g == c else " "
            cells.append(f"{v:.4f}{mark}".ljust(14))
            (diag_vals if g == c else offdiag_vals).append(v)
        print(g.ljust(22) + "".join(cells))

    diag_mean = sum(diag_vals) / len(diag_vals)
    offdiag_mean = sum(offdiag_vals) / len(offdiag_vals)
    gap = diag_mean - offdiag_mean
    print(f"\n對角線平均(用對的 caption): {diag_mean:.4f}")
    print(f"非對角線平均(用錯的 caption): {offdiag_mean:.4f}")
    verdict = (
        "→ 差距明顯,projector 確實依文字內容區分輸出"
        if gap > 0.01 else
        "→ 差距很小,懷疑 conditioning collapse:輸出幾乎不受 caption 內容影響"
    )
    print(f"差距: {gap:+.4f}  {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Projector conditioning-collapse cross-check")
    parser.add_argument("--anchor", default=SAMPLE_IDS[0], help="固定不變的來源圖 image_id(預設用第一個 held-out 樣本)")
    parser.add_argument("--variant", default="best", choices=list(CHECKPOINTS), help="要測試的 checkpoint")
    args = parser.parse_args()

    samples = load_samples()
    by_id = {s["image_id"]: s for s in samples}
    anchor = by_id[args.anchor]

    out_root = ROOT / "results_crosscheck" / args.variant / anchor["image_id"]
    gen_dirs = []
    for owner in samples:
        out_dir = out_root / f"caption_from_{owner['image_id']}"
        run_inference(args.variant, anchor, owner["image_id"], owner["caption"], out_dir)
        gen_dirs.append((owner["image_id"], out_dir))

    # 組 4x4 CSV:每個生成結果 x 每一句 caption,來源圖固定不變(clip_i 才有意義,這裡只用 clip_t_user)
    csv_path = ROOT / f"eval_crosscheck_{args.variant}_{anchor['image_id']}.csv"
    pairs = []  # (生成時用的 caption 主人, 計分用的 caption 主人)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for gen_owner, out_dir in gen_dirs:
            for score_owner in samples:
                w.writerow([str(SOURCES_DIR / anchor["file_name"]), str(out_dir), score_owner["caption"]])
                pairs.append((gen_owner, score_owner["image_id"]))

    out_dir_eval = ROOT / f"eval_out_crosscheck_{args.variant}_{anchor['image_id']}"
    cmd = [
        sys.executable, "evaluate_batch.py",
        "--batch_csv", str(csv_path),
        "--metrics", "clip_t_user",
        "--output_dir", str(out_dir_eval),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)

    with open(out_dir_eval / "eval_per_image.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(pairs), f"筆數不對:{len(rows)} vs {len(pairs)}"

    print(f"\n{'='*90}")
    print(f"  Conditioning-collapse 交叉檢測 — 錨點圖:{anchor['image_id']}  checkpoint:{args.variant}")
    print(f"{'='*90}")
    print_matrix([s["image_id"] for s in samples], pairs, rows)
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
