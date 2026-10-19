#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_projector_embedding.py — 定位 conditioning collapse 到底發生在哪一層

背景:
crosscheck_projector.py 的結果證實了 conditioning collapse:同一張來源圖
不管餵哪句 caption,clip_t_user 幾乎不變(對角線平均 = 非對角線平均,
差距 -0.0000)。這只證明了「最終生成的圖片」對文字內容不敏感,但沒說
問題出在哪一段:
    LLM hidden states -> [Projector] -> cond_embed -> [UNet + LoRA] -> 圖片
可能是 (a) Projector 本身輸出的 cond_embed 對不同 caption 幾乎沒有差異
(訓練沒學會分辨輸入),也可能是 (b) Projector 輸出其實有差異,但下游
UNet/LoRA 容量太小或 lambda_interp/early_stop_steps 的取樣範圍太窄,
沒有把這個差異表現在畫面上。

這個腳本只做 (a) 這一段的檢查:LLM encode 之後直接餵給 Projector,
不載入 VAE/UNet/BiSeNet,算出每句 caption 在幾個 inference 實際會用到的
timestep(0~250,對應 early_stop_steps=250)下的 cond_embed,兩兩算 cosine
similarity。

判讀方式:
    - 如果不同 caption 的 cond_embed 兩兩 cosine similarity 都逼近 1.0
      (跟同一句話比自己差不多),代表 Projector 這一層就已經把不同輸入
      壓成幾乎相同的向量 → 問題在 Projector / 訓練本身。
    - 如果 cosine similarity 有明顯區隔(例如同句話 vs 不同句話有清楚差距),
      代表 Projector 有學到區分輸入,問題出在下游 UNet/LoRA 或取樣設定
      (lambda_interp 太保守、early_stop_steps 太小),要往那個方向查。

用法:
    python check_projector_embedding.py --variant best
    python check_projector_embedding.py --projector checkpoints/tier1_beautyface/projector_step6000.pt
"""

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_IDS = ["21_46", "669_358_04", "9_23", "13_119"]  # 跟 compare/crosscheck 用同一組 held-out 樣本

CHECKPOINTS = {
    "untrained": ROOT / "checkpoints/tier1_beautyface/projector_untrained.pt",
    "best": ROOT / "checkpoints/tier1_beautyface/projector_best.pt",
    "step6000": ROOT / "checkpoints/tier1_beautyface/projector_step6000.pt",
}

# 0~250 是 pipeline_dreammakeup_tier1.py 實際會用到的範圍(early_stop_steps=250 的預設值);
# 999 額外拿來對照「訓練時有取樣、但推論從沒用過」的區間,看行為是否不同
PROBE_TIMESTEPS = [0, 60, 120, 180, 250, 999]


def load_captions():
    by_id = {}
    with open(ROOT / "phototest" / "beautyface_val.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_id[row["image_id"]] = row
    captions = [by_id[sid]["caption"] for sid in SAMPLE_IDS]
    labels = list(SAMPLE_IDS) + ["(uncond)"]
    captions.append("")
    return labels, captions


def cosine_matrix(vecs: torch.Tensor) -> torch.Tensor:
    v = torch.nn.functional.normalize(vecs, dim=-1)
    return v @ v.T


def print_matrix(labels, mat):
    header = "".ljust(14) + "".join(l.ljust(14) for l in labels)
    print(header)
    for i, li in enumerate(labels):
        row = "".join(f"{mat[i, j].item():.4f}".ljust(14) for j in range(len(labels)))
        print(li.ljust(14) + row)


def main():
    parser = argparse.ArgumentParser(description="檢查 Projector 輸出的 cond_embed 是否隨 caption 內容變化")
    parser.add_argument("--variant", default="best", choices=list(CHECKPOINTS))
    parser.add_argument("--projector", default=None, help="直接指定 checkpoint 路徑,覆寫 --variant")
    parser.add_argument("--llm_path", default=None)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    from models.llm_encoder import LLMTextEncoder
    from models.projector import MakeupProjector

    labels, captions = load_captions()
    print(f"[Captions] {len(captions)} 句(含 uncond):{labels}")

    encoder = LLMTextEncoder(model_path=args.llm_path, device=DEVICE, max_length=args.max_length)
    hidden, mask = encoder.encode(captions)  # [N, L, 4096], [N, L]
    encoder.free()

    ckpt_path = Path(args.projector) if args.projector else CHECKPOINTS[args.variant]
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    projector = MakeupProjector(**ckpt["config"]).to(DEVICE)
    projector.load_state_dict(ckpt["state_dict"])
    projector.eval()
    print(f"[Projector] 已載入 {ckpt_path}(step {ckpt.get('step', '?')})")

    n = hidden.shape[0]
    with torch.no_grad():
        for t in PROBE_TIMESTEPS:
            timestep = torch.full((n,), t, device=DEVICE, dtype=torch.long)
            cond_embed = projector(hidden.to(DEVICE), timestep, attention_mask=mask.to(DEVICE))
            flat = cond_embed.reshape(n, -1).float()
            mat = cosine_matrix(flat)

            print(f"\n{'='*90}")
            print(f"  t = {t}  (cond_embed cosine similarity,對角線必為 1.0000)")
            print(f"{'='*90}")
            print_matrix(labels, mat)

            offdiag = [mat[i, j].item() for i in range(n) for j in range(n)
                       if i != j and labels[i] != "(uncond)" and labels[j] != "(uncond)"]
            mean_offdiag = sum(offdiag) / len(offdiag)
            print(f"\n  4 句 caption 兩兩(不含 uncond)cosine similarity 平均:{mean_offdiag:.4f}")
            verdict = (
                "→ 逼近 1,不同 caption 在 Projector 輸出這一層就幾乎沒有區別,"
                "collapse 發生在 Projector / 訓練本身"
                if mean_offdiag > 0.98 else
                "→ 有清楚區隔,Projector 有學到區分輸入,若最終畫面仍不敏感,"
                "問題應該在下游 UNet/LoRA 容量或取樣設定(lambda_interp / early_stop_steps)"
            )
            print(f"  {verdict}")
    print()


if __name__ == "__main__":
    main()
