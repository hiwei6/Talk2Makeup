#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_beautyface_train_val.py — 依 overall_style 分層切分 train/val

beautyface 標註資料集(makeup_metadata_beautyface_tw.jsonl)的 overall_style
分佈嚴重不均(k_pop_glam 佔近六成,retro_classic/no_makeup 個位數百分比),
單純隨機切分很容易讓稀有風格整批落在同一邊、甚至驗證集完全沒有該類別樣本。
這裡改用分層抽樣:每個 overall_style 各自照 --val_frac 比例切,樣本數低於
--min_class_size 的類別直接全數留在 train(印出警告,不硬切)。

用法:
    python split_beautyface_train_val.py \\
        --input  makeup_metadata_beautyface_tw.jsonl \\
        --train_out beautyface_train.jsonl \\
        --val_out   beautyface_val.jsonl \\
        --val_frac 0.1
"""

import argparse
import collections
import json
import random


def main():
    parser = argparse.ArgumentParser(description="依 overall_style 分層切分 beautyface train/val")
    parser.add_argument("--input", default="makeup_metadata_beautyface_tw.jsonl")
    parser.add_argument("--train_out", default="beautyface_train.jsonl")
    parser.add_argument("--val_out", default="beautyface_val.jsonl")
    parser.add_argument("--val_frac", type=float, default=0.1, help="每個風格類別切多少比例當驗證集")
    parser.add_argument("--min_class_size", type=int, default=10,
                        help="樣本數低於此門檻的類別不切分,全部留在 train(避免驗證集切到 0 或 1 筆)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.input, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"[讀取] {args.input}:{len(rows)} 筆")

    by_style = collections.defaultdict(list)
    for row in rows:
        style = row.get("metadata", {}).get("overall_style", "_unknown")
        by_style[style].append(row)

    train_rows, val_rows = [], []
    print(f"\n{'風格':<18}{'total':>7}{'train':>8}{'val':>6}")
    print("─" * 40)
    for style, items in sorted(by_style.items(), key=lambda kv: -len(kv[1])):
        rng.shuffle(items)
        if len(items) < args.min_class_size:
            train_rows.extend(items)
            print(f"{style:<18}{len(items):>7}{len(items):>8}{0:>6}   (< min_class_size,全留 train)")
            continue
        n_val = max(1, round(len(items) * args.val_frac))
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])
        print(f"{style:<18}{len(items):>7}{len(items) - n_val:>8}{n_val:>6}")

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    with open(args.train_out, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.val_out, "w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("─" * 40)
    print(f"{'總計':<18}{len(rows):>7}{len(train_rows):>8}{len(val_rows):>6}")
    print(f"\n✅ train → {args.train_out}")
    print(f"✅ val   → {args.val_out}")


if __name__ == "__main__":
    main()
