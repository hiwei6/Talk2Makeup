# -*- coding: utf-8 -*-
"""
Projector 資訊量的完整比對分析(一次跑完,不必反覆送 job)。

前一支 probe_cond_embed.py(job 344200)得到的結論:
  1. raw_llm ≈ untrained(平均 balanced acc 0.788 vs 0.789)——妝容資訊在
     初始化時就是滿的,隨機投影完整保留。
  2. 訓練後 7/7 欄位都低於 untrained(contrastive −0.089、collapsed −0.078)
     ——訓練在「洗掉」資訊,不是學不動。
  3. untrained 的 offdiag cos 本來就是 0.9754,所以 0.996「逼近 1 = 塌縮」
     的判讀缺了對照組,指標在頂端沒有動態範圍。

這支腳本補完四個還沒被排除/量化的問題:

  [A] pooling 假說:上一支只量 mean-pool 後的線性可分性。訓練有沒有可能只是把
      資訊重新組織到 64 個 query 的結構裡,被 mean-pool 抹掉?
      → 加一組 flat_rp:把攤平的 [64*768] 用固定隨機矩陣投影到 512 維。
        隨機投影保留線性可分性(JL),又不會像直接餵 49152 維那樣過擬合;
        而且上面第 1 點已經證明隨機投影在這批資料上確實不損失資訊。

  [B] 退化的時間軌跡:是一開始就掉下去,還是隨步數單調惡化?
      → 掃 job 333319 的 step1000~6000 與 job 342370 的 step250~1000。

  [C] offdiag cos 到底能不能當指標?
      → 跨所有 checkpoint 算 offdiag cos 與 probe 準確率的相關係數。
        若接近 0 或為正,就正式否定這個指標。

  [D] 兩次 run 的對比:[改動 1][改動 2] 相對原本的訓練有沒有減緩退化。

注意 t=250 與 t=999 在上一支已經量過,差異全在雜訊內(untrained 因為
AdaLN-Zero 是完全相同),所以這裡預設只跑 t=250,需要時用 --timesteps 加回來。

用法:
    python probe_compare.py                     # 完整比對
    python probe_compare.py --poolings mean     # 只跑 mean-pool(較快)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.projector import MakeupProjector
from train_projector import caption_cache_path

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OLD = ROOT / "checkpoints/tier1_beautyface"              # job 333319:無對比項,梯度混在一起裁
NEW = ROOT / "checkpoints/tier1_beautyface_contrastive"  # job 342370:[改動 1][改動 2]

# 依訓練進程排序,輸出時才讀得出軌跡。best 不另外列:
# collapsed_best == old_step3000,contrastive_best == ctr_step750(權重相同)。
VARIANTS = [("raw_llm", None), ("untrained", OLD / "projector_untrained.pt")]
VARIANTS += [(f"old_step{s}", OLD / f"projector_step{s}.pt") for s in range(1000, 7000, 1000)]
VARIANTS += [(f"ctr_step{s}", NEW / f"projector_step{s}.pt") for s in (250, 500, 750, 1000)]

TARGETS = [
    "overall_style", "intensity_score", "lips.color_category", "lips.texture",
    "eyes.eyeshadow_tone", "blush.color_category", "brows.shape",
]

WD_GRID = (1e-4, 1e-2, 1e-1, 1.0, 10.0)


def dig(d, dotted):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ==========================================
# 特徵抽取:一次讀完 cache,同時算出所有 variant × pooling
# ==========================================
@torch.no_grad()
def extract(rows, cache_dir, projectors, timestep, rp, poolings, batch_size=32):
    """Returns {(variant, pooling): np.ndarray [N, D]}。

    刻意只掃一遍 cache:2997 個 .pt 檔的 IO 才是瓶頸,Projector forward 很便宜,
    所以把所有 variant 塞進同一次讀取裡算完。
    """
    acc = {}

    def push(key, arr):
        acc.setdefault(key, []).append(arr)

    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        cached = [torch.load(caption_cache_path(cache_dir, r["caption"]), map_location="cpu")
                  for r in chunk]
        max_len = max(c["hidden"].shape[0] for c in cached)
        hidden = torch.stack([F.pad(c["hidden"], (0, 0, 0, max_len - c["hidden"].shape[0]))
                              for c in cached]).to(DEVICE)
        mask = torch.stack([F.pad(c["mask"], (0, max_len - c["mask"].shape[0]))
                            for c in cached]).to(DEVICE)

        # raw_llm 只有 mean pooling(它沒有 query 結構,談不上 flat_rp)
        m = mask.unsqueeze(-1).float()
        push(("raw_llm", "mean"),
             ((hidden.float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())

        t = torch.full((len(chunk),), timestep, device=DEVICE, dtype=torch.long)
        for name, proj in projectors.items():
            cond = proj(hidden.to(next(proj.parameters()).dtype), t, attention_mask=mask).float()
            if "mean" in poolings:
                push((name, "mean"), cond.mean(dim=1).cpu().numpy())
            if "flat_rp" in poolings:
                push((name, "flat_rp"), (cond.reshape(cond.shape[0], -1) @ rp).cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def offdiag_cos(feats):
    """對齊 check_projector_embedding.py 的算法。塌縮時 ≈1,但注意 untrained 本來就 0.975。"""
    x = F.normalize(torch.from_numpy(feats).float(), dim=-1)
    sim = x @ x.T
    return sim[~torch.eye(sim.shape[0], dtype=torch.bool)].mean().item()


# ==========================================
# Linear probe
# ==========================================
def fit_probe(Xtr, ytr, Xte, yte, n_classes, wd, epochs=400, seed=0):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(Xtr.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=wd)
    # 類別極不平衡(k_pop_glam 58%),不加權 probe 會退化成永遠猜多數類
    counts = torch.bincount(ytr, minlength=n_classes).float().clamp(min=1)
    cw = counts.sum() / (n_classes * counts)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(lin(Xtr), ytr, weight=cw).backward()
        opt.step()
    pred = lin(Xte).argmax(1)
    rec = [(pred[yte == c] == c).float().mean().item()
           for c in range(n_classes) if (yte == c).any()]
    return float(np.mean(rec)), (pred == yte).float().mean().item()


def prep_labels(lab_tr, lab_val, min_class):
    """挑出樣本數足夠的類別,回傳 (index 對照, train 索引, val 索引, 被剔除的類別)。"""
    classes = sorted({l for l in lab_tr if l is not None}, key=str)
    keep = [c for c in classes if sum(1 for l in lab_tr if l == c) >= min_class]
    if len(keep) < 2:
        return None
    idx = {c: i for i, c in enumerate(keep)}
    tr = [i for i, l in enumerate(lab_tr) if l in idx]
    va = [i for i, l in enumerate(lab_val) if l in idx]
    if not va:
        return None
    return idx, tr, va, [c for c in classes if c not in idx]


def run_probe(Xtr_all, Xva_all, spec, lab_tr, lab_val, seed, with_shuffle):
    """單一 (variant, pooling, target) 的 probe。weight decay 用從 train 切出的
    dev 挑,絕不用 val 挑——否則各 variant 的比較會因為偷看答案而失真。"""
    idx, tr, va, dropped = spec
    X = Xtr_all[tr]
    Xv = Xva_all[va]
    y = np.array([idx[lab_tr[i]] for i in tr])
    yv = np.array([idx[lab_val[i]] for i in va])

    # 用 train 統計量標準化:raw_llm 是 4096 維、cond_embed 是 768 維,
    # 尺度差太多的話固定 lr 對其中一邊會完全不合適
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X, Xv = (X - mu) / sd, (Xv - mu) / sd

    X = torch.from_numpy(X).float().to(DEVICE)
    Xv = torch.from_numpy(Xv).float().to(DEVICE)
    y = torch.from_numpy(y).long().to(DEVICE)
    yv = torch.from_numpy(yv).long().to(DEVICE)
    K = len(idx)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    n_dev = max(1, len(X) // 10)
    dev_i = torch.from_numpy(perm[:n_dev]).to(DEVICE)
    sub_i = torch.from_numpy(perm[n_dev:]).to(DEVICE)

    best_wd, best_dev = WD_GRID[0], -1.0
    for wd in WD_GRID:
        b, _ = fit_probe(X[sub_i], y[sub_i], X[dev_i], y[dev_i], K, wd, seed=seed)
        if b > best_dev:
            best_dev, best_wd = b, wd

    bacc, acc = fit_probe(X, y, Xv, yv, K, best_wd, seed=seed)

    shuf = float("nan")
    if with_shuffle:
        ys = y[torch.from_numpy(rng.permutation(len(y))).to(DEVICE)]
        shuf, _ = fit_probe(X, ys, Xv, yv, K, best_wd, seed=seed)
    return dict(bacc=bacc, acc=acc, shuf=shuf, wd=best_wd, K=K,
                n_val=len(va), dropped=dropped)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="phototest/beautyface_train.jsonl")
    p.add_argument("--val_metadata", default="phototest/beautyface_val.jsonl")
    p.add_argument("--cache_dir", default="checkpoints/tier1_beautyface/llm_cache")
    p.add_argument("--timesteps", default="250")
    p.add_argument("--poolings", default="mean,flat_rp")
    p.add_argument("--rp_dim", type=int, default=512, help="flat_rp 的隨機投影目標維度")
    p.add_argument("--min_class", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    poolings = [s.strip() for s in args.poolings.split(",") if s.strip()]

    train_rows = load_jsonl(args.metadata)
    val_rows = load_jsonl(args.val_metadata)
    print(f"[Data] train {len(train_rows)} / val {len(val_rows)},device={DEVICE}")

    lab_tr = {t: [dig(r.get("metadata", {}), t) for r in train_rows] for t in TARGETS}
    lab_va = {t: [dig(r.get("metadata", {}), t) for r in val_rows] for t in TARGETS}
    specs = {t: prep_labels(lab_tr[t], lab_va[t], args.min_class) for t in TARGETS}

    projectors, names = {}, ["raw_llm"]
    for name, path in VARIANTS:
        if path is None:
            continue
        if not path.exists():
            print(f"[Skip] {name}:找不到 {path}")
            continue
        ck = torch.load(path, map_location=DEVICE)
        proj = MakeupProjector(**ck["config"]).to(DEVICE)
        proj.load_state_dict(ck["state_dict"])
        proj.eval()
        projectors[name] = proj
        names.append(name)
    print(f"[Load] {len(projectors)} 個 projector checkpoint")

    # 固定的隨機投影矩陣:所有 variant 共用同一個,比較才公平
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    cfg = next(iter(projectors.values()))
    flat_dim = cfg.num_queries * cfg.cross_attn_dim
    rp = (torch.randn(flat_dim, args.rp_dim, generator=g) / args.rp_dim ** 0.5).to(DEVICE)

    for ts in [int(x) for x in args.timesteps.split(",")]:
        print(f"\n[Extract] timestep={ts} ...")
        Ftr = extract(train_rows, args.cache_dir, projectors, ts, rp, poolings)
        Fva = extract(val_rows, args.cache_dir, projectors, ts, rp, poolings)

        print(f"\n{'='*104}\n  timestep = {ts}\n{'='*104}")
        results = {}   # (variant, pooling) -> {target: dict}
        cos = {}
        for name in names:
            for pool in (["mean"] if name == "raw_llm" else poolings):
                key = (name, pool)
                if key not in Fva:
                    continue
                if pool == "mean":
                    cos[name] = offdiag_cos(Fva[key])
                # shuffled 對照只對 mean pooling 跑:它衡量的是 probe 的過擬合地板,
                # 跟 variant 基本無關,全跑一遍純粹浪費時間
                res = {}
                for t in TARGETS:
                    if specs[t] is None:
                        continue
                    res[t] = run_probe(Ftr[key], Fva[key], specs[t], lab_tr[t], lab_va[t],
                                       args.seed, with_shuffle=(pool == "mean"))
                results[key] = res
                mb = np.mean([r["bacc"] for r in res.values()])
                print(f"  [done] {name:<14} {pool:<8} mean_bacc={mb:.3f}")

        # ---------- 逐欄位明細 ----------
        for t in TARGETS:
            if specs[t] is None:
                continue
            any_r = next(r[t] for r in results.values() if t in r)
            print(f"\n── {t}  ({any_r['K']} 類,chance={1/any_r['K']:.3f},val {any_r['n_val']}"
                  + (f",剔除稀有類 {any_r['dropped']}" if any_r["dropped"] else "") + ")")
            hdr = f"  {'variant':<14}"
            for pool in poolings:
                hdr += f"{pool+'_bacc':>14}"
            print(hdr + f"{'shuffled':>11}")
            for name in names:
                line = f"  {name:<14}"
                for pool in poolings:
                    r = results.get((name, pool), {}).get(t)
                    line += f"{r['bacc']:>14.3f}" if r else f"{'—':>14}"
                rm = results.get((name, "mean"), {}).get(t)
                print(line + (f"{rm['shuf']:>11.3f}" if rm else f"{'—':>11}"))

        # ---------- 總表:這才是主要結論 ----------
        base = {pool: np.mean([results[("untrained", pool)][t]["bacc"] for t in TARGETS
                               if ("untrained", pool) in results and t in results[("untrained", pool)]])
                for pool in poolings if ("untrained", pool) in results}
        print(f"\n{'='*104}\n  總表:各 checkpoint 的平均 balanced accuracy(7 個欄位),"
              f"Δ 是相對 untrained\n{'='*104}")
        hdr = f"  {'variant':<14}{'offdiag_cos':>13}"
        for pool in poolings:
            hdr += f"{pool:>12}{'Δ':>9}"
        print(hdr)
        summary = {}
        for name in names:
            line = f"  {name:<14}" + (f"{cos[name]:>13.4f}" if name in cos else f"{'—':>13}")
            for pool in poolings:
                r = results.get((name, pool))
                if r:
                    mb = np.mean([v["bacc"] for v in r.values()])
                    summary.setdefault(pool, {})[name] = mb
                    d = mb - base.get(pool, float("nan"))
                    line += f"{mb:>12.3f}{d:>+9.3f}"
                else:
                    line += f"{'—':>12}{'—':>9}"
            print(line)

        # ---------- [C] offdiag cos 到底有沒有預測力 ----------
        pts = [(cos[n], summary["mean"][n]) for n in names
               if n in cos and n in summary.get("mean", {}) and n != "raw_llm"]
        if len(pts) >= 3:
            a = np.array([x for x, _ in pts])
            b = np.array([y for _, y in pts])
            r = float(np.corrcoef(a, b)[0, 1])
            print(f"\n  [C] offdiag_cos vs mean_bacc 的 Pearson r = {r:+.3f}（n={len(pts)} 個 projector）")
            print("      r 若不顯著為負,代表『cos 越低 = 資訊越多』不成立,"
                  "offdiag cos 不能當訓練目標或收斂指標。")

    print("\n✅ 完成。")


if __name__ == "__main__":
    main()
