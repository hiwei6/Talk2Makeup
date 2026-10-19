# -*- coding: utf-8 -*-
"""
Linear probe:量「cond_embed 裡到底有沒有裝進妝容資訊」。

背景
────
job 333319(6000 步)與 342370(1000 步 + 反塌縮對比項)都遇到同一個問題:
val loss 完全在雜訊裡(σ≈0.0096,而觀測到的變化量只有 0.01 級),讀不出
訓練有沒有效。offdiag cos 雖然從 0.999 掉到 0.16,但那只證明「不同 caption
的向量不一樣」,不證明「不一樣的方向是妝容」——beautyface 的 caption 每筆
都 unique,光靠字面差異就能把 offdiag cos 壓下去。

這支腳本換一個低雜訊、可解釋的量測方式:凍住 Projector,把 cond_embed 抽出來,
訓一個 linear probe 去預測資料集自帶的結構化欄位(lips.color_category 等)。
probe 準確率明顯高於隨機 = 資訊有進去;接近隨機 = 沒進去。

關鍵是「對照組」而不是絕對數字
────────────────────────────
  raw_llm      LLaMA-3 hidden states 直接 mean-pool。這是 Projector 的輸入,
               代表「資訊上限」——Projector 不可能憑空變出這裡沒有的東西。
  untrained    隨機初始化的 Projector。隨機投影本來就會保留一部分線性可分性,
               所以這才是真正的地板,不是 1/K。
  <trained>    訓練後的 Projector。

要看的是 trained vs untrained vs raw_llm 的關係:
  trained ≈ untrained   → 訓練沒有讓 cond_embed 更能表達妝容(白練)
  trained <  untrained  → 訓練反而破壞了 LLM 本來就有的資訊(塌縮的典型症狀)
  trained >  untrained  → 有學到東西,瓶頸在別處(例如 UNet 沒被推著去用它)

另外同時報告 embedding 的尺度指標(offdiag cos / 相對變異),因為 linear probe
會先標準化特徵,所以「資訊存在但量級極小」也會測出高準確率——但那種 embedding
餵進 UNet cross-attention 等於沒有作用。兩個數字要一起讀。

用法
────
    python probe_cond_embed.py                      # 預設跑全部 variant,t=250
    python probe_cond_embed.py --timesteps 50,250,999
    python probe_cond_embed.py --variants raw_llm,contrastive_best
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.projector import MakeupProjector
from train_projector import caption_cache_path

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Projector checkpoint 對照組。raw_llm 不是 checkpoint,特別處理。
VARIANTS = {
    "raw_llm": None,
    "untrained": ROOT / "checkpoints/tier1_beautyface/projector_untrained.pt",
    "collapsed_best": ROOT / "checkpoints/tier1_beautyface/projector_best.pt",
    "contrastive_best": ROOT / "checkpoints/tier1_beautyface_contrastive/projector_best.pt",
}

# 要探測的結構化欄位(metadata 裡的路徑,用點號表示巢狀)
TARGETS = [
    "overall_style",
    "intensity_score",
    "lips.color_category",
    "lips.texture",
    "eyes.eyeshadow_tone",
    "blush.color_category",
    "brows.shape",
]


def dig(d, dotted):
    """依 "lips.color_category" 這種點號路徑取巢狀欄位,取不到回 None。"""
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ==========================================
# 特徵抽取
# ==========================================
@torch.no_grad()
def extract_features(rows, cache_dir, projectors, timestep, batch_size=32):
    """跑一次資料,同時算出所有 variant 的特徵,避免重複讀 cache。

    Returns:
        {variant_name: np.ndarray [N, D]}
        raw_llm 是 LLM hidden 對有效 token 的 mean-pool([N, 4096]);
        其餘是 cond_embed 對 64 個 query 的 mean-pool([N, 768])。

    注意 mean-pool 而非攤平:攤平是 49152 維,以 2698 筆樣本去訓 probe
    必然嚴重過擬合,測到的會是 probe 的記憶力而不是 embedding 的資訊量。
    """
    out = {name: [] for name in projectors}
    out["raw_llm"] = []

    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        cached = [torch.load(caption_cache_path(cache_dir, r["caption"]), map_location="cpu")
                  for r in chunk]

        max_len = max(c["hidden"].shape[0] for c in cached)
        hidden = torch.stack([F.pad(c["hidden"], (0, 0, 0, max_len - c["hidden"].shape[0]))
                              for c in cached]).to(DEVICE)
        mask = torch.stack([F.pad(c["mask"], (0, max_len - c["mask"].shape[0]))
                            for c in cached]).to(DEVICE)

        # raw_llm:只對 attention_mask=1 的 token 取平均,padding 不能算進去
        m = mask.unsqueeze(-1).float()
        pooled = (hidden.float() * m).sum(1) / m.sum(1).clamp(min=1)
        out["raw_llm"].append(pooled.cpu().numpy())

        t = torch.full((len(chunk),), timestep, device=DEVICE, dtype=torch.long)
        for name, proj in projectors.items():
            cond = proj(hidden.to(next(proj.parameters()).dtype), t, attention_mask=mask)
            out[name].append(cond.float().mean(dim=1).cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in out.items() if v}


@torch.no_grad()
def embedding_scale_stats(feats):
    """報告 embedding 的「可分辨程度」,用來區分『沒資訊』與『有資訊但量級太小』。

    offdiag_cos 對齊 check_projector_embedding.py / 訓練 log 的算法(攤平後
    的 cosine similarity),塌縮時 ≈0.999。
    """
    x = torch.from_numpy(feats).float()
    xn = F.normalize(x, dim=-1)
    sim = xn @ xn.T
    n = sim.shape[0]
    off = sim[~torch.eye(n, dtype=torch.bool)]
    return off.mean().item(), (x.std(0).mean() / x.abs().mean().clamp(min=1e-12)).item()


# ==========================================
# Linear probe
# ==========================================
def fit_probe(Xtr, ytr, Xte, yte, n_classes, weight_decay, epochs=600, seed=0):
    """多類別 logistic regression,full-batch AdamW。回傳 (balanced_acc, plain_acc)。"""
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(Xtr).float().to(DEVICE)
    Xte = torch.from_numpy(Xte).float().to(DEVICE)
    ytr = torch.from_numpy(ytr).long().to(DEVICE)
    yte_t = torch.from_numpy(yte).long().to(DEVICE)

    lin = torch.nn.Linear(Xtr.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=weight_decay)
    # class weight:beautyface 類別極不平衡(k_pop_glam 佔 58%),不加權的話
    # probe 會退化成「永遠猜多數類」,balanced accuracy 直接卡在 1/K
    counts = torch.bincount(ytr, minlength=n_classes).float().clamp(min=1)
    cw = (counts.sum() / (n_classes * counts)).to(DEVICE)

    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(lin(Xtr), ytr, weight=cw).backward()
        opt.step()

    pred = lin(Xte).argmax(1)
    recalls = []
    for c in range(n_classes):
        sel = yte_t == c
        if sel.any():
            recalls.append((pred[sel] == c).float().mean().item())
    return float(np.mean(recalls)), (pred == yte_t).float().mean().item()


def probe_target(feat_tr, feat_val, lab_tr, lab_val, min_class, seed=0):
    """對單一欄位跑 probe:先挑 weight decay(用切出來的 dev),再回報 val 成績。

    weight decay 絕對不能用 val 挑——那等於偷看答案,各 variant 的比較會失真。
    """
    classes = sorted({l for l in lab_tr if l is not None})
    counts = {c: sum(1 for l in lab_tr if l == c) for c in classes}
    keep = [c for c in classes if counts[c] >= min_class]
    if len(keep) < 2:
        return None
    idx = {c: i for i, c in enumerate(keep)}

    tr = [i for i, l in enumerate(lab_tr) if l in idx]
    va = [i for i, l in enumerate(lab_val) if l in idx]
    if not va:
        return None

    X, y = feat_tr[tr], np.array([idx[lab_tr[i]] for i in tr])
    Xv, yv = feat_val[va], np.array([idx[lab_val[i]] for i in va])

    # 標準化(用 train 統計量),否則 4096 維的 raw_llm 與 768 維的 cond_embed
    # 尺度差太多,AdamW 的固定 lr 對其中一邊會完全不合適
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X, Xv = (X - mu) / sd, (Xv - mu) / sd

    # 從 train 再切 10% 當 dev 來挑 weight decay
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    n_dev = max(1, len(X) // 10)
    dev_i, sub_i = perm[:n_dev], perm[n_dev:]

    best_wd, best_dev = None, -1.0
    for wd in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        bacc, _ = fit_probe(X[sub_i], y[sub_i], X[dev_i], y[dev_i], len(keep), wd, seed=seed)
        if bacc > best_dev:
            best_dev, best_wd = bacc, wd

    bacc, acc = fit_probe(X, y, Xv, yv, len(keep), best_wd, seed=seed)

    # 打亂標籤的對照:probe 本身的過擬合地板,理論上應該 ≈ 1/K
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    shuf_bacc, _ = fit_probe(X, y_shuf, Xv, yv, len(keep), best_wd, seed=seed)

    return {
        "n_classes": len(keep), "n_train": len(X), "n_val": len(Xv),
        "chance": 1.0 / len(keep), "bacc": bacc, "acc": acc,
        "shuffled_bacc": shuf_bacc, "wd": best_wd,
        "dropped": [c for c in classes if c not in idx],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="phototest")
    p.add_argument("--metadata", default="phototest/beautyface_train.jsonl")
    p.add_argument("--val_metadata", default="phototest/beautyface_val.jsonl")
    p.add_argument("--cache_dir", default="checkpoints/tier1_beautyface/llm_cache")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--timesteps", default="250", help="逗號分隔;cond_embed 是 timestep-aware 的")
    p.add_argument("--min_class", type=int, default=20, help="train 樣本數低於此的類別直接剔除")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_rows = load_jsonl(args.metadata)
    val_rows = load_jsonl(args.val_metadata)
    print(f"[Data] train {len(train_rows)} / val {len(val_rows)},device={DEVICE}")

    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in wanted if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"未知的 variant: {unknown},可選:{list(VARIANTS)}")

    labels_tr = {t: [dig(r.get("metadata", {}), t) for r in train_rows] for t in TARGETS}
    labels_va = {t: [dig(r.get("metadata", {}), t) for r in val_rows] for t in TARGETS}

    for ts in [int(x) for x in args.timesteps.split(",")]:
        projectors = {}
        for name in wanted:
            if name == "raw_llm":
                continue
            path = VARIANTS[name]
            if not path.exists():
                print(f"[Skip] {name}:找不到 {path}")
                continue
            ck = torch.load(path, map_location=DEVICE)
            proj = MakeupProjector(**ck["config"]).to(DEVICE)
            proj.load_state_dict(ck["state_dict"])
            proj.eval()
            projectors[name] = proj
            print(f"[Load] {name} ← {path}(step {ck.get('step', '?')})")

        print(f"\n[Extract] timestep={ts} ...")
        F_tr = extract_features(train_rows, args.cache_dir, projectors, ts)
        F_va = extract_features(val_rows, args.cache_dir, projectors, ts)

        order = [v for v in wanted if v in F_tr]
        print(f"\n{'='*100}\n  timestep = {ts}\n{'='*100}")
        print("  embedding 尺度(val set):offdiag cos 越接近 1 越塌縮\n")
        print(f"  {'variant':<20}{'dim':>6}{'offdiag_cos':>14}{'std/|mean|':>14}")
        for name in order:
            cos, rel = embedding_scale_stats(F_va[name])
            print(f"  {name:<20}{F_va[name].shape[1]:>6}{cos:>14.4f}{rel:>14.4f}")

        for tgt in TARGETS:
            head = None
            lines = []
            for name in order:
                r = probe_target(F_tr[name], F_va[name], labels_tr[tgt], labels_va[tgt],
                                 args.min_class, seed=args.seed)
                if r is None:
                    break
                head = r
                lines.append(f"  {name:<20}{r['bacc']:>14.3f}{r['acc']:>10.3f}"
                             f"{r['shuffled_bacc']:>14.3f}{r['wd']:>10.0e}")
            if head is None:
                print(f"\n── {tgt}:可用類別不足,跳過")
                continue
            print(f"\n── {tgt}  ({head['n_classes']} 類,chance={head['chance']:.3f},"
                  f"train {head['n_train']} / val {head['n_val']}"
                  + (f",剔除稀有類 {head['dropped']}" if head["dropped"] else "") + ")")
            print(f"  {'variant':<20}{'balanced_acc':>14}{'acc':>10}{'shuffled':>14}{'wd':>10}")
            for l in lines:
                print(l)

    print("\n✅ 完成。判讀:比較 trained vs untrained vs raw_llm,"
          "並對照 offdiag cos——資訊存在但量級極小的 embedding,UNet 實際上用不到。")


if __name__ == "__main__":
    main()
