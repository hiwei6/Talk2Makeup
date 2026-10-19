# -*- coding: utf-8 -*-
"""
驗證 caption 條件對 UNet noise 預測是否真的有作用(paired MSE)。

同一張 val 圖、同一組噪聲、同一個 timestep,只換條件,比較三種 noise-pred MSE:
    correct  : 這張圖自己的 caption
    shuffled : 另一張圖的 caption(固定位移,保證不是自己)
    empty    : 空字串(無條件)
若 caption 真的有被 UNet 使用,correct 應該系統性低於 shuffled / empty。
差值 Δ = MSE(其他條件) - MSE(correct);Δ/SE 遠大於 0 才算有效,接近 0 代表條件沒被用到。

用法(每個 --variant 格式 name:projector.pt[:unet_lora.pt],沒給 LoRA 就用原始 UNet):
    python verify_condition_effect.py \
        --variant untrained:checkpoints/tier1_beautyface/projector_untrained.pt \
        --variant best:checkpoints/tier1_beautyface/projector_best.pt:checkpoints/tier1_beautyface/unet_lora_best.pt
"""

import argparse
import csv
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from models.projector import MakeupProjector

SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
COND_NAMES = ["correct", "shuffled", "empty"]


def caption_key(caption):
    return hashlib.sha1(caption.encode("utf-8")).hexdigest()[:16]


def load_cached(cache_dir, caption):
    d = torch.load(os.path.join(cache_dir, f"{caption_key(caption)}.pt"), map_location="cpu")
    return d["hidden"], d["mask"]


def collate(items, device):
    max_len = max(h.shape[0] for h, _ in items)
    hid = torch.stack([torch.nn.functional.pad(h, (0, 0, 0, max_len - h.shape[0])) for h, _ in items])
    msk = torch.stack([torch.nn.functional.pad(m, (0, max_len - m.shape[0])) for _, m in items])
    return hid.to(device), msk.to(device)


def encode_images(vae, rows, data_root, device, bs=8):
    out = []
    for s in range(0, len(rows), bs):
        arrs = []
        for r in rows[s:s + bs]:
            img = Image.open(os.path.join(data_root, r["image"])).convert("RGB").resize((512, 512), Image.LANCZOS)
            arrs.append(torch.from_numpy(np.array(img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1))
        x = torch.stack(arrs).to(device)
        with torch.no_grad():
            out.append((vae.encode(x).latent_dist.mean * vae.config.scaling_factor).cpu())
    return torch.cat(out)


def build_projector(path, device):
    ckpt = torch.load(path, map_location="cpu")
    proj = MakeupProjector(**ckpt["config"])
    proj.load_state_dict(ckpt["state_dict"])
    return proj.to(device).eval(), ckpt.get("step", "?")


def setup_unet_variant(unet, state, lora_path, device):
    """no-LoRA 變體:有 adapter 就關掉;LoRA 變體:首次才 add_adapter,之後只換權重。"""
    if not lora_path:
        if state["adapter"]:
            unet.disable_adapters()
        return
    from peft import LoraConfig
    lora_sd = torch.load(lora_path, map_location="cpu")
    rank = next(v.shape[0] for k, v in lora_sd.items() if "lora_A" in k)
    if not state["adapter"]:
        unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
        state["adapter"] = True
    else:
        unet.enable_adapters()
    res = unet.load_state_dict(lora_sd, strict=False)
    unexpected = [k for k in res.unexpected_keys if "lora" in k]
    if unexpected:
        raise RuntimeError(f"LoRA 權重有 {len(unexpected)} 個 key 對不上 UNet,例如 {unexpected[0]}")
    unet.to(device)


def evaluate_variant(unet, proj, sched, latents, noise, hid_correct, hid_empty, shift, timesteps, bs, device, use_amp):
    n, T = len(latents), len(timesteps)
    mse = np.zeros((3, n, T), dtype=np.float64)
    cos_shuf, cos_empty = [], []
    mid_t = min(range(T), key=lambda i: abs(timesteps[i] - 500))
    for s in range(0, n, bs):
        idx = list(range(s, min(s + bs, n)))
        b = len(idx)
        lat = latents[idx].to(device)
        noi = noise[idx].to(device)
        conds = [
            collate([hid_correct[i] for i in idx], device),
            collate([hid_correct[(i + shift) % n] for i in idx], device),
            collate([hid_empty] * b, device),
        ]
        for ti, t in enumerate(timesteps):
            tt = torch.full((b,), t, device=device, dtype=torch.long)
            noisy = sched.add_noise(lat, noi, tt)
            embs = []
            for ci, (hid, msk) in enumerate(conds):
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    emb = proj(hid, tt, attention_mask=msk)
                    pred = unet(noisy, tt, encoder_hidden_states=emb, return_dict=False)[0]
                mse[ci, idx, ti] = ((pred.float() - noi) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
                embs.append(emb.float().reshape(b, -1))
            if ti == mid_t:
                n0 = torch.nn.functional.normalize(embs[0], dim=-1)
                cos_shuf += (n0 * torch.nn.functional.normalize(embs[1], dim=-1)).sum(-1).cpu().tolist()
                cos_empty += (n0 * torch.nn.functional.normalize(embs[2], dim=-1)).sum(-1).cpu().tolist()
    return mse, float(np.mean(cos_shuf)), float(np.mean(cos_empty))


def paired_stat(delta_nT):
    per_sample = delta_nT.mean(axis=1)
    n = len(per_sample)
    mean = float(per_sample.mean())
    se = float(per_sample.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return mean, se, (mean / se if se and se > 0 else float("nan")), float((per_sample > 0).mean())


def main():
    ap = argparse.ArgumentParser(description="驗證 caption 條件對 UNet 預測的實際作用(paired MSE)")
    ap.add_argument("--variant", action="append", required=True, help="name:projector.pt[:unet_lora.pt],可重複")
    ap.add_argument("--data_root", default="phototest")
    ap.add_argument("--val_metadata", default="phototest/beautyface_val.jsonl")
    ap.add_argument("--cache_dir", default="checkpoints/tier1_beautyface/llm_cache")
    ap.add_argument("--n", type=int, default=299, help="使用前 n 筆 val 樣本")
    ap.add_argument("--timesteps", default="50,150,250,400,550,700,850,950")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output_dir", default="verify_out")
    args = ap.parse_args()

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device == "cuda"
    timesteps = [int(x) for x in args.timesteps.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.val_metadata, encoding="utf-8") if l.strip()][: args.n]
    n = len(rows)
    if n < 2:
        raise SystemExit("至少需要 2 筆樣本才能做 shuffled 比較")
    shift = max(1, n // 2)
    print(f"[Setup] device={device} n={n} timesteps={timesteps} shift={shift}")

    vae = AutoencoderKL.from_pretrained(SD_MODEL_ID, subfolder="vae").to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(SD_MODEL_ID, subfolder="unet").to(device).eval()
    unet.requires_grad_(False)
    sched = DDPMScheduler.from_pretrained(SD_MODEL_ID, subfolder="scheduler")

    t0 = time.time()
    latents = encode_images(vae, rows, args.data_root, device)
    del vae
    noise = torch.randn(latents.shape, generator=torch.Generator().manual_seed(args.seed))
    hid_correct = [load_cached(args.cache_dir, r["caption"]) for r in rows]
    hid_empty = load_cached(args.cache_dir, "")
    print(f"[Setup] latents {tuple(latents.shape)} 與 hidden states 載入完成 ({time.time() - t0:.0f}s)")

    variants = []
    for v in args.variant:
        parts = v.split(":")
        variants.append((parts[0], parts[1], parts[2] if len(parts) > 2 else None))
    variants.sort(key=lambda x: x[2] is not None)  # 無 LoRA 的先跑,之後才 add_adapter

    state = {"adapter": False}
    results = {}
    for name, proj_path, lora_path in variants:
        t0 = time.time()
        proj, step = build_projector(proj_path, device)
        setup_unet_variant(unet, state, lora_path, device)
        unet.eval()
        mse, cos_shuf, cos_empty = evaluate_variant(
            unet, proj, sched, latents, noise, hid_correct, hid_empty, shift, timesteps,
            args.batch_size, device, use_amp,
        )
        results[name] = {"mse": mse, "cos_shuf": cos_shuf, "cos_empty": cos_empty, "step": step}
        print(f"[Done] {name} (step {step}) {time.time() - t0:.0f}s")
        del proj

    print("\n" + "=" * 78)
    print(f"caption 條件效果驗證 (n={n}, paired;Δ = MSE(其他條件) - MSE(correct),>0 代表 caption 有幫助)")
    print("=" * 78)
    summary = {}
    for name, r in results.items():
        mse = r["mse"]
        c, s, e = mse
        ms, ses, ts, fs = paired_stat(s - c)
        me, see, te, fe = paired_stat(e - c)
        base = float(c.mean())
        print(f"\n[{name}] step={r['step']}  平均 MSE(correct)={base:.5f}")
        print(f"  Δ shuffled-correct : {ms:+.6f} ± {ses:.6f} (SE)  Δ/SE={ts:+.1f}  correct 勝出比例={fs * 100:.0f}%  (相對 {ms / base * 100:+.3f}%)")
        print(f"  Δ empty-correct    : {me:+.6f} ± {see:.6f} (SE)  Δ/SE={te:+.1f}  correct 勝出比例={fe * 100:.0f}%  (相對 {me / base * 100:+.3f}%)")
        print(f"  cond_embed cosine @t≈500:  correct vs shuffled={r['cos_shuf']:.4f}  correct vs empty={r['cos_empty']:.4f}")
        print("  各 timestep:    t     MSE(correct)   Δshuf       Δempty")
        for ti, t in enumerate(timesteps):
            print(f"               {t:5d}   {c[:, ti].mean():.5f}   {(s - c)[:, ti].mean():+.6f}   {(e - c)[:, ti].mean():+.6f}")
        summary[name] = {
            "step": r["step"], "mse_correct": base,
            "delta_shuffled": {"mean": ms, "se": ses, "t": ts, "win_frac": fs},
            "delta_empty": {"mean": me, "se": see, "t": te, "win_frac": fe},
            "cos_correct_vs_shuffled": r["cos_shuf"], "cos_correct_vs_empty": r["cos_empty"],
        }

    print("\n判讀參考:Δ/SE 遠大於 3 且 correct 勝出比例明顯高於 50% → 條件有被 UNet 使用;")
    print("        Δ/SE 接近 0、勝出比例約 50% → 條件幾乎沒被用到(資訊即使存在,UNet 也沒用它)。")

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "per_sample.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "image", "timestep"] + [f"mse_{c}" for c in COND_NAMES])
        for name, r in results.items():
            for i, row in enumerate(rows):
                for ti, t in enumerate(timesteps):
                    w.writerow([name, row["image"], t] + [f"{r['mse'][ci, i, ti]:.6f}" for ci in range(3)])
    print(f"\n結果已存到 {args.output_dir}/summary.json 與 per_sample.csv")


if __name__ == "__main__":
    main()
