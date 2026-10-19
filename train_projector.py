# -*- coding: utf-8 -*-
"""
Tier1 Projector 訓練腳本。

訓練目標:讓 MakeupProjector 學會把 LLM hidden states 翻譯成 UNet 看得懂的
條件向量。採標準 diffusion 訓練範式(對目標圖加噪、預測噪聲的 MSE),
凍結 LLM、VAE、UNet 主幹,只訓練 Projector(可選 UNet LoRA 一起訓練)。

────────────────────────────────────────────────────────────────────────
2026-09 診斷後的兩項改動(搜尋 [改動 1] / [改動 2] 可找到所有相關位置)
────────────────────────────────────────────────────────────────────────
問題:job 333319 訓練 6000 步後,train loss 前 50 步平均 0.1356、後 50 步
平均 0.1388(等於沒學),val loss 在 0.113~0.156 間雜訊震盪沒有收斂趨勢。
用 check_projector_embedding.py 量 Projector 的輸出,發現 4 句風格差異極大
的 caption 兩兩 cosine similarity 高達 0.996~0.999,連 caption vs 空字串
(uncond)都分不出來——Projector 塌縮成「不管輸入什麼都吐出幾乎同一個向量」。

[改動 1] 梯度裁剪改成各參數群組分開裁,並記錄兩者各自的 grad norm。
         原本 Projector 與 LoRA 混在一起算 total norm 再裁到 1.0,若 LoRA
         梯度量級遠大於 Projector,縮放係數由 LoRA 主導,會把 Projector
         本來就微弱的梯度再壓小一輪。

[改動 2] 加入反塌縮(anti-collapse)對比項,用 cross entropy 逼同一個 batch
         內不同 caption 的 cond_embed 彼此推開,提供一條不經過 pixel 重建
         誤差的梯度路徑(重建誤差被背景/髮型/臉型主導,妝容訊號被稀釋掉,
         這正是原本學不動的根因)。

資料格式(--metadata 指向 jsonl,每行一筆):
    {"image": "images/0001.jpg", "caption": "微醺感日系桃花妝,水潤蜜桃色唇妝"}
    image 路徑相對於 --data_root。

流程分兩階段以節省 VRAM:
    Stage 1: 載入 LLM,把所有 caption(含空字串,供 CFG dropout 用)的
             hidden states 算好存到 --cache_dir,然後釋放 LLM。
    Stage 2: 載入 SD1.5,跑 diffusion 訓練迴圈,條件向量從快取讀取。

用法範例:
    python train_projector.py \
        --data_root data/makeup --metadata data/makeup/metadata.jsonl \
        --output_dir checkpoints/tier1 --max_steps 20000 --unet_lora
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from models.projector import MakeupProjector

# 把工作目錄切到本檔案所在位置,確保相對路徑(如 checkpoints/、data/)
# 不會因為使用者從別的目錄執行 python 而跑掉
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 有 GPU 就用 GPU,否則退回 CPU
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"  # HuggingFace 上的 SD1.5 權重 repo id(VAE/UNet/Scheduler 皆從這裡載入)


# ==========================================
# Stage 1: 預先計算 LLM hidden states
# ==========================================
def caption_key(caption: str) -> str:
    """caption 的短雜湊。同時當作快取檔名,以及 [改動 2] 判斷「是不是同一句 caption」的比對鍵。"""
    return hashlib.sha1(caption.encode("utf-8")).hexdigest()[:16]


def caption_cache_path(cache_dir: str, caption: str) -> str:
    """把 caption 文字轉成固定長度的檔名,回傳這筆 caption 對應的快取檔路徑。

    用 sha1 雜湊而非直接拿 caption 當檔名,是因為 caption 可能含有
    無法當檔名的字元(斜線、過長等),雜湊後長度固定又不會撞名。
    """
    return os.path.join(cache_dir, f"{caption_key(caption)}.pt")


def precompute_hidden_states(metadata, cache_dir, llm_path, max_length):
    """Stage 1 主流程:把所有出現過的 caption 過一次 LLM,存下 hidden states。

    這樣 Stage 2 訓練 diffusion 迴圈時就不用同時把 LLM 也塞進 VRAM,
    只要從硬碟讀快取的 tensor 即可,大幅省下顯存與重複前向運算。
    """
    from models.llm_encoder import LLMTextEncoder  # 延後 import,避免 Stage 2 用不到時也要載入 LLM 相關套件

    os.makedirs(cache_dir, exist_ok=True)
    # 蒐集所有不重複的 caption,並額外加入空字串 "" ——
    # 這是給 CFG(classifier-free guidance)dropout 用的無條件分支
    captions = sorted({item["caption"] for item in metadata} | {""})
    # 只計算「快取檔還不存在」的 caption,讓腳本可以中斷後重跑、不必重算已算過的部分
    todo = [c for c in captions if not os.path.exists(caption_cache_path(cache_dir, c))]

    if not todo:
        print(f"[Cache] {len(captions)} 筆 hidden states 皆已存在,跳過 LLM 載入")
        return  # 全部都算過了,連 LLM 都不用載入,直接省下這筆開銷

    # 只有真的有東西要算,才載入 LLM(佔用大量 VRAM)
    encoder = LLMTextEncoder(model_path=llm_path, device=DEVICE, max_length=max_length)
    print(f"[Cache] 需計算 {len(todo)}/{len(captions)} 筆 hidden states")
    for caption in tqdm(todo):
        hidden, mask = encoder.encode(caption)  # hidden: LLM 最後一層輸出, mask: padding attention mask
        torch.save(
            # [0] 是拿掉 batch 維度(encode 一次只丟一筆 caption 進去,batch size = 1)
            {"hidden": hidden[0].cpu(), "mask": mask[0].cpu(), "caption": caption},
            caption_cache_path(cache_dir, caption),
        )
    encoder.free()  # 算完立刻釋放 LLM 佔用的顯存,把空間留給 Stage 2 的 SD 模型


# ==========================================
# Dataset
# ==========================================
class MakeupPairDataset(Dataset):
    """把 (圖片, caption) 這對資料組成訓練樣本的 PyTorch Dataset。

    caption 本身不在這裡跑 LLM——那太貴了,每個 epoch 重算沒有意義。
    這裡只是拿 caption 當 key,去 Stage 1 存好的快取裡撈對應的
    hidden states / mask,搭配當場讀取、resize 的圖片一起回傳。
    """

    def __init__(self, data_root, metadata, cache_dir, caption_dropout=0.1):
        self.data_root = data_root          # 圖片檔案的根目錄,metadata 裡的 image 路徑是相對於它
        self.metadata = metadata            # list of {"image": ..., "caption": ...},從 jsonl 讀進來的全部樣本
        self.cache_dir = cache_dir          # Stage 1 產生的 hidden states 快取資料夾
        self.caption_dropout = caption_dropout  # CFG dropout 機率:每筆樣本有這個機率把 caption 換成空字串

    def __len__(self):
        return len(self.metadata)

    def _load_image(self, rel_path):
        """讀取單張圖片,轉成模型要的 tensor 格式。"""
        img = Image.open(os.path.join(self.data_root, rel_path)).convert("RGB")  # 統一轉成 RGB,避免灰階/RGBA 圖出錯
        img = img.resize((512, 512), Image.LANCZOS)  # SD1.5 訓練解析度固定 512x512,LANCZOS 是高品質縮放濾波器
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0   # 像素值從 [0, 255] 正規化到 [-1, 1](VAE 預期的輸入範圍)
        return torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW,符合 PyTorch 影像 tensor 慣例

    def __getitem__(self, idx):
        item = self.metadata[idx]
        caption = item["caption"]
        # CFG dropout:一定比例把條件換成空字串,訓練出無條件分支,
        # 這樣推論時才能用 classifier-free guidance(有條件/無條件預測的差值)來增強效果
        if random.random() < self.caption_dropout:
            caption = ""
        # 依 caption(可能已被換成 "")去讀 Stage 1 存好的 hidden states 快取,
        # 不用即時跑 LLM forward,大幅加速訓練迴圈
        cached = torch.load(caption_cache_path(self.cache_dir, caption))
        return {
            "pixel_values": self._load_image(item["image"]),  # 目標圖片(diffusion 訓練時會加噪、當作 ground truth)
            "hidden": cached["hidden"],   # LLM 對這個 caption 的 hidden states,序列長度依 caption 長短而異
            "mask": cached["mask"],       # 對應的 attention mask(標記哪些 token 是有效的、哪些是 padding)
            # [改動 2] 回傳這筆樣本「實際使用」的 caption 雜湊(已套用上面的 CFG dropout)。
            # 對比損失要靠它辨認同一個 batch 裡哪些樣本其實是同一句話——被 dropout 換成
            # "" 的樣本彼此是同一句,把它們推開會製造矛盾梯度,必須遮掉。
            "caption_key": caption_key(caption),
        }


def collate_fn(batch):
    """hidden states 長度不一,padding 到 batch 內最大長度。

    每筆樣本的 caption token 數不同,DataLoader 沒辦法直接把長度不一的
    tensor 疊成一個 batch,所以這裡手動 pad 到這個 batch 裡最長的那筆。
    """
    max_len = max(b["hidden"].shape[0] for b in batch)  # 這個 batch 中最長的 token 序列長度
    hidden_dim = batch[0]["hidden"].shape[1]  # noqa: F841  # hidden 維度(目前未直接使用,保留供除錯/擴充參考)

    hiddens, masks = [], []
    for b in batch:
        L = b["hidden"].shape[0]
        pad = max_len - L  # 這筆樣本要補多少 padding 才能對齊 batch 內最長序列
        # F.pad 對 2D tensor (seq_len, hidden_dim) 的 pad 參數是 (左, 右, 上, 下):
        # 這裡只在序列長度(第 0 維)後面補 0,hidden_dim 維度不動
        hiddens.append(F.pad(b["hidden"], (0, 0, 0, pad)))
        masks.append(F.pad(b["mask"], (0, pad)))  # mask 是 1D,padding 的位置補 0 代表「這裡是假資料,忽略」

    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),  # (B, 3, 512, 512)
        "hidden": torch.stack(hiddens),  # (B, max_len, hidden_dim)
        "mask": torch.stack(masks),      # (B, max_len)
        # [改動 2] 字串不需要(也不能)stack 成 tensor,保持 list 原樣傳給對比損失比對用
        "caption_keys": [b["caption_key"] for b in batch],
    }


# ==========================================
# [改動 2] 反塌縮對比損失
# ==========================================
def contrastive_term(cond_embed: torch.Tensor, caption_keys: list, temperature: float):
    """把同一個 batch 內「不同 caption」的 cond_embed 互相推開的 InfoNCE 式損失。

    定位說明(避免誤解):這不是 CLIP 那種「影像塔 vs 文字塔」的跨模態對齊——
    這裡相似度矩陣的兩邊都是 Projector 自己的輸出,對角線恆等於 1,所以它實際
    發揮的作用是 uniformity / 反塌縮:單純逼不同 caption 的條件向量分開。
    「這個向量對這張圖到底有沒有用」仍然由原本的 noise-MSE 負責,兩者互補,
    所以是相加而不是取代。

    Returns:
        (loss, offdiag_cos):後者是「不同 caption 配對的平均 cosine similarity」,
        純診斷用,對應 check_projector_embedding.py 量到的 0.996~0.999,
        訓練若真的有效,這個數字應該隨步數下降。
    """
    bsz = cond_embed.shape[0]
    device = cond_embed.device

    # 刻意攤平成 [B, num_queries*dim] 而不是對 query 取平均:診斷腳本量的就是攤平後的
    # cosine similarity,訓練用同一種算法,才能直接對照那個數字有沒有被推動。
    # 另外這裡一律轉 fp32——bf16 的 mantissa 只有 8 bits,相似度落在 0.99 附近時
    # 最小可分辨間隔(~0.004)跟我們要優化的差距同量級,在 bf16 下算會把梯度磨平。
    flat = F.normalize(cond_embed.reshape(bsz, -1).float(), dim=-1)
    sim = flat @ flat.T

    # 同一句 caption 出現在同一個 batch 裡(主要來自 CFG dropout 把 caption 換成 ""),
    # 這些非對角線配對是假負樣本,推開它們只會製造矛盾梯度,設成 -inf 從 softmax 排除。
    same = torch.tensor(
        [[a == b for b in caption_keys] for a in caption_keys],
        dtype=torch.bool, device=device,
    )
    eye = torch.eye(bsz, dtype=torch.bool, device=device)
    logits = (sim / temperature).masked_fill(same & ~eye, float("-inf"))
    loss = F.cross_entropy(logits, torch.arange(bsz, device=device))

    with torch.no_grad():
        different = ~same  # 只統計「真的不同 caption」的配對(同時也排除了對角線)
        offdiag_cos = sim[different].mean().item() if different.any() else float("nan")

    return loss, offdiag_cos


# ==========================================
# Stage 2: Diffusion 訓練迴圈
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Tier1 Projector Training")
    # --- 資料相關 ---
    parser.add_argument("--data_root", required=True)  # 圖片根目錄
    parser.add_argument("--metadata", required=True, help="jsonl:每行 {image, caption}")
    parser.add_argument("--val_metadata", default=None, help="驗證集 jsonl(格式同 --metadata);提供時定期算 validation loss")
    parser.add_argument("--val_every", type=int, default=None, help="每隔幾步算一次 validation loss(預設同 --save_every)")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="連續幾次 validation 沒有創新低就提前停止(0=關閉;需搭配 --val_metadata 才有作用)")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                        help="val loss 至少下降多少才算「有進步」,避免小數點雜訊誤判(預設 0)")
    parser.add_argument("--output_dir", default="checkpoints/tier1")  # checkpoint 輸出目錄
    parser.add_argument("--cache_dir", default=None, help="hidden states 快取(預設 output_dir/llm_cache)")
    parser.add_argument("--llm_path", default=None, help="LLM 權重路徑(預設環境變數 LLM_ENCODER_PATH)")
    parser.add_argument("--max_length", type=int, default=128)  # LLM tokenize 時的最大長度

    # --- 訓練超參數 ---
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)  # 梯度累積步數,等效放大 batch size 而不吃額外顯存
    parser.add_argument("--lr", type=float, default=1e-4)      # Projector 的學習率
    parser.add_argument("--lora_lr", type=float, default=1e-5)  # UNet LoRA 的學習率(通常要比 Projector 小)
    parser.add_argument("--max_steps", type=int, default=20000)  # 總訓練步數(以 optimizer step 計)
    parser.add_argument("--save_every", type=int, default=2000)  # 每隔幾步存一次 checkpoint
    parser.add_argument("--caption_dropout", type=float, default=0.1)  # CFG dropout 機率
    parser.add_argument("--seed", type=int, default=42)  # 隨機種子,確保實驗可重現

    # --- [改動 2] 反塌縮對比損失 ---
    parser.add_argument("--contrastive_weight", type=float, default=0.1,
                        help="對比損失權重 alpha,總損失 = mse + alpha * contrastive。"
                             "參考量級:mse 約 0.13,對比項起始值約 ln(batch_size)≈2.08,"
                             "alpha=0.1 時兩者貢獻相當(0.13 vs 0.21);設 0 可完全關閉退回原本行為")
    parser.add_argument("--contrastive_temp", type=float, default=0.07,
                        help="對比損失的 temperature(CLIP 慣用 0.07);越小對相似度差異越敏感")

    # --- Projector 架構參數 ---
    parser.add_argument("--num_queries", type=int, default=64)  # Projector 內部 query token 數量
    parser.add_argument("--depth", type=int, default=4)         # Projector 的層數/深度

    # --- UNet LoRA 選項 ---
    parser.add_argument("--unet_lora", action="store_true", help="同時訓練 UNet LoRA")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--resume", default=None, help="從 projector checkpoint 續訓")
    args = parser.parse_args()

    # 這支腳本一律在 GPU 上跑。沒有這道檢查的話,DEVICE 會靜默退回 "cpu",
    # 訓練仍然「正常」進行、log 也照寫,只是慢約 180 倍(實測 45 秒/步 vs GPU 的
    # 0.245 秒/步),等到 SLURM 時限到了才發現連第一個 checkpoint 都沒存到。
    # 寧可在這裡直接失敗,也不要浪費一整個 GPU 配額跑 CPU。
    if DEVICE == "cpu":
        raise SystemExit(
            "❌ torch.cuda.is_available() == False,拒絕在 CPU 上訓練。\n"
            "   SLURM 有配到 GPU 不代表 PyTorch 看得到它,請檢查:\n"
            "   1. sbatch 是否有 --gpus-per-node=1\n"
            "   2. 該節點的 GPU 是否正常(ssh <node> nvidia-smi)\n"
            "   3. conda 環境的 torch 是否為 CUDA 版(python -c 'import torch; print(torch.__version__)')"
        )

    # 固定隨機種子(PyTorch + Python random),讓每次訓練的初始化/資料順序/dropout 都可重現
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.output_dir, "llm_cache")

    # 讀取 jsonl 格式的 metadata,每行一筆 {"image": ..., "caption": ...}
    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f if line.strip()]  # 跳過空白行
    print(f"[Data] {len(metadata)} 筆訓練樣本")

    val_metadata = None
    if args.val_metadata:
        with open(args.val_metadata, "r", encoding="utf-8") as f:
            val_metadata = [json.loads(line) for line in f if line.strip()]
        print(f"[Data] {len(val_metadata)} 筆驗證樣本")

    # ---- Stage 1:hidden states 快取(結束後 LLM 已釋放) ----
    precompute_hidden_states(metadata, cache_dir, args.llm_path, args.max_length)
    if val_metadata:
        precompute_hidden_states(val_metadata, cache_dir, args.llm_path, args.max_length)
    # 隨便撈一筆快取,只是為了讀出 LLM hidden state 的維度大小,拿去初始化 Projector 的輸入層
    sample = torch.load(caption_cache_path(cache_dir, metadata[0]["caption"]))
    llm_hidden_dim = sample["hidden"].shape[1]
    print(f"[Data] LLM hidden_dim = {llm_hidden_dim}")

    # ---- Stage 2:載入 SD 元件(VAE/UNet 凍結) ----
    print("[SD] 載入 VAE / UNet / Scheduler ...")
    vae = AutoencoderKL.from_pretrained(SD_MODEL_ID, subfolder="vae").to(DEVICE)  # 圖片 <-> latent 空間的編解碼器
    unet = UNet2DConditionModel.from_pretrained(SD_MODEL_ID, subfolder="unet").to(DEVICE)  # 預測噪聲的主幹網路
    noise_scheduler = DDPMScheduler.from_pretrained(SD_MODEL_ID, subfolder="scheduler")  # 定義加噪/去噪的排程

    # 凍結 VAE 與 UNet 主幹,這次訓練只更新 Projector(以及可選的 UNet LoRA)
    vae.requires_grad_(False)
    vae.eval()
    unet.requires_grad_(False)

    cross_attn_dim = unet.config.cross_attention_dim   # SD1.5: 768,UNet cross-attention 期望的條件向量維度

    # Projector 負責把 LLM 的 hidden states 翻譯成 UNet 看得懂的條件向量(維度對齊 cross_attn_dim)
    projector = MakeupProjector(
        llm_hidden_dim=llm_hidden_dim,
        cross_attn_dim=cross_attn_dim,
        num_queries=args.num_queries,
        depth=args.depth,
    ).to(DEVICE)

    # 若指定 --resume,從既有 checkpoint 載入 Projector 權重,接續訓練
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        projector.load_state_dict(ckpt["state_dict"])
        print(f"[Resume] 已載入 {args.resume}")

    # optimizer 的參數群組:Projector 一定會訓練,UNet LoRA 視 --unet_lora 而定,兩者可用不同學習率
    param_groups = [{"params": projector.parameters(), "lr": args.lr}]
    lora_params = []  # [改動 1] 先宣告,讓下面分開裁剪梯度時不必再判斷 --unet_lora 有沒有開

    if args.unet_lora:
        from peft import LoraConfig

        # LoRA 只在 attention 的 q/k/v/輸出投影層插入低秩可訓練參數,UNet 主幹權重維持凍結
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(lora_config)
        lora_params = [p for p in unet.parameters() if p.requires_grad]  # 加完 adapter 後,只有 LoRA 參數 requires_grad=True
        param_groups.append({"params": lora_params, "lr": args.lora_lr})
        print(f"[LoRA] UNet LoRA 參數量 {sum(p.numel() for p in lora_params) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

    dataset = MakeupPairDataset(args.data_root, metadata, cache_dir, args.caption_dropout)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, drop_last=True,  # drop_last 避免最後一個不滿的 batch 造成 shape 不一致
    )

    # 驗證集不做 CFG dropout(caption_dropout=0.0),才能單純衡量語意條件化的效果
    val_loader = None
    if val_metadata:
        val_dataset = MakeupPairDataset(args.data_root, val_metadata, cache_dir, caption_dropout=0.0)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, collate_fn=collate_fn, drop_last=False,
        )
    val_every = args.val_every or args.save_every

    if args.early_stop_patience > 0 and val_loader is None:
        print("[EarlyStop] 警告:設了 --early_stop_patience 但沒有 --val_metadata,early stop 不會生效")

    n_train = sum(p.numel() for g in param_groups for p in g["params"])
    print(f"[Train] 可訓練參數共 {n_train / 1e6:.2f}M,目標 {args.max_steps} steps")

    # ---- loss 紀錄:每個 step 寫一行到 CSV,才能事後畫訓練曲線 ----
    # --resume 續訓時用 append,避免蓋掉先前的紀錄;全新訓練則覆寫
    log_path = os.path.join(args.output_dir, "train_log.csv")
    log_file = open(log_path, "a" if args.resume else "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if log_file.tell() == 0:
        # [改動 1][改動 2] 欄位從原本的 (step, loss) 擴充,才能事後分辨
        # 「總損失有沒有降」是來自哪一項,以及兩組參數的梯度量級差多少。
        # offdiag_cos 是這次最關鍵的觀察值:塌縮時 ≈0.999,訓練有效應隨步數下降。
        log_writer.writerow([
            "step", "loss", "mse_loss", "contrastive_loss",
            "offdiag_cos", "proj_grad_norm", "lora_grad_norm",
        ])

    val_log_file = val_log_writer = None
    if val_loader is not None:
        val_log_path = os.path.join(args.output_dir, "val_log.csv")
        val_log_file = open(val_log_path, "a" if args.resume else "w", newline="", encoding="utf-8")
        val_log_writer = csv.writer(val_log_file)
        if val_log_file.tell() == 0:
            # [改動 2] 多記一欄 held-out caption 的 cond_embed 非對角線 cosine similarity,
            # 用來確認「能區分不同 caption」這件事有沒有推廣到沒訓練過的句子上
            val_log_writer.writerow(["step", "val_loss", "val_offdiag_cos"])

    # ---- 訓練迴圈(bf16 autocast,H100 原生支援) ----
    projector.train()
    unet.train() if args.unet_lora else unet.eval()  # 只有訓練 LoRA 時才需要 UNet 處於 train 模式(啟用 dropout 等)

    global_step = 0
    best_val_loss = float("inf")
    no_improve_evals = 0  # 連續幾次 validation 沒有創新低(單位是「評估點」,不是 step)
    early_stop_triggered = False
    pbar = tqdm(total=args.max_steps)
    # 外層 while + 內層 for:資料集跑完一輪(epoch)就自動從頭 shuffle 再來一輪,
    # 直到累積達到 max_steps 為止,所以這裡不是用 epoch 數控制,而是用 step 數控制
    while global_step < args.max_steps:
        for batch in loader:
            if global_step >= args.max_steps:
                break

            pixel_values = batch["pixel_values"].to(DEVICE)
            hidden = batch["hidden"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            bsz = pixel_values.shape[0]

            # 把圖片編碼進 VAE 的 latent 空間;VAE 凍結所以不需要梯度,用 no_grad 省顯存
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor  # SD 慣例:縮放到訓練時的數值範圍

            # 標準 DDPM 訓練目標:隨機取一個時間步,加對應強度的噪聲,讓模型學會預測這個噪聲
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=DEVICE
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # bf16 混合精度前向:H100 對 bf16 有原生加速,且比 fp16 更穩定(不需要 loss scaling)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                # 核心:LLM hidden states -> Projector -> UNet 看得懂的條件向量
                cond_embed = projector(hidden, timesteps, attention_mask=mask)
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states=cond_embed,
                    return_dict=False,
                )[0]
                # 原本的訓練目標:讓 UNet 預測出的噪聲跟實際加的噪聲越接近越好(MSE)
                mse_loss = F.mse_loss(noise_pred.float(), noise.float())

            # [改動 2] 對比項刻意放在 autocast 區塊「外面」,強制用 fp32 計算,
            # 理由見 contrastive_term() 的說明(bf16 解析度不足以分辨 0.99 附近的差距)。
            contrastive_loss, offdiag_cos = contrastive_term(
                cond_embed, batch["caption_keys"], args.contrastive_temp
            )
            # 總損失 = 原本的重建 MSE + alpha * 反塌縮對比項。
            # 除以 grad_accum 是為了讓多步累積後的梯度總量等同於一次大 batch 的梯度
            loss = (mse_loss + args.contrastive_weight * contrastive_loss) / args.grad_accum

            loss.backward()

            # 梯度累積:每 grad_accum 步才真正更新一次參數,等效放大 batch size
            proj_grad_norm = lora_grad_norm = ""  # 非累積邊界的 step 不做裁剪,CSV 留空
            if (global_step + 1) % args.grad_accum == 0:
                # [改動 1] 梯度裁剪從「兩組參數混在一起算 total norm」改成「各自分開裁」。
                # 混在一起時,clip_grad_norm_ 會用同一個縮放係數縮放所有參數,而那個係數
                # 由梯度較大的一方(LoRA 直接改 UNet attention,對 pixel loss 的影響比隔了
                # 一層 cross-attention 的 Projector 直接得多)主導,等於把 Projector 本來
                # 就微弱的梯度又壓小一輪。分開裁之後,Projector 的更新幅度只取決於自己。
                # clip_grad_norm_ 回傳的是「裁剪前」的 total norm,正好拿來當診斷數值:
                # 若 proj_grad_norm 長期遠小於 lora_grad_norm,就證實了上述壓制效應。
                proj_grad_norm = torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0).item()
                if lora_params:
                    lora_grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, 1.0).item()
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            step_loss = loss.item() * args.grad_accum  # 還原成未除以 grad_accum 前的實際 loss
            pbar.update(1)
            pbar.set_postfix(loss=f"{step_loss:.4f}", cos=f"{offdiag_cos:.4f}")

            # 每個 step 都寫一行,flush 確保就算 job 被砍掉(如 SLURM 時限)也不會遺失
            log_writer.writerow([
                global_step, step_loss, mse_loss.item(), contrastive_loss.item(),
                offdiag_cos, proj_grad_norm, lora_grad_norm,
            ])
            log_file.flush()

            # 每隔 save_every 步,或訓練跑到最後一步時,存一次 checkpoint
            if global_step % args.save_every == 0 or global_step == args.max_steps:
                save_checkpoint(projector, unet, args, global_step)

                if val_loader is not None and (global_step % val_every == 0 or global_step == args.max_steps):
                    v_loss, v_cos = compute_val_loss(projector, unet, vae, noise_scheduler, val_loader)
                    val_log_writer.writerow([global_step, v_loss, v_cos])
                    val_log_file.flush()
                    print(f"\n[Val] step {global_step} → loss {v_loss:.4f},"
                          f"cond_embed 非對角線 cos {v_cos:.4f}(塌縮時 ≈0.999,越低代表越能區分不同 caption)")
                    # 算完驗證集後要切回 train 模式,不然接下來的訓練 step 會用 eval 模式跑(BN/dropout 行為錯誤)
                    projector.train()
                    if args.unet_lora:
                        unet.train()

                    if v_loss < best_val_loss - args.early_stop_min_delta:
                        best_val_loss = v_loss
                        no_improve_evals = 0
                        save_best_checkpoint(args, global_step)
                        print(f"[Val] 新最佳 val loss {best_val_loss:.4f}(step {global_step}),已存為 *_best.pt")
                    else:
                        no_improve_evals += 1
                        patience_str = args.early_stop_patience if args.early_stop_patience > 0 else "∞"
                        print(f"[Val] 連續 {no_improve_evals}/{patience_str} 次沒有創新低(目前最佳 {best_val_loss:.4f})")

                    if args.early_stop_patience > 0 and no_improve_evals >= args.early_stop_patience:
                        print(f"\n🛑 [EarlyStop] val loss 連續 {no_improve_evals} 次沒有改善,"
                              f"在 step {global_step} 提前停止(最佳 val loss {best_val_loss:.4f})")
                        early_stop_triggered = True
                        break

        if early_stop_triggered:
            break

    pbar.close()
    log_file.close()
    if val_log_file is not None:
        val_log_file.close()
    print(f"✅ 訓練完成,checkpoints 在 {args.output_dir}/")
    print(f"   train loss 紀錄:{log_path}")
    if val_loader is not None:
        print(f"   val loss 紀錄:{val_log_path}")


@torch.no_grad()
def compute_val_loss(projector, unet, vae, noise_scheduler, val_loader):
    """對整個 val_loader 跑一次平均 noise-prediction MSE,不更新任何參數。

    切到 eval 模式、caption_dropout=0(由呼叫端的 val_dataset 保證),
    才能單純衡量「這個 checkpoint 對沒訓練過的 caption/圖片配對翻譯得準不準」,
    不會被 CFG 的隨機無條件分支或 dropout 雜訊干擾。

    [改動 2] 回傳值從單一 val loss 改成 (val_mse, val_offdiag_cos)。
    後者是 held-out caption 的 cond_embed 非對角線 cosine similarity,用來看
    「能區分不同 caption」有沒有推廣到沒訓練過的句子——這是判斷塌縮有沒有真正
    解除的關鍵指標。注意 best checkpoint 的挑選仍然只看 val_mse(維持原行為),
    val_offdiag_cos 目前純粹是觀察用。
    """
    projector.eval()
    unet.eval()

    total_loss, n_batches = 0.0, 0
    total_cos, n_cos = 0.0, 0
    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(DEVICE)
        hidden = batch["hidden"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        bsz = pixel_values.shape[0]

        latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=DEVICE)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            cond_embed = projector(hidden, timesteps, attention_mask=mask)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=cond_embed, return_dict=False)[0]
            loss = F.mse_loss(noise_pred.float(), noise.float())

        total_loss += loss.item()
        n_batches += 1

        # [改動 2] 只取診斷用的 cosine similarity,不需要 loss 本身。
        # val_loader 的 drop_last=False,最後一批可能只有 1 筆,那時沒有任何配對可比
        # (offdiag_cos 會是 nan),直接跳過不列入平均。
        _, cos = contrastive_term(cond_embed, batch["caption_keys"], 1.0)
        if cos == cos:  # nan != nan,用這個判斷排除 nan
            total_cos += cos
            n_cos += 1

    return total_loss / max(n_batches, 1), (total_cos / n_cos if n_cos else float("nan"))


def save_checkpoint(projector, unet, args, step):
    """把目前的訓練狀態存成 checkpoint 檔案。

    Projector 權重與(可選的)UNet LoRA 權重分開存成兩個檔案,
    方便日後只載入其中一個(例如只想抽換 Projector,LoRA 沿用舊的)。
    """
    ckpt = {
        "state_dict": projector.state_dict(),
        # 把架構參數一起存下來,推論時不需要另外手動指定,直接從 checkpoint 還原模型結構
        "config": {
            "llm_hidden_dim": projector.input_proj.in_features,
            "cross_attn_dim": projector.cross_attn_dim,
            "num_queries": projector.num_queries,
            "depth": len(projector.blocks),
        },
        "step": step,
    }
    path = os.path.join(args.output_dir, f"projector_step{step}.pt")
    torch.save(ckpt, path)

    if args.unet_lora:
        # 只挑出名字含 "lora" 的參數存檔,UNet 主幹本身沒變、不用重複存一份完整權重
        lora_sd = {k: v for k, v in unet.state_dict().items() if "lora" in k}
        torch.save(lora_sd, os.path.join(args.output_dir, f"unet_lora_step{step}.pt"))

    print(f"\n[Save] step {step} → {path}")


def save_best_checkpoint(args, step):
    """把剛剛存好的 step checkpoint 複製一份成 *_best.pt。

    複製而非重新 torch.save,因為 save_checkpoint() 已經在同一個 save_every
    邊界存過這個 step 的檔案了,直接拷貝可以避免重複序列化一次模型權重。
    """
    src = os.path.join(args.output_dir, f"projector_step{step}.pt")
    dst = os.path.join(args.output_dir, "projector_best.pt")
    shutil.copy2(src, dst)

    if args.unet_lora:
        lora_src = os.path.join(args.output_dir, f"unet_lora_step{step}.pt")
        lora_dst = os.path.join(args.output_dir, "unet_lora_best.pt")
        shutil.copy2(lora_src, lora_dst)


if __name__ == "__main__":
    main()
