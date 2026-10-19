# Tier1:LLM-conditioned Diffusion

本目錄在 talk2makeup_0719 的基礎上,加入 Tier1 架構:以本地 LLM 的 hidden
states 經可訓練 Projector 直接條件化 UNet,取代「JSON prompt 欄位 → CLIP
text encoder」這條會壓縮語意的路徑。設計參考 ELLA(arXiv:2403.05135)的
Timestep-Aware Semantic Connector 與 DiT(arXiv:2212.09748)的 AdaLN-Zero。

## 檔案總覽

| 檔案 | 角色 | 相對原版 |
|:---|:---|:---|
| `models/projector.py` | MakeupProjector:Perceiver 式 resampler + AdaLN timestep 調變,約 41.8M 參數,唯一從零訓練的模組 | 新增 |
| `models/llm_encoder.py` | 以 transformers 本地載入 LLM(凍結),抽取最後一層 hidden states | 新增 |
| `pipeline_dreammakeup_tier1.py` | 繼承 `DreamMakeupPipeline`,條件向量改為每個時間步呼叫 Projector;DDIM 反推、插值採樣邏輯不變 | 新增 |
| `train_projector.py` | 訓練腳本:凍結 LLM/VAE/UNet,標準噪聲預測 MSE,可選 UNet LoRA | 新增 |
| `inference_tier1.py` | 端到端推論:顏色路徑沿用原版,風格路徑走 Tier1 | 新增 |
| `pipeline_dreammakeup.py`、`inference.py`、`app.py` | 原版流程 | **完全未動** |
| BiSeNet 分割 + LAB 混色 + z'_0 插值錨點 | 顏色精準路徑 | **完全未動** |

## 架構

```
使用者文字描述
     │
     ├────────────────────────────┐
     ▼                            ▼
 Ollama JSON(顏色路徑,不變)    本地 LLM(凍結)→ hidden states [B,L,4096]
     │                            │
     ▼                            ▼
 BiSeNet + LAB 混色           MakeupProjector(訓練)→ cond_embed [B,64,768]
     │                            │(每個去噪步 t 重新計算,timestep-aware)
     ▼                            ▼
 VAE encode → z'_0 ─────────▶ UNet(凍結,可選 LoRA)
                                  │
                                  ▼
                     早停 DDIM 反推 + 插值採樣(不變)
                                  │
                                  ▼
                              最終影像
```

## 訓練

資料格式:`metadata.jsonl`,每行一筆,`image` 路徑相對於 `--data_root`:

```json
{"image": "images/0001.jpg", "caption": "微醺感日系桃花妝,水潤蜜桃色唇妝"}
```

指令(conda 環境 `llm`):

```bash
export LLM_ENCODER_PATH=/path/to/Meta-Llama-3-8B-Instruct   # 或 HF 名稱
python train_projector.py \
    --data_root data/makeup \
    --metadata  data/makeup/metadata.jsonl \
    --output_dir checkpoints/tier1 \
    --batch_size 8 --max_steps 20000 \
    --unet_lora
```

流程分兩階段:先把所有 caption 的 hidden states 快取到
`output_dir/llm_cache/`(含空字串,供 10% caption dropout 訓練 CFG 無條件
分支),隨即釋放 LLM;再載入 SD1.5 跑 diffusion 訓練。VAE/UNet 主幹凍結,
只更新 Projector(lr 1e-4)與可選的 UNet LoRA(lr 1e-5),bf16 autocast。

產出:`projector_step*.pt`(含結構設定,推論端自動還原)與
`unet_lora_step*.pt`。

## 推論

```bash
python inference_tier1.py -i assets/input.jpg \
    -p "幫我畫微醺感的日系桃花妝,口紅要水潤的蜜桃色" \
    --projector checkpoints/tier1/projector_step20000.pt \
    --unet_lora checkpoints/tier1/unet_lora_step20000.pt
```

- `-p` 的原始描述同時餵給兩條路徑:Ollama 解析 JSON 供顏色路徑;原文直接
  經 LLM → Projector 供 Tier1 條件化(不經 JSON prompt 欄位壓縮,這是
  Tier1 的核心)。
- `-j` 讀 JSON 檔跳過 Ollama 時,建議搭配 `--style_text` 提供條件文字,
  否則退回用 JSON 內的英文關鍵字拼接。
- 記憶體:LLM 取完 hidden states 立即釋放,再載入 SD;單張 H100 足夠。

## 已知限制與注意事項

- Projector 未訓練前,`inference_tier1.py` 產出的圖**沒有意義**——條件向量
  是隨機初始化的翻譯結果。必須先完成訓練。
- LLaMA-3 是 decoder-only(causal attention),單一 token 的 hidden state
  看不到後文;Projector 的 cross-attention 可部分緩解,但效果差異需實驗
  驗證(參見 LLM2Vec, arXiv:2404.05961)。
- 顏色精準度仍由 z'_0 插值錨點保證,Tier1 只負責風格/質感語意;兩者分工
  與原版一致。
- 原版 `assets/BKG1.safetensors`(人像 LoRA)未複製到本目錄;若要沿用,
  注意它與新訓練的 UNet LoRA 疊加可能互相干擾,建議擇一。
