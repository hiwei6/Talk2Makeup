"""
MakeupProjector: 連接 LLaMA-3 語意理解與 SD1.5 UNet 條件化輸入的橋接模組。

背景（Tier1 架構）：
    現況 pipeline 讓 LLM 把理解壓縮成幾個英文關鍵字（JSON 的 prompt 欄位），
    再交給 CLIP text encoder 重新編碼餵給 UNet。這裡的 Projector 取代 CLIP
    的角色，直接把 LLaMA-3 的 hidden states 轉換成 UNet cross-attention
    吃得懂的條件向量，讓兩個模型之間可以端到端訓練、梯度互通。

設計參考 ELLA（Efficient Large Language Model Adapter）的 Timestep-aware
Semantic Connector：用一組固定數量的可學習 query token，對 LLM 的
hidden states 做 Perceiver 式 cross-attention 重新採樣成固定長度輸出，
並讓目前的 diffusion timestep 調變每一層的輸出，使早期步驟（構圖/整體
風格）與晚期步驟（局部質感細節）可以從同一組 LLM 理解中取出不同重點。
"""

import math

import torch
import torch.nn as nn


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """標準 diffusion 正弦時間步編碼（與 diffusers 的實作等價）。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    args = timesteps.float().reshape(-1, 1) * freqs.reshape(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, freq_dim: int, out_dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_timestep_embedding(timesteps, self.freq_dim))


class PerceiverAttentionBlock(nn.Module):
    """可學習 query 對 LLM hidden states 做 cross-attention，並以 timestep 做 AdaLN 調變。"""

    def __init__(self, dim: int, num_heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

        # AdaLN-Zero: 由 timestep embedding 產生 shift/scale，初始輸出為 0
        # 讓訓練剛開始時這個 block 接近恆等映射，穩定收斂。
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 4),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        t_emb: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, shift_ff, scale_ff = self.adaLN_modulation(t_emb).chunk(4, dim=-1)

        q = self._modulate(self.norm_q(queries), shift_attn, scale_attn)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.attn(q, kv_n, kv_n, key_padding_mask=key_padding_mask, need_weights=False)
        queries = queries + attn_out

        ff_in = self._modulate(self.norm_ff(queries), shift_ff, scale_ff)
        queries = queries + self.ff(ff_in)
        return queries


class MakeupProjector(nn.Module):
    """
    把 LLaMA-3 hidden states 投影成 SD1.5 UNet cross-attention 的條件向量。

    Args:
        llm_hidden_dim: LLM hidden state 維度（LLaMA-3-8B 為 4096）
        cross_attn_dim: UNet cross-attention 維度（SD1.5 為 768）
        num_queries: 重採樣後的固定 token 數（取代 CLIP 原本的 77）
        depth: PerceiverAttentionBlock 堆疊層數
        num_heads: attention head 數
        time_freq_dim: 時間步正弦編碼的頻率維度
    """

    def __init__(
        self,
        llm_hidden_dim: int = 4096,
        cross_attn_dim: int = 768,
        num_queries: int = 64,
        depth: int = 4,
        num_heads: int = 8,
        time_freq_dim: int = 256,
    ):
        super().__init__()
        self.cross_attn_dim = cross_attn_dim
        self.num_queries = num_queries

        self.input_proj = nn.Linear(llm_hidden_dim, cross_attn_dim)
        self.queries = nn.Parameter(torch.randn(1, num_queries, cross_attn_dim) * 0.02)
        self.time_embed = TimestepEmbedding(time_freq_dim, cross_attn_dim)

        self.blocks = nn.ModuleList(
            [PerceiverAttentionBlock(cross_attn_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(cross_attn_dim)

    def forward(
        self,
        llm_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            llm_hidden_states: [B, L, llm_hidden_dim]，LLaMA-3 最後一層 hidden states
            timestep: [B]，目前 diffusion 反向採樣的時間步 t
            attention_mask: [B, L]，可選，1=有效 token、0=padding

        Returns:
            cond_embed: [B, num_queries, cross_attn_dim]，可直接取代
            DreamMakeupPipeline 裡原本 CLIP 產生的 prompt_embeds
        """
        if timestep.dim() == 0:
            timestep = timestep.expand(llm_hidden_states.shape[0])

        kv = self.input_proj(llm_hidden_states)
        key_padding_mask = None if attention_mask is None else attention_mask == 0

        t_emb = self.time_embed(timestep)
        queries = self.queries.expand(llm_hidden_states.shape[0], -1, -1)

        for block in self.blocks:
            queries = block(queries, kv, t_emb, key_padding_mask=key_padding_mask)

        return self.norm_out(queries)


if __name__ == "__main__":
    # 形狀自我檢查：模擬 LLaMA-3-8B hidden states 接到 SD1.5 cross-attention 維度
    projector = MakeupProjector()
    batch, seq_len = 2, 40
    dummy_hidden = torch.randn(batch, seq_len, 4096)
    dummy_mask = torch.ones(batch, seq_len)
    dummy_mask[1, 30:] = 0  # 模擬第二筆樣本有 padding
    dummy_t = torch.randint(0, 1000, (batch,))

    out = projector(dummy_hidden, dummy_t, attention_mask=dummy_mask)
    n_params = sum(p.numel() for p in projector.parameters())

    print(f"輸入 hidden_states: {tuple(dummy_hidden.shape)}")
    print(f"輸出 cond_embed:    {tuple(out.shape)}")
    print(f"可訓練參數量:        {n_params / 1e6:.2f}M")
    assert out.shape == (batch, projector.num_queries, projector.cross_attn_dim)
    print("✅ 形狀檢查通過")
