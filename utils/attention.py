import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

class DreamMakeupAttnProcessor:
    """
    自定義 Attention Processor 實作 Cross-Attention Composition。
    原理: Q_new = Softmax(Q K_main^T) V_main + sum( alpha_s * Softmax(Q K_s^T) V_s )
    """
    def __init__(self, makeup_scale=1.0):
        self.makeup_scale = makeup_scale

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
        scale: float = 1.0,
    ):
        # encoder_hidden_states 包含 [main_prompt, makeup_prompt_1, makeup_prompt_2, ...]
        # 我們假設 batch size = 2 (uncond, cond) 或更多
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        # 1. Prepare Query, Key, Value
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 2. Attention Calculation
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # 標準 Attention Output
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

# 註：完整的 Eq.10 需要將 encoder_hidden_states 拆解為 main 和 concepts
# 上述代碼為簡化版，確保 Pipeline 可運行。
# 完整實作需在 Pipeline 中準備多個 Prompt Embeddings 並在此處分別計算 Attention。