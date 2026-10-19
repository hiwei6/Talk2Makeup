"""
LLMTextEncoder: 以 transformers 本地載入 LLM,抽取 hidden states 供 Projector 使用。

Tier1 架構中,LLM 不再只透過 Ollama API 吐出 JSON 文字——那條 API 只回傳
文字,拿不到模型內部向量。這裡改用 transformers 直接載入權重,對輸入文字
跑一次 forward,取最後一層 hidden states 作為 Projector 的輸入。

注意:
- LLM 權重完全凍結,永遠在 torch.no_grad() 下執行,不參與訓練。
- 用 AutoModel(不含 lm_head)而非 AutoModelForCausalLM,省下輸出層的記憶體。
- 顏色精準路徑(Ollama JSON → LAB 混色)不經過這裡,維持原樣。
"""

import os

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_LLM_PATH = os.getenv("LLM_ENCODER_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")


class LLMTextEncoder:
    """
    Args:
        model_path: HuggingFace 模型名稱或本地權重路徑(預設讀環境變數 LLM_ENCODER_PATH)
        device: 執行裝置
        dtype: 權重精度(H100 上 fp16/bf16 皆可)
        max_length: tokenizer 截斷長度;妝容描述通常遠短於此
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_length: int = 128,
    ):
        self.model_path = model_path or DEFAULT_LLM_PATH
        self.device = device
        self.max_length = max_length

        print(f"[LLMEncoder] 載入 {self.model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(self.model_path, torch_dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.hidden_size = self.model.config.hidden_size
        print(f"[LLMEncoder] 載入完成,hidden_size={self.hidden_size}")

    @torch.no_grad()
    def encode(self, texts):
        """
        Args:
            texts: str 或 list[str]

        Returns:
            hidden_states: [B, L, hidden_size] float32(Projector 以 fp32 運算)
            attention_mask: [B, L],1=有效 token、0=padding
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        out = self.model(**inputs)
        hidden = out.last_hidden_state.float()
        return hidden, inputs["attention_mask"]

    def free(self):
        """釋放 LLM 佔用的 GPU 記憶體(hidden states 取完後即可呼叫)。"""
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[LLMEncoder] 已釋放模型記憶體")
