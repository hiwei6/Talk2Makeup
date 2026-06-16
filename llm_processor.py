import json
import os
from openai import OpenAI

# ==========================================
# LLM Provider 設定
# 切換方式：設定環境變數 LLM_PROVIDER
#   Ollama (預設): LLM_PROVIDER=ollama  OLLAMA_MODEL=llama3
#   Claude:        LLM_PROVIDER=claude  ANTHROPIC_API_KEY=sk-...
# ==========================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

SYSTEM_PROMPT = """
You are a professional Makeup Artistry AI. Your task is to translate user requests
into a structured JSON configuration for a computer vision rendering engine.

## Rules:
1. Output MUST be a single, valid JSON object. No markdown, no extra text.
2. Use 'force_lightness': true for extreme makeup (clown, avatar, fantasy) to override skin texture.
3. Use 'dilation' (0-30) to expand makeup beyond natural boundaries (e.g. smoky eye = 15).
4. Choose 'placement' from: ["apple", "cheekbone", "sunburn", "igari"].
5. Colors are RGB arrays [R, G, B] with values 0-255.
6. 'alpha' controls opacity: 0.0 = skip this part, 0.3 = light, 0.8 = heavy.

## JSON Schema:
{
  "description": "A detailed Stable Diffusion prompt (English) focusing on makeup textures.",
  "foundation": {"color": [R, G, B], "alpha": 0.0-1.0, "force_lightness": bool},
  "lips": {"color": [R, G, B], "alpha": 0.0-1.0, "dilation": 0-30},
  "eyes": {"color": [R, G, B], "alpha": 0.0-1.0, "dilation": 0-20},
  "blush": {"color": [R, G, B], "alpha": 0.0-1.0, "placement": "string", "scale": 0.5-2.0},
  "hair": {"color": [R, G, B], "alpha": 0.0-1.0}
}
"""

DEFAULT_PLAN = {
    "description": "Natural daily makeup, soft and clean",
    "foundation": {"color": [255, 230, 210], "alpha": 0.0, "force_lightness": False},
    "lips": {"color": [210, 80, 80], "alpha": 0.3, "dilation": 0},
    "eyes": {"color": [60, 40, 20], "alpha": 0.0, "dilation": 0},
    "blush": {"color": [255, 150, 150], "alpha": 0.2, "placement": "apple", "scale": 1.0},
    "hair": {"color": [0, 0, 0], "alpha": 0.0},
}


def get_makeup_plan_from_llm(user_input: str) -> dict:
    if LLM_PROVIDER == "claude":
        return _call_claude(user_input)
    return _call_ollama(user_input)


def _call_ollama(user_input: str) -> dict:
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    print(f"[Ollama/{OLLAMA_MODEL}] 處理指令: {user_input}")
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Request: {user_input}"},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[Ollama] 失敗: {e}，使用預設值")
        return DEFAULT_PLAN


def _call_claude(user_input: str) -> dict:
    # 未來擴充：安裝 anthropic 後取消此區塊的 import 與實作
    # pip install anthropic
    try:
        import anthropic  # noqa: F401
    except ImportError:
        raise RuntimeError("請先安裝 anthropic: pip install anthropic")

    import anthropic as ant

    client = ant.Anthropic()
    print(f"[Claude] 處理指令: {user_input}")
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Request: {user_input}"}],
        )
        raw = message.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[Claude] 失敗: {e}，使用預設值")
        return DEFAULT_PLAN
