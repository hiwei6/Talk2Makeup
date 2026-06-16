#!/bin/bash
# 啟動 talk2makeup：確保 Ollama 在跑並預熱模型，再啟動 Gradio app
# 用法：
#   bash start.sh          （自動使用 llm conda env）
#   conda activate llm && python app.py  （手動，Ollama 需已啟動）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3}"

# 決定 Python：若 llm env 已啟動就用當前 python，否則用完整路徑
LLM_PYTHON="$(conda run -n llm which python 2>/dev/null || echo /home/willyliu0623/miniconda3/envs/llm/bin/python)"

# 啟動 Ollama (若尚未執行)
if ! curl -s http://127.0.0.1:11434/ > /dev/null 2>&1; then
    echo "[1/3] 啟動 Ollama..."
    OLLAMA_MODELS=~/.ollama/models \
    OLLAMA_KEEP_ALIVE=1h \
    LD_LIBRARY_PATH=~/lib/ollama:~/lib/ollama/cuda_v12:$LD_LIBRARY_PATH \
    ~/bin/ollama serve > "$SCRIPT_DIR/ollama.log" 2>&1 &
    for i in $(seq 1 15); do
        sleep 1
        curl -s http://127.0.0.1:11434/ > /dev/null 2>&1 && echo "      Ollama 已就緒" && break
        echo "      等待 Ollama ($i/15)..."
    done
else
    echo "[1/3] Ollama 已在執行中"
fi

# 預熱：觸發模型載入（從 NFS 約需 3 分鐘，避免第一次使用時卡住）
LOADED=$(curl -s http://127.0.0.1:11434/api/ps | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('models',[])))" 2>/dev/null)
if [ "$LOADED" = "0" ]; then
    echo "[2/3] 預熱 $OLLAMA_MODEL（首次載入需 ~3 分鐘，請等待）..."
    curl -s --max-time 400 http://127.0.0.1:11434/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$OLLAMA_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
        > /dev/null 2>&1
    echo "      模型預熱完成"
else
    echo "[2/3] $OLLAMA_MODEL 已在 GPU，跳過預熱"
fi

# 啟動 Gradio app
echo "[3/3] 啟動 talk2makeup..."
"$LLM_PYTHON" "$SCRIPT_DIR/app.py"
