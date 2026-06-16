import gradio as gr
import sys

print("正在啟動 Gradio (本地模式)...")

def greet(name):
    return "Hello " + name + "!"

with gr.Blocks() as demo:
    gr.Markdown("# 測試成功！")
    inp = gr.Textbox(placeholder="輸入名字")
    out = gr.Textbox()
    inp.change(greet, inp, out)

if __name__ == "__main__":
    try:
        # 關鍵設定：
        # 1. share=False (不連外網，解決卡住/報錯主因)
        # 2. server_name="0.0.0.0" (允許 SSH 通道連入)
        # 3. debug=True (顯示詳細錯誤)
        print("準備啟動伺服器，請注意看下方的 URL...")
        demo.launch(
            share=True, 
            server_name="0.0.0.0", 
            server_port=8888,
            debug=True
        )
    except Exception as e:
        print(f"啟動發生錯誤: {e}")