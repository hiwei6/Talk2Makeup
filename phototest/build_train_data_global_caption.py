"""build_train_data_global_caption.py

把已標註好的 makeup metadata jsonl(ai_anno.py / ai_anno_fixed.py 產生的,
只要每行有 "metadata" 欄位、裡面有 "dense_caption" 即可)轉成
../train_projector.py 可直接吃的訓練 jsonl:

    {"image": "<相對 --data_root 的路徑>", "caption": "<整體 dense_caption>"}

"caption" 只取整體的 dense_caption,不含 foundation/eyes/lips/blush/brows
各部位的結構化細節——如果要 caption 裡包含各部位的資訊,改用同資料夾的
build_train_data_regional_caption.py。

純粹是本地格式轉換,不會呼叫任何 API,可重複執行、不花錢、不用等 RPM 限制。

用法:
    python build_train_data_global_caption.py \
        --input makeup_metadata_test100.jsonl \
        --output train_data_global_caption.jsonl
"""

import argparse
import json
import os


def resolve_image_path(record: dict, image_field_prefix: str) -> str:
    """優先用 record 裡既有的 "image"(例如 ai_anno_fixed.py 產生的),
    否則用 file_name/image_id 拼出相對路徑。"""
    if record.get("image"):
        return record["image"]
    file_name = record.get("file_name") or f"{record.get('image_id')}.png"
    return os.path.join(image_field_prefix, file_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="把已標註的 metadata jsonl 轉成只用整體 dense_caption 的 projector 訓練資料"
    )
    parser.add_argument(
        "--input", default="./makeup_metadata_test100.jsonl",
        help="已標註好的 metadata jsonl(需含 'metadata' 欄位)",
    )
    parser.add_argument("--output", default="./train_data_global_caption.jsonl", help="輸出的訓練用 jsonl")
    parser.add_argument(
        "--image_field_prefix", default="test_images",
        help="record 裡沒有 'image' 欄位時,拼路徑用的前綴目錄;需對應 train_projector.py 的 --data_root",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    n_written, n_skipped = 0, 0
    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {})
            caption = (metadata.get("dense_caption") or record.get("caption") or "").strip()
            if not caption:
                n_skipped += 1
                continue

            out_record = {
                "image": resolve_image_path(record, args.image_field_prefix),
                "caption": caption,
                "image_id": record.get("image_id"),
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[Done] 寫入 {n_written} 筆,略過(無 caption){n_skipped} 筆 → {args.output}")


if __name__ == "__main__":
    main()
