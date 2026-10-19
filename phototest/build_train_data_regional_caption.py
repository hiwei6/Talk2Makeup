"""build_train_data_regional_caption.py

把已標註好的 makeup metadata jsonl(ai_anno.py / ai_anno_fixed.py 產生的)
轉成 ../train_projector.py 可直接吃的訓練 jsonl,但 "caption" 不只放整體
dense_caption,而是把 foundation / eyes / lips / blush / brows 五個部位的
結構化標註(枚舉值 + description)逐一組成文字,最後再接上原本的
dense_caption 收尾——確保每個部位的細節都會實際出現在丟進 LLM 的文字裡,
而不是只靠 dense_caption 有沒有剛好提到。

如果只想要跟原本一樣、只用整體 caption,改用同資料夾的
build_train_data_global_caption.py。

純粹是本地格式轉換,不會呼叫任何 API,可重複執行、不花錢。

注意:組出來的 caption 比純 dense_caption 長很多,訓練時 LLM tokenize
的 --max_length(train_projector.py 預設 128)可能不夠,建議視實際 token
數拉高(例如 --max_length 256)。

用法:
    python build_train_data_regional_caption.py \
        --input makeup_metadata_test100.jsonl \
        --output train_data_regional_caption.jsonl
"""

import argparse
import json
import os


def readable(value) -> str:
    """把 schema 裡的 snake_case 列舉值轉成較自然的文字(rose_pink -> rose pink)。"""
    if isinstance(value, bool):
        return "with" if value else "without"
    return str(value).replace("_", " ")


def build_regional_caption(metadata: dict) -> str:
    """把五個部位的結構化欄位 + dense_caption 組成一段完整的訓練用文字。"""
    parts = []

    parts.append(
        f"Overall style: {readable(metadata.get('overall_style'))}, "
        f"intensity {metadata.get('intensity_score')}/5."
    )

    f = metadata.get("foundation") or {}
    parts.append(
        f"Foundation: {readable(f.get('coverage'))} coverage, {readable(f.get('finish'))} finish, "
        f"tone {readable(f.get('tone_match'))}. {(f.get('description') or '').strip()}"
    )

    e = metadata.get("eyes") or {}
    aegyo = "with aegyo-sal highlight" if e.get("aegyo_sal_highlight") else "without aegyo-sal highlight"
    parts.append(
        f"Eyes: {readable(e.get('eyeshadow_tone'))} eyeshadow, {readable(e.get('eyeshadow_finish'))} finish, "
        f"{readable(e.get('eyeliner_style'))} eyeliner, {aegyo}."
    )

    l = metadata.get("lips") or {}
    parts.append(
        f"Lips: {readable(l.get('color_category'))}, {readable(l.get('texture'))} texture, "
        f"{readable(l.get('gradient_style'))}."
    )

    b = metadata.get("blush") or {}
    parts.append(
        f"Blush: {readable(b.get('color_category'))} at {readable(b.get('placement'))}, "
        f"{readable(b.get('intensity'))} intensity. {(b.get('description') or '').strip()}"
    )

    br = metadata.get("brows") or {}
    parts.append(
        f"Brows: {readable(br.get('shape'))} shape, {readable(br.get('color_match'))}, "
        f"{readable(br.get('technique'))} technique. {(br.get('description') or '').strip()}"
    )

    dense = (metadata.get("dense_caption") or "").strip()
    if dense:
        parts.append(dense)

    return " ".join(p.strip() for p in parts if p.strip())


def resolve_image_path(record: dict, image_field_prefix: str) -> str:
    if record.get("image"):
        return record["image"]
    file_name = record.get("file_name") or f"{record.get('image_id')}.png"
    return os.path.join(image_field_prefix, file_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="把已標註的 metadata jsonl 轉成含各部位細節 caption 的 projector 訓練資料"
    )
    parser.add_argument(
        "--input", default="./makeup_metadata_test100.jsonl",
        help="已標註好的 metadata jsonl(需含 'metadata' 欄位)",
    )
    parser.add_argument("--output", default="./train_data_regional_caption.jsonl", help="輸出的訓練用 jsonl")
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
            metadata = record.get("metadata")
            if not metadata:
                n_skipped += 1
                continue

            caption = build_regional_caption(metadata)
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

    print(f"[Done] 寫入 {n_written} 筆,略過(無 metadata){n_skipped} 筆 → {args.output}")


if __name__ == "__main__":
    main()
