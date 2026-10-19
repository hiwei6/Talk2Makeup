"""ai_anno.py / ai_anno_tw.py 的分批版本。

背景:
    talk2makeup_0819/phototest/data/beautyface/images 底下共有 3000 張圖,
    但 Gemini API 每日限額只有 500 次請求,一次跑不完,需要分成多天/多批執行。

跟原本兩支腳本的差異:
    1. 用 --lang {en,tw} 切換英文/繁中兩種 schema 與 prompt,取代原本兩支
       幾乎一樣的檔案。
    2. 圖片來源預設改成 data/beautyface/images(3000 張),而不是原本的
       test_images(100 張測試用)。
    3. 新增每日額度控管:額度紀錄在 .gemini_daily_quota.json,依「帳號」
       分開累計(帳號代碼預設從 GEMINI_API_KEY 自動算出,同一把 key 一定
       對到同一個代碼,不用手動輸入;也可用 --account_name 給一個好記的
       名字)。同一個帳號底下,--lang en 與 --lang tw 共用同一組每日額度
       (因為兩者用同一把 key,額度是 API key 層級,不是腳本層級)。
       若手上有多組不同帳號的 API key,可以手動切換 GEMINI_API_KEY 分別
       執行,每次執行都會把該帳號今天用了幾筆記錄下來,並在結束時印出
       目前記錄到的「所有帳號」今日用量,方便下次手動決定要換哪組 key
       繼續跑。額度用完就自動停止,印出還剩多少張沒標註,明天/額度重置後
       重新執行同一指令即可從中斷處(斷點續傳)接著跑。
    4. 沿用 ai_anno_fixed.py 的作法:API 金鑰只吃環境變數 GEMINI_API_KEY,
       不在原始碼中寫死金鑰。

用法範例:
    export GEMINI_API_KEY=xxxx
    python ai_anno_batch.py --lang en       # 跑英文版,今天額度內盡量跑
    python ai_anno_batch.py --lang tw       # 跑繁中版,共用同一份每日額度
    # 隔天額度重置後,重複執行同一指令即可繼續跑剩下的圖片

    # 手上有另一組帳號的 key 時,換一把 key 再跑,會被記成另一個帳號的額度:
    export GEMINI_API_KEY=<另一組帳號的 key>
    python ai_anno_batch.py --lang en --account_name acct_B
"""

import argparse
import fcntl
import glob
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from typing import Literal, Type
from zoneinfo import ZoneInfo

from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm
from google import genai
from google.genai import types
from google.genai.errors import APIError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEMINI_MODEL = "gemini-3.1-flash-lite"

# Gemini API 每日額度是以美國太平洋時間(America/Los_Angeles)午夜重置,
# 不是使用者的本地時間(例如台灣時間下午才會遇到重置),所以額度記錄檔的
# 「今天」日期必須用太平洋時間計算,否則本地時間與實際重置時間對不上,
# 會誤判額度已用完/尚有剩餘。
GEMINI_QUOTA_TZ = ZoneInfo("America/Los_Angeles")

# ==========================================
# 1. Schema 與 Prompt(英文 / 繁體中文 兩組)
# ==========================================
class EyesSchema(BaseModel):
    eyeshadow_tone: Literal["warm_peach", "cool_brown", "smoky", "neutral_nude", "rose_pink", "other"]
    eyeshadow_finish: Literal["matte", "shimmer", "glitter", "satin"]
    eyeliner_style: Literal["none", "tightline", "subtle_wing", "dramatic_cat_eye"]
    aegyo_sal_highlight: bool


class LipsSchema(BaseModel):
    color_category: Literal["coral_peach", "true_red", "nude_pink", "plum_berry", "brick_brown", "other"]
    texture: Literal["matte", "velvet", "satin", "glossy", "metallic"]
    gradient_style: Literal["full_coverage", "inner_gradient", "blurred_lip"]


class FoundationSchemaEN(BaseModel):
    coverage: Literal["sheer", "light", "medium", "full"]
    finish: Literal["matte", "semi_matte", "dewy_glow", "satin"]
    tone_match: Literal["accurate", "lighter_than_skin", "darker_than_skin", "gray_ashy", "other"]
    description: str = Field(..., description="Concrete or abstract observation about the base makeup.")


class BlushSchemaEN(BaseModel):
    color_category: Literal["peach", "coral", "rose_pink", "warm_terracotta", "berry", "none"]
    placement: Literal["apple_cheek", "draped_diagonal", "under_eye_aegyo", "none"]
    intensity: Literal["none", "subtle", "moderate", "strong"]
    description: str = Field(..., description="Concrete or abstract observation about the blush.")


class BrowsSchemaEN(BaseModel):
    shape: Literal["straight", "soft_arch", "high_arch", "angular", "feathered"]
    color_match: Literal["natural_matching", "darker_than_hair", "lighter_than_hair", "tinted_color"]
    technique: Literal["powder", "pencil_strokes", "gel_soap", "microblading_look", "bare"]
    description: str = Field(..., description="Concrete or abstract observation about the brows.")


class MakeupAnnotationEN(BaseModel):
    overall_style: Literal["natural_daily", "k_pop_glam", "western_bold", "retro_classic", "editorial", "no_makeup"]
    intensity_score: int = Field(..., description="Likert Scale from 1 (very light) to 5 (heavy)")
    foundation: FoundationSchemaEN
    eyes: EyesSchema
    lips: LipsSchema
    blush: BlushSchemaEN
    brows: BrowsSchemaEN
    dense_caption: str = Field(
        ...,
        description="Objective, dense description of the overall makeup look, covering both concrete details and abstract style impressions.",
    )


class FoundationSchemaZhTw(BaseModel):
    coverage: Literal["sheer", "light", "medium", "full"]
    finish: Literal["matte", "semi_matte", "dewy_glow", "satin"]
    tone_match: Literal["accurate", "lighter_than_skin", "darker_than_skin", "gray_ashy", "other"]
    description: str = Field(..., description="以繁體中文描述底妝的具體或抽象觀察。")


class BlushSchemaZhTw(BaseModel):
    color_category: Literal["peach", "coral", "rose_pink", "warm_terracotta", "berry", "none"]
    placement: Literal["apple_cheek", "draped_diagonal", "under_eye_aegyo", "none"]
    intensity: Literal["none", "subtle", "moderate", "strong"]
    description: str = Field(..., description="以繁體中文描述腮紅的具體或抽象觀察。")


class BrowsSchemaZhTw(BaseModel):
    shape: Literal["straight", "soft_arch", "high_arch", "angular", "feathered"]
    color_match: Literal["natural_matching", "darker_than_hair", "lighter_than_hair", "tinted_color"]
    technique: Literal["powder", "pencil_strokes", "gel_soap", "microblading_look", "bare"]
    description: str = Field(..., description="以繁體中文描述眉毛的具體或抽象觀察。")


class MakeupAnnotationZhTw(BaseModel):
    overall_style: Literal["natural_daily", "k_pop_glam", "western_bold", "retro_classic", "editorial", "no_makeup"]
    intensity_score: int = Field(..., description="Likert 量表,1(極淡妝)到 5(濃妝)")
    foundation: FoundationSchemaZhTw
    eyes: EyesSchema
    lips: LipsSchema
    blush: BlushSchemaZhTw
    brows: BrowsSchemaZhTw
    dense_caption: str = Field(
        ...,
        description="以繁體中文撰寫的客觀、詳實整體妝容描述,需涵蓋具體細節與抽象風格印象。",
    )


PROMPT_TEXT_EN = """
You are an expert aesthetician and computer vision researcher specializing in facial cosmetics.
Analyze the provided facial image and describe the makeup at two levels:
1. Overall (global) impression: style, intensity, and the general look, in both concrete and abstract terms.
2. Region-level breakdown: foundation, eyes, lips, blush, and brows — describe what is actually observed
   in each region, even if a region shows no visible makeup (state that explicitly).
Extract the fine-grained makeup attributes according to the schema and write an objective, dense textual
caption describing the overall makeup look.
"""

PROMPT_TEXT_TW = """
You are an expert aesthetician and computer vision researcher specializing in facial cosmetics.
Analyze the provided facial image and describe the makeup at two levels:
1. Overall (global) impression: style, intensity, and the general look, in both concrete and abstract terms.
2. Region-level breakdown: foundation, eyes, lips, blush, and brows — describe what is actually observed
   in each region, even if a region shows no visible makeup (state that explicitly).
Extract the fine-grained makeup attributes according to the schema.

IMPORTANT: All free-text fields in the schema (foundation.description, blush.description,
brows.description, and dense_caption) MUST be written in Traditional Chinese (繁體中文,台灣用語),
even though the field names and enum values remain in English. Do not use Simplified Chinese.
Write an objective, dense Traditional Chinese caption describing the overall makeup look.
"""

LANG_CONFIG = {
    "en": {
        "schema": MakeupAnnotationEN,
        "prompt": PROMPT_TEXT_EN,
        "default_output": os.path.join(SCRIPT_DIR, "makeup_metadata_beautyface.jsonl"),
    },
    "tw": {
        "schema": MakeupAnnotationZhTw,
        "prompt": PROMPT_TEXT_TW,
        "default_output": os.path.join(SCRIPT_DIR, "makeup_metadata_beautyface_tw.jsonl"),
    },
}

# ==========================================
# 2. 每日額度控管(依帳號分開記錄,同帳號底下 en / tw 共用同一份額度)
# ==========================================
def derive_account_label(api_key: str) -> str:
    """從 API key 算出一個穩定、不會外洩金鑰本身的帳號代碼。"""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:8]


def _read_quota_state(f) -> dict:
    f.seek(0)
    raw = f.read().strip()
    state = json.loads(raw) if raw else {}

    today = datetime.now(GEMINI_QUOTA_TZ).date().isoformat()
    if state.get("date") != today:
        state = {"date": today, "accounts": {}}
    state.setdefault("accounts", {})
    return state


def reserve_quota_slot(quota_state_file: str, daily_limit: int, account_label: str) -> int:
    """嘗試為指定帳號預約一次今日額度。回傳預約後該帳號今日累積用量;
    若該帳號已達上限則回傳 -1。

    用檔案鎖(flock)保護讀取-修改-寫入,避免同時跑多個行程(例如同帳號的
    --lang en 與 --lang tw,或不同帳號的多個行程)時互相把對方的用量蓋掉。
    """
    with open(quota_state_file, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            state = _read_quota_state(f)
            used = state["accounts"].get(account_label, 0)

            if used >= daily_limit:
                return -1

            used += 1
            state["accounts"][account_label] = used
            f.seek(0)
            f.truncate()
            f.write(json.dumps(state))
            f.flush()
            return used
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_quota_summary(quota_state_file: str) -> dict:
    """讀取目前記錄到的今日各帳號用量(唯讀,不會修改額度)。"""
    if not os.path.exists(quota_state_file):
        return {}
    with open(quota_state_file, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            state = _read_quota_state(f)
            return state["accounts"]
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "[Error] 未設定環境變數 GEMINI_API_KEY。"
            "請先執行 `export GEMINI_API_KEY=<your_key>` 再重跑此腳本"
            "(不要把金鑰寫死在原始碼裡)。"
        )
    return api_key


def load_processed_ids(output_path: str) -> set:
    """讀取已完成標註的 ID,支援斷點續傳"""
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed.add(data.get("image_id"))
                    except json.JSONDecodeError:
                        continue
    return processed


def annotate_image_with_retry(
    client: genai.Client,
    img_path: str,
    schema: Type[BaseModel],
    prompt_text: str,
    max_retries: int = 5,
):
    """呼叫 Gemini API 並包含指數退避重試"""
    img = Image.open(img_path)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt_text, img],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.0,
                ),
            )
            if not response.text:
                # response.text 為 None/空字串,通常代表這張圖被 Gemini 的安全機制
                # 擋下(常見於人臉近拍、看起來像未成年的臉孔觸發 child-safety
                # 分類,這類別無法透過 safety_settings 調整,重試也不會成功)。
                # 印出 prompt_feedback / finish_reason / safety_ratings 方便診斷。
                block_reason = getattr(response.prompt_feedback, "block_reason", None) if response.prompt_feedback else None
                print(f"\n[Blocked] Gemini 未回傳內容(可能被安全機制擋下)。prompt_feedback.block_reason={block_reason}")
                if response.candidates:
                    for i, cand in enumerate(response.candidates):
                        print(f"  candidate[{i}]: finish_reason={cand.finish_reason}, safety_ratings={cand.safety_ratings}")
                else:
                    print("  candidates=[] (無候選結果)")
                break
            return json.loads(response.text)
        except APIError as e:
            wait_time = (2 ** attempt) * 2
            print(f"\n[API Error] {e}. 重試中 ({attempt + 1}/{max_retries}),等待 {wait_time} 秒...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"\n[Error] 未知錯誤: {e}")
            break
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="分批標註 beautyface 資料集(每日 500 額度)")
    parser.add_argument("--lang", choices=["en", "tw"], required=True, help="輸出語言")
    parser.add_argument(
        "--image_dir",
        default=os.path.join(SCRIPT_DIR, "data", "beautyface", "images"),
        help="圖片資料夾(預設: phototest/data/beautyface/images,共 3000 張)",
    )
    parser.add_argument("--output_file", default=None, help="輸出 jsonl 路徑(預設依 --lang 決定)")
    parser.add_argument(
        "--image_field_prefix",
        default=os.path.join("data", "beautyface", "images"),
        help="寫入 record['image'] 時的前綴目錄,須對應訓練腳本的 --data_root",
    )
    parser.add_argument("--daily_limit", type=int, default=500, help="每個帳號的每日 API 額度上限")
    parser.add_argument(
        "--quota_state_file",
        default=os.path.join(SCRIPT_DIR, ".gemini_daily_quota.json"),
        help="每日額度記錄檔,依帳號分開記錄(同帳號底下 en / tw 共用同一份額度)",
    )
    parser.add_argument(
        "--account_name",
        default=None,
        help="這把 GEMINI_API_KEY 對應的帳號代碼,用於分開記錄每日額度。"
        "預設會從 API key 自動算出一個穩定代碼,不用手動填;"
        "手上有多組帳號時可用這個參數給一個好記的名字(例如 acct_A)。",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="最多標註幾張圖(預設不限制,跑到今日額度用完為止)")
    parser.add_argument("--sleep", type=float, default=3.0, help="每張圖之間的間隔秒數(RPM 限制)")
    parser.add_argument("--max_retries", type=int, default=5, help="單張圖片 API 呼叫失敗時的重試次數")
    return parser.parse_args()


def main():
    args = parse_args()
    lang_cfg = LANG_CONFIG[args.lang]
    output_file = args.output_file or lang_cfg["default_output"]

    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    account_label = args.account_name or derive_account_label(api_key)

    processed_ids = load_processed_ids(output_file)
    print(f"[{args.lang}] 已完成標註數量: {len(processed_ids)} 筆")
    print(f"[{args.lang}] 目前使用帳號: {account_label}")

    all_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        all_images.extend(glob.glob(os.path.join(args.image_dir, ext)))
    all_images.sort()

    target_images = all_images[: args.max_samples] if args.max_samples else all_images
    pending_images = [
        p for p in target_images if os.path.splitext(os.path.basename(p))[0] not in processed_ids
    ]

    print(f"[{args.lang}] 總待標註數量: {len(pending_images)} / {len(target_images)}")
    print(f"[{args.lang}] 每個帳號的每日額度上限: {args.daily_limit}(同帳號底下 en/tw 共用同一份額度)")

    quota_exhausted = False
    done_this_run = 0

    for img_path in tqdm(pending_images, desc=f"標註進度[{args.lang}]"):
        used_today = reserve_quota_slot(args.quota_state_file, args.daily_limit, account_label)
        if used_today == -1:
            quota_exhausted = True
            break

        sample_id = os.path.splitext(os.path.basename(img_path))[0]
        file_name = os.path.basename(img_path)

        result_json = annotate_image_with_retry(
            client, img_path, lang_cfg["schema"], lang_cfg["prompt"], max_retries=args.max_retries
        )

        if result_json:
            caption = result_json.get("dense_caption", "").strip()
            if not caption:
                print(f"\n[Warn] {sample_id} 沒有 dense_caption,略過此筆(不寫入)")
            else:
                record = {
                    "image": os.path.join(args.image_field_prefix, file_name),
                    "caption": caption,
                    "image_id": sample_id,
                    "file_name": file_name,
                    "metadata": result_json,
                }
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                done_this_run += 1
        else:
            print(f"\n[Failed] 跳過樣本: {sample_id}")

        time.sleep(args.sleep)

    remaining = len(pending_images) - done_this_run
    print(f"\n[{args.lang}] 本次執行完成 {done_this_run} 筆,結果已儲存至: {output_file}")
    if quota_exhausted:
        print(f"[{args.lang}] 帳號 {account_label} 今日額度({args.daily_limit})已用完,尚有 {remaining} 張未標註。")
        print(f"[{args.lang}] 可換一組帳號的 GEMINI_API_KEY 繼續跑,或等明天額度重置後重新執行同一指令"
              f"(會自動略過已完成的圖片)。")
    elif remaining > 0:
        print(f"[{args.lang}] 尚有 {remaining} 張未標註,可再次執行同一指令繼續。")
    else:
        print(f"[{args.lang}] 全部圖片皆已標註完成!")

    accounts_today = read_quota_summary(args.quota_state_file)
    if accounts_today:
        print(f"\n[今日各帳號用量紀錄]({args.quota_state_file})")
        for label, used in accounts_today.items():
            marker = " <- 目前這次" if label == account_label else ""
            print(f"  {label}: {used}/{args.daily_limit}{marker}")


if __name__ == "__main__":
    main()
