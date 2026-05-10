"""
圖文比例計算器 v3
─────────────────────────────────────────────
功能：
  1. 掃描根資料夾下所有專案子資料夾
  2. 每個專案讀取 content.txt + log.csv
  3. 字數計算：中文字＋英文詞，排除標點符號
  4. 輸出一份彙整 CSV（每專案一列）
  5. media_per_100_words = 媒體數 ÷ 字數 × 100（每百字有幾筆媒體）

資料夾結構範例：
  projects/
  ├── 專案A/
  │   ├── content.txt
  │   └── log.csv
  ├── 專案B/
  │   ├── content.txt
  │   └── log.csv
  └── ...

輸出欄位：
  project, word_count, image_count, video_count, media_per_100_words
"""

import os
import re
import csv
import argparse


# ── 設定 ───────────────────────────────────────────────────

CONTENT_FILE = "content.txt"   # 每個專案的內文檔名
LOG_FILE     = "resource_log.csv"       # 每個專案的媒體 log 檔名
OUTPUT_FILE  = "result.csv"    # 輸出結果檔名


# ── 1. 中文＋英文詞數計算（排除標點） ─────────────────────

# 中文標點、全形符號
PUNCTUATION_PATTERN = re.compile(
    r'['
    r'\u0000-\u001F'          # 控制字元
    r'\u0020'                 # 空白
    r'！-／：-＠［-｀｛-～'    # 全形標點
    r'\u3000-\u303F'          # CJK 標點
    r'\uFF00-\uFFEF'          # 全形英數（標點部分）
    r'!-/:-@\[-`{-~'         # 半形標點
    r']+',
    re.UNICODE
)

def count_words(txt_path: str) -> int:
    """
    計算文字量：
      - 中文：每個 Unicode CJK 字元算 1 字
      - 英文：以空白切分後的英文詞算 1 詞
      - 標點符號全部排除
    """
    if not os.path.isfile(txt_path):
        return 0

    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 計算中文字數（CJK 統一表意文字區段）
    chinese_chars = re.findall(r'[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]', content)
    chinese_count = len(chinese_chars)

    # 移除中文字後，計算剩餘英文詞數（去除標點後以空白切分）
    no_chinese = re.sub(r'[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]', ' ', content)
    no_punct   = PUNCTUATION_PATTERN.sub(' ', no_chinese)
    english_words = [w for w in no_punct.split() if re.search(r'[A-Za-z]', w)]
    english_count = len(english_words)

    return chinese_count + english_count


# ── 2. 讀取 log.csv，統計圖片與影片 ───────────────────────

def classify_type(type_value: str) -> str:
    t = type_value.strip().lower()
    if t.startswith('image'):
        return 'image'
    elif t == 'video':
        return 'video'
    return 'unknown'


def count_media(csv_path: str) -> dict:
    """讀取 log.csv，回傳圖片數與影片數。"""
    result = {'image_count': 0, 'video_count': 0}

    if not os.path.isfile(csv_path):
        return result

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            return result

        # 欄位名容錯（統一轉小寫去空白）
        fieldnames_lower = {k.strip().lower(): k for k in reader.fieldnames}

        if 'type' not in fieldnames_lower:
            return result

        type_col = fieldnames_lower['type']

        for row in reader:
            kind = classify_type(row[type_col])
            if kind == 'image':
                result['image_count'] += 1
            elif kind == 'video':
                result['video_count'] += 1

    return result


# ── 3. 計算圖文比例 ────────────────────────────────────────

def calc_ratio(word_count: int, media_count: int) -> str:
    """每百字有幾筆媒體（media_count ÷ word_count × 100）。
    字數為 0 時回傳 'N/A'。"""
    if word_count == 0:
        return 'N/A'
    return str(round(media_count / word_count * 100, 2))


# ── 4. 批次掃描所有專案 ────────────────────────────────────

def process_all(root_dir: str) -> list[dict]:
    """
    掃描 root_dir 下每個子資料夾（即每個專案），
    回傳所有專案的統計結果列表。
    """
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"找不到根資料夾：{root_dir}")

    results = []

    project_dirs = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    if not project_dirs:
        print("⚠️  根資料夾內沒有找到任何子資料夾（專案）。")
        return results

    for project_name in project_dirs:
        project_path = os.path.join(root_dir, project_name)
        txt_path     = os.path.join(project_path, CONTENT_FILE)
        csv_path     = os.path.join(project_path, LOG_FILE)

        # 檔案存在性提示
        has_txt = os.path.isfile(txt_path)
        has_csv = os.path.isfile(csv_path)

        if not has_txt:
            print(f"  ⚠️  [{project_name}] 找不到 {CONTENT_FILE}，字數記為 0")
        if not has_csv:
            print(f"  ⚠️  [{project_name}] 找不到 {LOG_FILE}，媒體數記為 0")

        word_count  = count_words(txt_path) if has_txt else 0
        media_info  = count_media(csv_path) if has_csv else {'image_count': 0, 'video_count': 0}
        total_media = media_info['image_count'] + media_info['video_count']

        results.append({
            'project':              project_name,
            'word_count':           word_count,
            'image_count':          media_info['image_count'],
            'video_count':          media_info['video_count'],
            'media_per_100_words':  calc_ratio(word_count, total_media),
        })

        print(f"  ✅  [{project_name}] 字數={word_count}  圖片={media_info['image_count']}  "
              f"影片={media_info['video_count']}  每百字媒體數={results[-1]['media_per_100_words']}")

    return results


# ── 5. 輸出 CSV ────────────────────────────────────────────

FIELDNAMES = ['project', 'word_count', 'image_count', 'video_count', 'media_per_100_words']

def write_csv(results: list[dict], output_path: str):
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n📁 結果已輸出至：{output_path}")


# ── CLI 入口 ───────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='批次計算多專案圖文比例，輸出彙整 CSV'
    )
    parser.add_argument(
        'root',
        help='包含各專案子資料夾的根目錄，例如：projects/'
    )
    parser.add_argument(
        '--output', default=OUTPUT_FILE,
        help=f'輸出 CSV 路徑（預設：{OUTPUT_FILE}）'
    )

    args = parser.parse_args()

    print("=" * 55)
    print("     📊 圖文比例計算器 v3（批次 + CSV 輸出）")
    print("=" * 55)
    print(f"\n🔍 掃描根資料夾：{args.root}\n")

    results = process_all(args.root)

    if results:
        write_csv(results, args.output)
        print(f"   共處理 {len(results)} 個專案")
    else:
        print("沒有可處理的專案。")

    # ── 若偏好直接寫死路徑，取消下方註解並刪除上方 parser 區塊 ──
    # ROOT_DIR    = "projects/"
    # OUTPUT_PATH = "result.csv"
    # results = process_all(ROOT_DIR)
    # if results:
    #     write_csv(results, OUTPUT_PATH)
