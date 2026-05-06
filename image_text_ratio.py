"""
圖文比例計算器
計算 txt 內文字數（或段落數）與資料夾內圖片數量的比例
"""

import os
import sys
import argparse

# 支援的圖片副檔名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.svg'}


def count_text(txt_path: str, mode: str = 'chars') -> dict:
    """
    讀取 txt 檔案，計算文字數量。

    mode:
        'chars'      — 計算非空白字元數（中英文皆適用）
        'words'      — 計算以空白分隔的詞數（英文友好）
        'paragraphs' — 計算非空行數（段落數）
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"找不到文字檔：{txt_path}")

    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.splitlines()
    non_empty_lines = [l for l in lines if l.strip()]

    char_count      = len(content.replace(' ', '').replace('\n', '').replace('\t', ''))
    word_count      = len(content.split())
    paragraph_count = len(non_empty_lines)

    return {
        'chars':      char_count,
        'words':      word_count,
        'paragraphs': paragraph_count,
        'selected':   {'chars': char_count, 'words': word_count,
                       'paragraphs': paragraph_count}[mode],
        'mode':       mode,
    }


def count_images(folder_path: str) -> dict:
    """
    掃描資料夾，計算圖片數量（不含子資料夾）。
    若要含子資料夾，請改用 recursive=True。
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"找不到資料夾：{folder_path}")

    all_files   = os.listdir(folder_path)
    image_files = [
        f for f in all_files
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    return {
        'count':  len(image_files),
        'files':  image_files,
        'folder': folder_path,
    }


def calculate_ratio(text_count: int, image_count: int) -> dict:
    """計算圖文比例，回傳多種表示方式。"""
    if image_count == 0:
        return {
            'text_per_image': None,
            'images_per_text_unit': None,
            'ratio_str': f"文字：圖片 = {text_count} : 0（無圖片）",
            'percentage': None,
        }
    if text_count == 0:
        return {
            'text_per_image': 0,
            'images_per_text_unit': None,
            'ratio_str': f"文字：圖片 = 0 : {image_count}（無文字）",
            'percentage': None,
        }

    # 化簡比例
    from math import gcd
    g = gcd(text_count, image_count)
    simplified = f"{text_count // g} : {image_count // g}"

    return {
        'text_per_image':        round(text_count / image_count, 2),
        'images_per_text_unit':  round(image_count / text_count, 4),
        'ratio_str':             f"文字：圖片 = {simplified}（原始 {text_count} : {image_count}）",
        'image_ratio_percent':   round(image_count / (text_count + image_count) * 100, 2),
        'text_ratio_percent':    round(text_count  / (text_count + image_count) * 100, 2),
    }


def report(txt_path: str, folder_path: str, mode: str = 'chars'):
    """主函式：顯示完整報告。"""
    print("=" * 55)
    print("         📄 圖文比例計算器")
    print("=" * 55)

    # --- 文字統計 ---
    text_info = count_text(txt_path, mode)
    print(f"\n📝 文字檔：{txt_path}")
    print(f"   字元數（非空白）：{text_info['chars']:,}")
    print(f"   詞數            ：{text_info['words']:,}")
    print(f"   段落數（非空行）：{text_info['paragraphs']:,}")
    print(f"   ➜  採用計算模式：【{mode}】→ {text_info['selected']:,}")

    # --- 圖片統計 ---
    img_info = count_images(folder_path)
    print(f"\n🖼  圖片資料夾：{folder_path}")
    print(f"   圖片總數：{img_info['count']}")
    if img_info['files']:
        preview = img_info['files'][:5]
        print(f"   檔案預覽：{', '.join(preview)}"
              + (" ..." if len(img_info['files']) > 5 else ""))

    # --- 比例計算 ---
    ratio = calculate_ratio(text_info['selected'], img_info['count'])
    print(f"\n📊 圖文比例結果")
    print(f"   {ratio['ratio_str']}")

    if ratio['text_per_image'] is not None:
        print(f"   每張圖片對應文字量 ：{ratio['text_per_image']:,} （{mode}）")
        print(f"   文字佔比：{ratio['text_ratio_percent']}%　圖片佔比：{ratio['image_ratio_percent']}%")

    print("\n" + "=" * 55)
    return {
        'text': text_info,
        'images': img_info,
        'ratio': ratio,
    }


# ── CLI 入口 ──────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='計算 txt 文字檔與圖片資料夾的圖文比例'
    )
    parser.add_argument('txt',    help='txt 文字檔路徑，例如：article.txt')
    parser.add_argument('folder', help='圖片資料夾路徑，例如：images/')
    parser.add_argument(
        '--mode', choices=['chars', 'words', 'paragraphs'],
        default='chars',
        help='文字計算模式：chars（字元數）| words（詞數）| paragraphs（段落數），預設 chars'
    )

    args = parser.parse_args()
    report(args.txt, args.folder, args.mode)
