# 檔案名稱：extractor_text_colab.py
# 版本：Colab + Google Drive 版
# 資料夾結構：
#   ZecZec_Group_Data/
#   └── ZecZec_Dataset_E&G/
#       └── {類型}/
#           └── {類別}/
#               └── {狀態}/
#                   └── {專案資料夾}/        ← 含 *_content.txt → 視為專案
#                       ├── *_content.txt
#                       ├── *_ocr_content.txt
#                       └── *_resource_log.csv

# ── 安裝必要套件（Colab 環境） ──────────────────────────────────────
import subprocess, sys

def _pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import jieba
except ImportError:
    _pip_install("jieba")
    import jieba

try:
    import jieba.posseg as pseg
except ImportError:
    _pip_install("jieba")
    import jieba.posseg as pseg

# ── 掛載 Google Drive ───────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# ── 標準函式庫 ──────────────────────────────────────────────────────
import os
import re
import glob
import pandas as pd
import jieba.posseg as pseg
from typing import Optional, Dict, Any

# ══════════════════════════════════════════════════════════════════════
# ⚙️  使用者設定區  ←  只需修改這裡
# ══════════════════════════════════════════════════════════════════════

# Google Drive 內的根資料夾路徑（含 ZecZec_Group_Data）
DRIVE_ROOT = "/content/drive/MyDrive/ZecZec_Group_Data/ZecZec_Dataset_E&G"

# 輸出 CSV 存放位置（存回 Drive 方便取用）
OUTPUT_CSV = "/content/drive/MyDrive/ZecZec_Group_Data/feat_out_text.csv"

# 是否遞迴搜尋所有子層資料夾（True = 自動掃全部類別/狀態）
RECURSIVE = True

# ══════════════════════════════════════════════════════════════════════


class DictionaryFreeRiskExtractor:
    """從繁體中文募資頁面文字中萃取結構化風險特徵。"""

    def __init__(self):
        # 預先編譯正則表達式以提升效能
        self.social_pattern = re.compile(
            r'(facebook\.com|fb\.me|fb\.com|instagram\.com|ig\.me'
            r'|line\.me|lin\.ee|linktr\.ee)',
            re.IGNORECASE,
        )
        self.risk_pattern = re.compile(
            r'(風險與挑戰|退換貨規則|注意事項)(.*?)(?=(產品規格|常見問題|$))',
            re.DOTALL,
        )
        self.spec_pattern = re.compile(
            r'(產品規格|規格說明)(.*?)(?=(風險與挑戰|常見問題|$))',
            re.DOTALL,
        )
        self.sentence_split_pattern = re.compile(r'[。！？!\?\n]')
        self.sensational_punct_pattern = re.compile(r'[！？!\?]')

    # ── 私有方法 ──────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """移除所有空白字元，用於計算純字元數。"""
        return re.sub(r'\s', '', text)

    # ── 主要萃取方法 ──────────────────────────────────────────────────

    def extract_all(self, text: str) -> Optional[Dict[str, Any]]:
        """
        從合併文字中萃取所有特徵。
        回傳特徵字典，若文字不合格則回傳 None。
        """
        if not text or not isinstance(text, str) or len(text.strip()) < 5:
            return None

        clean_text   = self._clean_text(text)
        total_chars  = len(clean_text)
        sentences    = [s for s in self.sentence_split_pattern.split(text) if s.strip()]
        num_sentences = max(1, len(sentences))

        # ── 區塊比例計算 ──────────────────────────────────────────────
        risk_match = self.risk_pattern.search(text)
        spec_match = self.spec_pattern.search(text)

        risk_len  = len(self._clean_text(risk_match.group(0))) if risk_match else 0
        spec_len  = len(self._clean_text(spec_match.group(0))) if spec_match else 0
        story_len = max(0, total_chars - risk_len - spec_len)

        # ── 社群連結偵測 ──────────────────────────────────────────────
        has_social_link = 1 if self.social_pattern.search(text) else 0

        # ── 詞性標註 ──────────────────────────────────────────────────
        words_with_flags = list(pseg.cut(text))
        valid_words  = [(w, f) for w, f in words_with_flags if f != 'x']
        total_words  = max(1, len(valid_words))
        unique_words = set(w for w, _ in valid_words)

        pos_counts = {"d": 0, "a": 0, "n": 0, "v": 0, "m": 0, "q": 0, "nt": 0, "nz": 0}
        for _, flag in valid_words:
            for key in pos_counts:
                if flag.startswith(key):
                    pos_counts[key] += 1

        return {
            "feat_text_story_ratio":  round(story_len / total_chars, 4),
            "feat_text_spec_ratio":   round(spec_len  / total_chars, 4),
            "feat_text_risk_ratio":   round(risk_len  / total_chars, 4),
            "feat_has_social_link":   has_social_link,
            "feat_adverb_density":    round(pos_counts["d"] / total_words, 4),
            "feat_adj_nv_ratio":      round(pos_counts["a"] / max(1, pos_counts["n"] + pos_counts["v"]), 4),
            "feat_punct_intensity":   round(len(self.sensational_punct_pattern.findall(text)) / num_sentences, 4),
            "feat_numeral_density":   round((pos_counts["m"] + pos_counts["q"]) / total_words, 4),
            "feat_entity_density":    round((pos_counts["nt"] + pos_counts["nz"]) / total_words, 4),
            "feat_avg_sentence_len":  round(total_chars / num_sentences, 2),
            "feat_type_token_ratio":  round(len(unique_words) / total_words, 4),
        }


# ══════════════════════════════════════════════════════════════════════
# 資料夾掃描邏輯
# ══════════════════════════════════════════════════════════════════════

def find_project_folders(root: str) -> list[tuple[str, str, str, str, str]]:
    """
    遞迴掃描 root，找出所有「含 *_content.txt 的專案資料夾」。

    預期結構：
        root/{類型}/{類別}/{狀態}/{專案資料夾}/

    回傳：list of (project_folder_path, 類型, 類別, 狀態, project_id)
    """
    results = []

    for dirpath, dirnames, filenames in os.walk(root):
        # 判斷該資料夾是否含有 *_content.txt（且不是 *_ocr_content.txt）
        has_content = any(
            f.endswith('_content.txt') and '_ocr_content' not in f
            for f in filenames
        )
        if not has_content:
            continue

        # 解析相對路徑以取得層級資訊
        rel = os.path.relpath(dirpath, root)          # e.g. 預購式專案/教育/成功/ES32_hohoka-01
        parts = rel.replace('\\', '/').split('/')

        # 彈性處理：至少要有專案資料夾名稱
        proj_folder = os.path.basename(dirpath)
        category    = parts[-4] if len(parts) >= 4 else ''
        subcategory = parts[-3] if len(parts) >= 3 else ''
        status      = parts[-2] if len(parts) >= 2 else ''

        # project_id = 資料夾名稱（如 ES32_hohoka-01）
        project_id = proj_folder

        results.append((dirpath, category, subcategory, status, project_id))

    return results


def read_project_text(folder_path: str, project_id: str) -> tuple[str, int]:
    """
    讀取單一專案資料夾內的文字檔並合併。
    優先嘗試：{id}_content.txt → {id}_ocr_content.txt → 任何 *_content.txt
    回傳 (合併文字, 合併檔案數)
    """
    combined_text = ""
    files_merged  = 0

    # 1. 精確名稱比對
    candidates = [
        os.path.join(folder_path, f"{project_id}_content.txt"),
        os.path.join(folder_path, f"{project_id}_ocr_content.txt"),
    ]
    # 2. 萬用字元補漏
    candidates += glob.glob(os.path.join(folder_path, '*_content.txt'))
    candidates += glob.glob(os.path.join(folder_path, '*_ocr_content.txt'))

    seen = set()
    for fp in candidates:
        if fp in seen or not os.path.isfile(fp):
            continue
        seen.add(fp)
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            combined_text += f.read() + "\n\n"
        files_merged += 1

    return combined_text, files_merged


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def process_project_files(drive_root: str, output_csv: str):
    print("\n--- [模組一] 開始萃取文字與 OCR 特徵 ---")

    if not os.path.exists(drive_root):
        print(f"❌ 找不到根資料夾：'{drive_root}'")
        print("   請確認 Google Drive 已掛載，且路徑正確。")
        return

    # ── 掃描所有專案資料夾 ────────────────────────────────────────────
    project_entries = find_project_folders(drive_root)
    print(f"📂 共發現 {len(project_entries)} 個專案資料夾。")

    if not project_entries:
        print("❌ 未找到任何含 *_content.txt 的資料夾，請確認結構是否正確。")
        return

    extractor       = DictionaryFreeRiskExtractor()
    feature_records = []

    for folder_path, category, subcategory, status, project_id in project_entries:
        combined_text, files_merged = read_project_text(folder_path, project_id)

        if not combined_text.strip():
            print(f"  ⚠️  [{project_id}] 無可讀文字，略過。")
            continue

        features = extractor.extract_all(combined_text)
        if features is None:
            print(f"  ⚠️  [{project_id}] 文字過短，略過。")
            continue

        features['project_id']       = project_id
        features['category']         = category       # e.g. 預購式專案
        features['subcategory']      = subcategory    # e.g. 教育
        features['status']           = status         # e.g. 成功
        features['merged_file_count'] = files_merged
        feature_records.append(features)

    # ── 輸出 CSV ──────────────────────────────────────────────────────
    if feature_records:
        df   = pd.DataFrame(feature_records)
        meta = ['project_id', 'status', 'merged_file_count']
        feat = [c for c in df.columns if c not in meta]
        df   = df[meta + feat]

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ 共萃取 {len(df)} 筆專案特徵，已儲存為：{output_csv}")
        print(df.head())
    else:
        print("❌ 未成功萃取任何特徵。")


# ── 執行 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    process_project_files(DRIVE_ROOT, OUTPUT_CSV)
