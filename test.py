# 檔案名稱：extractor_text.py
import os
import glob
import re
import pandas as pd
import jieba
import jieba.posseg as pseg
from typing import Optional, Dict, Any

class DictionaryFreeRiskExtractor:
    def __init__(self):
        # 預先編譯正則表達式以提升效能
        self.social_pattern = re.compile(r'(facebook\.com|fb\.me|fb\.com|instagram\.com|ig\.me|line\.me|lin\.ee|linktr\.ee)', re.IGNORECASE)
        self.risk_pattern = re.compile(r'(風險與挑戰|退換貨規則|注意事項)(.*?)(?=(產品規格|常見問題|$))', re.DOTALL)
        self.spec_pattern = re.compile(r'(產品規格|規格說明)(.*?)(?=(風險與挑戰|常見問題|$))', re.DOTALL)
        self.sentence_split_pattern = re.compile(r'[。！？!\?\n]')
        self.sensational_punct_pattern = re.compile(r'[！？!\?]')

    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s', '', text)

    def extract_all(self, text: str) -> Optional[Dict[str, Any]]:
        if not text or not isinstance(text, str) or len(text.strip()) < 5:
            return None

        clean_text = self._clean_text(text)
        total_chars = len(clean_text)
        sentences = [s for s in self.sentence_split_pattern.split(text) if s.strip()]
        num_sentences = max(1, len(sentences))

        # 區塊比例計算
        risk_match = self.risk_pattern.search(text)
        spec_match = self.spec_pattern.search(text)
        
        risk_len = len(self._clean_text(risk_match.group(0))) if risk_match else 0
        spec_len = len(self._clean_text(spec_match.group(0))) if spec_match else 0
        story_len = max(0, total_chars - risk_len - spec_len)

        # 社群連結與詞性標註
        has_social_link = 1 if self.social_pattern.search(text) else 0
        words_with_flags = list(pseg.cut(text))
        valid_words = [(w, f) for w, f in words_with_flags if f != 'x']
        total_words = max(1, len(valid_words))
        unique_words = set(w for w, f in valid_words)

        pos_counts = {"d": 0, "a": 0, "n": 0, "v": 0, "m": 0, "q": 0, "nt": 0, "nz": 0}
        for word, flag in valid_words:
            for key in pos_counts.keys():
                if flag.startswith(key):
                    pos_counts[key] += 1

        return {
            "feat_text_story_ratio": round(story_len / total_chars, 4),
            "feat_text_spec_ratio": round(spec_len / total_chars, 4),
            "feat_text_risk_ratio": round(risk_len / total_chars, 4),
            "feat_has_social_link": has_social_link,
            "feat_adverb_density": round(pos_counts["d"] / total_words, 4), 
            "feat_adj_nv_ratio": round(pos_counts["a"] / max(1, pos_counts["n"] + pos_counts["v"]), 4), 
            "feat_punct_intensity": round(len(self.sensational_punct_pattern.findall(text)) / num_sentences, 4),
            "feat_numeral_density": round((pos_counts["m"] + pos_counts["q"]) / total_words, 4), 
            "feat_entity_density": round((pos_counts["nt"] + pos_counts["nz"]) / total_words, 4), 
            "feat_avg_sentence_len": round(total_chars / num_sentences, 2), 
            "feat_type_token_ratio": round(len(unique_words) / total_words, 4) 
        }

def process_project_files(folder_path: str, output_csv: str):
    print(f"\n--- [模組一] 開始萃取文字與 OCR 特徵 ---")
    if not os.path.exists(folder_path):
        print(f"❌ 找不到資料夾：'{folder_path}'")
        return

    all_txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    project_ids = set()
    for f in all_txt_files:
        basename = os.path.basename(f)
        pid = re.sub(r'(_content|_ocr)?\.txt$', '', basename, flags=re.IGNORECASE)
        project_ids.add(pid)

    print(f"📂 共發現 {len(project_ids)} 個獨立專案。")

    extractor = DictionaryFreeRiskExtractor()
    feature_records = []

    for pid in project_ids:
        combined_text = ""
        files_merged = 0
        possible_files = [
            os.path.join(folder_path, f"{pid}_content.txt"),
            os.path.join(folder_path, f"{pid}_ocr.txt"),
            os.path.join(folder_path, f"{pid}.txt")
        ]

        for file_path in possible_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    combined_text += f.read() + "\n\n"
                files_merged += 1
        
        if combined_text.strip():
            features = extractor.extract_all(combined_text)
            if features:
                features['project_id'] = pid
                features['merged_file_count'] = files_merged
                feature_records.append(features)

    if feature_records:
        df = pd.DataFrame(feature_records)
        cols = ['project_id', 'merged_file_count'] + [c for c in df.columns if c not in ['project_id', 'merged_file_count']]
        df = df[cols]
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 文字特徵已儲存為：{output_csv}")
    else:
        print("❌ 未成功萃取任何特徵。")

if __name__ == "__main__":
    process_project_files("txt", "feat_out_text.csv")