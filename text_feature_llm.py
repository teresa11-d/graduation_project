# 檔案名稱：agent_copywriting_gemini.py
# 版本：Multi-Agent 架構版（文案語意分析 Agent）
# 說明：結合規則式萃取與 Gemini JSON Mode，除了計算比例與提煉字詞，
#       更進一步輸出「信任感評分」與「AI 顧問修改建議」，直接對接 SaaS 前端介面。

# ── 安裝必要套件 ────────────────────────────────────────────────────
import subprocess, sys

def _pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["google-generativeai", "jieba"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        _pip_install(pkg)

# ── 掛載 Google Drive（Colab 環境） ─────────────────────────────────
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ── 標準函式庫 ──────────────────────────────────────────────────────
import os
import re
import glob
import time
import json
import pandas as pd
import jieba.posseg as pseg
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from typing import Optional, Dict, Any

# ══════════════════════════════════════════════════════════════════════
# ⚙️  使用者設定區 
# ══════════════════════════════════════════════════════════════════════

DRIVE_ROOT = "/content/drive/MyDrive/ZecZec_Group_Data/ZecZec_Dataset_E&G"
OUTPUT_CSV = "/content/drive/MyDrive/ZecZec_Group_Data/agent_copywriting_result.csv"

LLM_MAX_CHARS = 3000
API_CALL_DELAY = 1.0

# ══════════════════════════════════════════════════════════════════════
# AI 代理人：文案語意 Agent (Gemini 實作)
# ══════════════════════════════════════════════════════════════════════

class CopywritingAgent:
    """
    募資風險診斷系統 - 文案語意分析 Agent
    負責評估「文案說服力」維度，萃取感性/理性比例、信任詞彙，並給出診斷建議。
    """

    SYSTEM_PROMPT = """你現在是募資風險診斷系統中的核心 AI 顧問：「文案語意分析 Agent」。
你的任務是精準剖析這篇募資文案的說服力、情感渲染力與信任建立機制。

請分析給定文本，並嚴格回傳包含以下 6 個欄位的 JSON：

1. "feat_emotional_ratio": 浮點數 (0.0~1.0)，評估「情感性、品牌初衷、渲染性」詞語佔整體的比例。
2. "feat_spec_ratio": 浮點數 (0.0~1.0)，評估「規格性、技術性、客觀描述」詞語的比例。
3. "feat_trust_score": 整數 (0~100)，評估這篇文案建立贊助者信任感的能力（是否誠懇、有無揭露風險、是否有具體保證）。
4. "emotional_words": 字串陣列，列出你提煉的代表性「感情/渲染字詞」(最多 5 個)。
5. "spec_words": 字串陣列，列出你提煉的代表性「規格/技術字詞」(最多 5 個)。
6. "agent_advice": 字串，以資先行銷顧問的語氣，給予提案團隊一句話的文案修改建議（例如：「規格描述過於生硬，建議在開頭加入更多開發初衷以引發情感共鳴。」或「用語過於誇大，建議補充具體技術數據以提升信任感。」字數限 50 字以內）。

請確保輸出格式為純 JSON。"""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("找不到 Gemini API 金鑰。請設定環境變數 GEMINI_API_KEY")
        genai.configure(api_key=key)
        
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=self.SYSTEM_PROMPT,
            generation_config={"response_mime_type": "application/json"}
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        truncated = text[:LLM_MAX_CHARS] if len(text) > LLM_MAX_CHARS else text

        try:
            response = self.model.generate_content(f"請診斷以下募資文案：\n\n{truncated}")
            raw = response.text.strip()
            result = json.loads(raw)

            emo_words = result.get("emotional_words", [])
            spec_words = result.get("spec_words", [])

            return {
                "feat_emotional_ratio": round(float(result.get("feat_emotional_ratio", -1.0)), 4),
                "feat_spec_ratio":      round(float(result.get("feat_spec_ratio", -1.0)), 4),
                "feat_trust_score":     int(result.get("feat_trust_score", 0)),
                "extracted_emotional_words": ", ".join(emo_words) if emo_words else "無",
                "extracted_spec_words": ", ".join(spec_words) if spec_words else "無",
                "agent_advice":         result.get("agent_advice", "無法生成建議。")
            }

        except json.JSONDecodeError as e:
            print(f"    ⚠️  JSON 解析失敗：{e}")
        except ResourceExhausted:
            print("    ⚠️  API Rate Limit，等待 15 秒後繼續...")
            time.sleep(15)
        except Exception as e:
            print(f"    ⚠️  Gemini API 錯誤：{e}")

        return {
            "feat_emotional_ratio": -1.0, 
            "feat_spec_ratio": -1.0,
            "feat_trust_score": 0,
            "extracted_emotional_words": "錯誤",
            "extracted_spec_words": "錯誤",
            "agent_advice": "系統發生錯誤，無法診斷。"
        }

# ══════════════════════════════════════════════════════════════════════
# 規則式特徵萃取器（保留原有邏輯）
# ══════════════════════════════════════════════════════════════════════

class RuleBasedExtractor:
    def __init__(self):
        self.social_pattern = re.compile(r'(facebook\.com|fb\.me|fb\.com|instagram\.com|ig\.me|line\.me|lin\.ee|linktr\.ee)', re.IGNORECASE)
        self.risk_pattern = re.compile(r'(風險與挑戰|退換貨規則|注意事項)(.*?)(?=(產品規格|常見問題|$))', re.DOTALL)
        self.spec_pattern = re.compile(r'(產品規格|規格說明)(.*?)(?=(風險與挑戰|常見問題|$))', re.DOTALL)
        self.sentence_split_pattern = re.compile(r'[。！？!\?\n]')
        self.sensational_punct_pattern = re.compile(r'[！？!\?]')

    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s', '', text)

    def extract(self, text: str) -> Optional[Dict[str, Any]]:
        if not text or not isinstance(text, str) or len(text.strip()) < 5: return None

        clean_text    = self._clean_text(text)
        total_chars   = len(clean_text)
        sentences     = [s for s in self.sentence_split_pattern.split(text) if s.strip()]
        num_sentences = max(1, len(sentences))

        risk_match = self.risk_pattern.search(text)
        spec_match = self.spec_pattern.search(text)
        risk_len   = len(self._clean_text(risk_match.group(0))) if risk_match else 0
        spec_len   = len(self._clean_text(spec_match.group(0))) if spec_match else 0
        story_len  = max(0, total_chars - risk_len - spec_len)

        has_social_link = 1 if self.social_pattern.search(text) else 0

        words_with_flags = list(pseg.cut(text))
        valid_words      = [(w, f) for w, f in words_with_flags if f != 'x']
        total_words      = max(1, len(valid_words))
        unique_words     = set(w for w, _ in valid_words)

        return {
            "feat_text_story_ratio": round(story_len / total_chars, 4),
            "feat_text_spec_ratio_rule":  round(spec_len  / total_chars, 4),
            "feat_text_risk_ratio":  round(risk_len  / total_chars, 4),
            "feat_has_social_link":  has_social_link,
            "feat_punct_intensity":  round(len(self.sensational_punct_pattern.findall(text)) / num_sentences, 4),
            "feat_avg_sentence_len": round(total_chars / num_sentences, 2),
            "feat_type_token_ratio": round(len(unique_words) / total_words, 4),
        }

# ══════════════════════════════════════════════════════════════════════
# 資料夾掃描與讀取邏輯
# ══════════════════════════════════════════════════════════════════════

def find_project_folders(root: str) -> list:
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(f.endswith('_content.txt') and '_ocr_content' not in f for f in filenames):
            rel   = os.path.relpath(dirpath, root)
            parts = rel.replace('\\', '/').split('/')
            proj_folder = os.path.basename(dirpath)
            category    = parts[-4] if len(parts) >= 4 else ''
            subcategory = parts[-3] if len(parts) >= 3 else ''
            status      = parts[-2] if len(parts) >= 2 else ''
            results.append((dirpath, category, subcategory, status, proj_folder))
    return results

def read_project_text(folder_path: str, project_id: str):
    combined_text = ""
    files_merged  = 0
    candidates  = [os.path.join(folder_path, f"{project_id}_content.txt"), os.path.join(folder_path, f"{project_id}_ocr_content.txt")]
    candidates += glob.glob(os.path.join(folder_path, '*_content.txt'))
    candidates += glob.glob(os.path.join(folder_path, '*_ocr_content.txt'))

    seen = set()
    for fp in candidates:
        if fp in seen or not os.path.isfile(fp): continue
        seen.add(fp)
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            combined_text += f.read() + "\n\n"
        files_merged += 1
    return combined_text, files_merged

# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def process_project_files(drive_root: str, output_csv: str):
    print("\n========================================================")
    print(" 🧠 啟動 Multi-Agent 系統：[文案語意分析 Agent]")
    print("========================================================")
    
    if not os.path.exists(drive_root):
        print(f"❌ 找不到根資料夾：'{drive_root}'")
        return

    project_entries = find_project_folders(drive_root)
    print(f"📂 掃描完畢，共發現 {len(project_entries)} 個待診斷專案。\n")
    if not project_entries: return

    rule_extractor = RuleBasedExtractor()
    copywriting_agent = CopywritingAgent() 
    feature_records = []

    for idx, (folder_path, category, subcategory, status, project_id) in enumerate(project_entries, 1):
        print(f"[{idx}/{len(project_entries)}] 正在診斷專案：{project_id}")

        combined_text, files_merged = read_project_text(folder_path, project_id)

        if not combined_text.strip():
            print("  ⚠️ 無可讀文字，略過。\n")
            continue

        rule_feats = rule_extractor.extract(combined_text)
        if rule_feats is None:
            print("  ⚠️ 文字過短，略過。\n")
            continue

        time.sleep(API_CALL_DELAY)
        agent_feats = copywriting_agent.analyze(combined_text)

        # 終端機顯示排版 (模擬 Agent 回報)
        if agent_feats["feat_emotional_ratio"] >= 0:
            print(f"  📊 診斷分數 | 信任度: {agent_feats['feat_trust_score']}/100 | 情感比: {agent_feats['feat_emotional_ratio']:.2f} | 規格比: {agent_feats['feat_spec_ratio']:.2f}")
            print(f"  🔑 關鍵字彙 | {agent_feats['extracted_emotional_words']} / {agent_feats['extracted_spec_words']}")
            print(f"  💬 顧問建議 | {agent_feats['agent_advice']}\n")
        else:
            print("  ⚠️ Agent 診斷失敗\n")

        record = {
            "project_id":         project_id,
            "category":           category,
            "subcategory":        subcategory,
            "status":             status,
            "merged_file_count":  files_merged,
            **rule_feats,
            **agent_feats,
        }
        feature_records.append(record)

    if feature_records:
        df = pd.DataFrame(feature_records)
        
        # 重新排序欄位，將 Agent 建議放在最後面方便閱讀
        meta_cols = ["project_id", "category", "subcategory", "status", "merged_file_count"]
        agent_cols = ["feat_trust_score", "feat_emotional_ratio", "feat_spec_ratio", "extracted_emotional_words", "extracted_spec_words", "agent_advice"]
        rule_cols = [c for c in df.columns if c not in meta_cols + agent_cols]

        df = df[meta_cols + rule_cols + agent_cols]

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')

        print(f"✅ 診斷完成！共產出 {len(df)} 份文案語意報告，已儲存至：{output_csv}")
    else:
        print("❌ 未成功產出任何報告。")

if __name__ == "__main__":
    process_project_files(DRIVE_ROOT, OUTPUT_CSV)