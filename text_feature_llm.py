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
import signal
import pandas as pd
import jieba.posseg as pseg
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from typing import Optional, Dict, Any

# ══════════════════════════════════════════════════════════════════════
# ⚙️  使用者設定區 
# ══════════════════════════════════════════════════════════════════════

DRIVE_ROOT = "/content/drive/MyDrive/ZecZec_Group_Data/時尚_New_ZecZec_Dataset"
OUTPUT_CSV = "/content/drive/MyDrive/ZecZec_Group_Data/時尚_New_ZecZec_Dataset/agent_copywriting_result.csv"

LLM_MAX_CHARS = 3000
API_CALL_DELAY = 1.0

# ── 續傳與防中斷設定 ────────────────────────────────────────────────
# 每處理完一筆就立刻寫入 OUTPUT_CSV（append 模式），重新執行時會自動
# 讀取 OUTPUT_CSV 裡已完成的 project_id 並跳過，達成「續傳」效果。
CHECKPOINT_EVERY_N = 1          # 每處理 N 筆就強制 flush 到磁碟（預設每筆都存）
RESUME_ENABLED     = True       # 是否啟用續傳（讀取既有 CSV，跳過已完成項目）

# ── API 額度限制（429 / quota exceeded）重試設定 ────────────────────
MAX_QUOTA_RETRIES     = 3       # 單一專案遇到額度限制時，最多自動重試幾次
QUOTA_FALLBACK_WAIT   = 20.0    # 若無法從錯誤訊息解析出建議等待秒數，預設等待秒數
QUOTA_WAIT_BUFFER     = 3.0     # 在 Google 建議的等待秒數上，額外加的緩衝秒數
QUOTA_WAIT_PATTERN    = re.compile(r"retry in\s+(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


class QuotaExhaustedError(Exception):
    """代表 Gemini API 額度（429 / rate limit）在重試多次後仍然失敗。"""
    pass

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
        key = api_key or os.environ.get("GEMINI_API_KEY", "AIzaSyApjbcyyxR5vlP4aDdE0ba3Wd3fd5cf3L8")
        if not key:
            raise ValueError("找不到 Gemini API 金鑰。請設定環境變數 GEMINI_API_KEY")
        genai.configure(api_key=key)
        
        self.model = genai.GenerativeModel(
            model_name="gemini-3.5-flash-lite",
            system_instruction=self.SYSTEM_PROMPT,
            generation_config={"response_mime_type": "application/json"}
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        truncated = text[:LLM_MAX_CHARS] if len(text) > LLM_MAX_CHARS else text

        for attempt in range(1, MAX_QUOTA_RETRIES + 1):
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
                break  # 非額度問題，重試也沒用，直接視為失敗

            except Exception as e:
                msg = str(e)
                is_quota_error = (
                    isinstance(e, ResourceExhausted)
                    or "429" in msg
                    or "quota" in msg.lower()
                    or "rate limit" in msg.lower()
                )

                if not is_quota_error:
                    print(f"    ⚠️  Gemini API 錯誤：{e}")
                    break

                match = QUOTA_WAIT_PATTERN.search(msg)
                wait_sec = (float(match.group(1)) + QUOTA_WAIT_BUFFER) if match else QUOTA_FALLBACK_WAIT

                if attempt < MAX_QUOTA_RETRIES:
                    print(f"    ⚠️  API 額度已達上限（第 {attempt}/{MAX_QUOTA_RETRIES} 次），"
                          f"將等待 {wait_sec:.0f} 秒後自動重試...")
                    time.sleep(wait_sec)
                    continue
                else:
                    print(f"    ⛔ 已重試 {MAX_QUOTA_RETRIES} 次仍達 API 額度上限，暫停此專案。")
                    raise QuotaExhaustedError(msg) from e

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
# 續傳與防中斷保存機制
# ══════════════════════════════════════════════════════════════════════

# 完整欄位順序（與最終輸出一致），用來確保每次 append 時欄位對齊
META_COLS  = ["project_id", "category", "subcategory", "status", "merged_file_count"]
RULE_COLS  = [
    "feat_text_story_ratio", "feat_text_spec_ratio_rule", "feat_text_risk_ratio",
    "feat_has_social_link", "feat_punct_intensity", "feat_avg_sentence_len",
    "feat_type_token_ratio",
]
AGENT_COLS = [
    "feat_trust_score", "feat_emotional_ratio", "feat_spec_ratio",
    "extracted_emotional_words", "extracted_spec_words", "agent_advice",
]
ALL_COLS = META_COLS + RULE_COLS + AGENT_COLS


def load_processed_ids(output_csv: str) -> set:
    """讀取既有輸出 CSV，回傳已經處理完成的 project_id 集合（用於續傳）。"""
    if not RESUME_ENABLED or not os.path.exists(output_csv):
        return set()
    try:
        existing_df = pd.read_csv(output_csv, encoding='utf-8-sig')
        if "project_id" in existing_df.columns:
            done = set(existing_df["project_id"].astype(str))
            print(f"🔄 偵測到既有輸出檔，已完成 {len(done)} 筆，將自動跳過。")
            return done
    except (pd.errors.EmptyDataError, Exception) as e:
        print(f"  ⚠️ 讀取既有輸出檔失敗（將視為從頭開始）：{e}")
    return set()


def append_record_to_csv(record: Dict[str, Any], output_csv: str):
    """將單一筆紀錄立刻寫入磁碟（append 模式），避免中斷時遺失已完成的工作。"""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    row_df = pd.DataFrame([record])
    row_df = row_df.reindex(columns=ALL_COLS)  # 確保欄位順序一致
    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    row_df.to_csv(
        output_csv,
        mode='a',
        header=write_header,
        index=False,
        encoding='utf-8-sig',
    )


class GracefulInterrupt:
    """攔截 Ctrl+C / SIGTERM，讓目前這筆處理完、存檔後才乾淨地結束。"""
    def __init__(self):
        self.stop_requested = False
        self._orig_sigint  = signal.getsignal(signal.SIGINT)
        try:
            self._orig_sigterm = signal.getsignal(signal.SIGTERM)
        except (ValueError, AttributeError):
            self._orig_sigterm = None

    def __enter__(self):
        signal.signal(signal.SIGINT, self._handle)
        if self._orig_sigterm is not None:
            signal.signal(signal.SIGTERM, self._handle)
        return self

    def _handle(self, signum, frame):
        if self.stop_requested:
            # 使用者連按第二次 Ctrl+C，代表要立刻強制中止
            print("\n⛔ 偵測到第二次中斷訊號，強制結束。")
            raise KeyboardInterrupt
        print("\n🛑 偵測到中斷訊號，將完成目前這筆診斷並安全存檔後停止...")
        self.stop_requested = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self._orig_sigint)
        if self._orig_sigterm is not None:
            signal.signal(signal.SIGTERM, self._orig_sigterm)
        return False


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
    total_found = len(project_entries)
    print(f"📂 掃描完畢，共發現 {total_found} 個待診斷專案。\n")
    if not project_entries:
        return

    # ── 續傳：跳過既有輸出檔中已完成的 project_id ──────────────────
    processed_ids = load_processed_ids(output_csv)
    pending_entries = [e for e in project_entries if e[4] not in processed_ids]
    skipped = total_found - len(pending_entries)
    if skipped:
        print(f"⏭️  已跳過 {skipped} 個先前已完成的專案，剩餘 {len(pending_entries)} 個待處理。\n")
    if not pending_entries:
        print("✅ 所有專案皆已處理完成，無需再執行。")
        return

    rule_extractor = RuleBasedExtractor()
    copywriting_agent = CopywritingAgent()

    success_count = 0
    fail_count = 0
    interrupted = False
    quota_stopped = False

    with GracefulInterrupt() as guard:
        for idx, (folder_path, category, subcategory, status, project_id) in enumerate(pending_entries, 1):
            if guard.stop_requested:
                interrupted = True
                break

            print(f"[{idx}/{len(pending_entries)}] 正在診斷專案：{project_id}")

            try:
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

                # ── 防中斷保存：每筆處理完立刻寫入磁碟，不等到最後才存 ──
                append_record_to_csv(record, output_csv)
                success_count += 1

            except QuotaExhaustedError:
                # API 額度用盡：這筆「不寫入」CSV，避免被誤標記為已完成，
                # 下次重新執行時續傳機制會自動再次處理這個專案。
                # 且沒有理由繼續打其他專案（大概率一樣會被拒絕），故安全停止整批。
                quota_stopped = True
                print(f"  ⛔ 專案 {project_id} 因 API 額度用盡而暫停，尚未寫入輸出檔，下次執行會自動重試。\n")
                break
            except KeyboardInterrupt:
                # 使用者連按兩次 Ctrl+C，強制中止；目前這筆若未寫入就捨棄
                interrupted = True
                break
            except Exception as e:
                fail_count += 1
                print(f"  ⚠️ 處理專案 {project_id} 時發生未預期錯誤，已略過並繼續：{e}\n")
                continue

    print("\n========================================================")
    if quota_stopped:
        print(f"⛔ 因 API 額度限制安全停止。本次執行成功診斷 {success_count} 筆（失敗 {fail_count} 筆）。")
        print(f"   進度已即時存於：{output_csv}")
        print("   請稍候一段時間（或檢查你的 Gemini 方案額度）後，重新執行本程式即可自動續傳。")
    elif interrupted:
        print(f"🛑 已安全中斷。本次執行成功診斷 {success_count} 筆（失敗 {fail_count} 筆）。")
        print(f"   進度已即時存於：{output_csv}")
        print("   直接重新執行本程式即可自動從中斷處續傳。")
    else:
        print(f"✅ 診斷完成！本次執行成功診斷 {success_count} 筆（失敗 {fail_count} 筆）。")
        print(f"   結果已累積儲存至：{output_csv}")
    print("========================================================")

if __name__ == "__main__":
    process_project_files(DRIVE_ROOT, OUTPUT_CSV)