# 檔案名稱：zeczec_market_price.py
# 版本：v7.0（整合最終版：核心價格 + 市場行情，含完整容錯與額度管理）
#
# ── 這支檔案做什麼 ──────────────────────────────────────────────────
# 讀取募資專案的 plans.txt（方案定價）與 content.txt（募資文案），算出：
#   1. plans_core_price：這個專案「最合理可比價」的核心售價
#   2. market_price_min / market_price_max：對應的台灣市場行情區間
#   3. price_deviation_ratio / is_price_competitive：核心價跟市場行情的偏離程度
#
# ── 核心售價怎麼算（PlansParser._pick_core） ──────────────────────────
#   優先序：0.明確標示「單入/單本/單件」的方案 → 1.排除法推論的標準單入
#          → 2.標準多入(拆算單價) → 3.早鳥單入 → 4.兜底(納入組合包)
#   每一層都用「同標題方案分組→組內取最高價（排除早鳥異常低價）→
#   跨組取中位數」而不是直接取全體最高價，避免被離群的高價方案拉高。
#   「＋/+」組合包、加購方案一律不列入候選，除非所有方案都是組合包才兜底納入。
#
# ── 市場行情怎麼查（MarketPriceAgent，每個商品最多 2 次 Gemini 呼叫）──
#   Call 1：判斷「含規格特徵的關鍵字」＋核心售價（若 plans.txt 已提供則沿用）
#   Call 2：開啟 Google Search Grounding，在同一次生成內自行嘗試
#           精準搜尋 → 放寬品類 → AI知識推斷 三層策略，要求盡量跨 ≥2 個
#           台灣主要電商平台（蝦皮/PChome/momo/Yahoo/樂天等）比對，
#           規格類似即可（不需完全相同），但保證「絕對要有價格」——
#           就算 Gemini 完全無回應，也會用核心售價 ±30% 做本地保底估算
#           （標記為 low 品質，不會跟真實查到的行情混淆）。
#   同一份文案 / 同一個關鍵字重複執行不會重打 API（雜湊快取），
#   失敗結果不會被快取（下次會自動重試），啟動時也會自動清除舊版殘留的
#   失敗快取紀錄。
#
# ── 額度管理 ───────────────────────────────────────────────────────
#   模型用 gemini-flash-latest 別名（自動指向當前最新穩定 Flash 模型，
#   不受單一版本下架影響；可用環境變數 GEMINI_MODEL_OVERRIDE 指定固定版本）。
#   429 限流錯誤會解析 Google 回傳的 quotaId/retryDelay，精準判斷是每分鐘
#   還是每日額度，並用建議的等待秒數重試；判定為每日額度用完時，批次處理
#   會立即停止（不會對剩餘專案繼續徒勞重試），已完成的結果不會遺失。
#
# ── 使用方式 ────────────────────────────────────────────────────────
#   單一專案測試：
#     from zeczec_market_price import run_test
#     run_test("PF1_plans.txt", "PF1_content.txt")
#
#   批次處理整個資料集（走訪含 *_content.txt 的資料夾，輸出成 CSV）：
#     from zeczec_market_price import batch_run
#     batch_run(root_dir="ZecZec_Dataset_E&G", output_csv="market_price_results.csv")
#
#   API Key 設定（三選一，優先序：環境變數 → Colab Secrets → 互動輸入）：
#     os.environ["GEMINI_API_KEY"] = "AIza..."          # 本機/Jupyter
#     Colab 左側 🔑 面板新增 GEMINI_API_KEY               # Colab
#     不設定的話，執行時會跳出 getpass 輸入框請你貼上         # 兩者皆可

import subprocess, sys

def _pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg, imp in [("google-genai", "google.genai"), ("numpy", "numpy"),
                 ("requests", "requests")]:
    try:
        __import__(imp)
    except ImportError:
        _pip_install(pkg)

import os, re, json, time, hashlib
import numpy as np
from google import genai
from google.genai import types
from typing import Optional, Dict, Any, Tuple, List

# ══════════════════════════════════════════════════════════════════════
# ⚙️  設定區
# ══════════════════════════════════════════════════════════════════════

PLANS_FILE   = "PF2_plans.txt"
CONTENT_FILE = "PF2_content.txt"

CACHE_FILE      = "market_price_cache.json"
LLM_MAX_CHARS   = 3000
# 用「latest」別名而非釘死版本號，Google 之後汰換模型世代時會自動指向當時的
# 最新穩定 Flash 模型，不需要每次舊版本停用（例如 gemini-2.5-flash 已不開放
# 新帳號使用）就要手動改程式碼。如果你想固定用某個特定版本（例如評估用途
# 需要結果可重現），可以改成明確版本號，或設定環境變數 GEMINI_MODEL_OVERRIDE。
GEMINI_MODEL    = os.environ.get("GEMINI_MODEL_OVERRIDE", "gemini-flash-latest")


def _get_api_key() -> str:
    """依序嘗試：環境變數 → Colab Secrets → 互動輸入，避免把金鑰寫死在程式碼裡。"""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    # Colab Secrets（左側 🔑 面板設定的密鑰）
    try:
        from google.colab import userdata  # 只有在 Colab 環境才存在
        try:
            key = (userdata.get("GEMINI_API_KEY") or "").strip()
            if key:
                os.environ["GEMINI_API_KEY"] = key
                return key
        except Exception:
            pass  # 使用者未設定該 Secret，往下走互動輸入
    except ImportError:
        pass  # 非 Colab 環境

    # 互動輸入（Colab / Jupyter / 終端機皆可用）
    try:
        import getpass
        key = getpass.getpass("請貼上你的 GEMINI_API_KEY（輸入內容不會顯示）：").strip()
        if key:
            os.environ["GEMINI_API_KEY"] = key
    except Exception:
        pass

    return key

# ══════════════════════════════════════════════════════════════════════


def _read_file(path: str) -> str:
    if not os.path.isfile(path):
        print(f"  ⚠️  找不到檔案：{path}")
        return ""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def _parse_json(raw: str) -> Optional[dict]:
    # 清除 markdown fence
    cleaned = re.sub(r'```(?:json)?|```', '', raw).strip()
    # 1. 正常 parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 2. JSON 被截斷時：用 regex 抓出 keywords 陣列中所有已完整出現的 "xxx" 字串
    m = re.search(r'"keywords"\s*:\s*\[', cleaned)
    if m:
        arr_fragment = cleaned[m.end():]
        items = re.findall(r'"([^"]+)"', arr_fragment)
        if items:
            return {"keywords": items}
    # 3. 對其他欄位（keyword, core_price 等）：暴力補全尾巴後再 parse
    for suffix in ('"]}', '"', '"]}', ']"}', '"}}', '}', ']}'):
        try:
            return json.loads(cleaned + suffix)
        except json.JSONDecodeError:
            pass
    return None

class DailyQuotaExceededError(Exception):
    """當偵測到 Gemini API 每日額度（RPD）用完時拋出，讓上層（例如批次處理）
    可以判斷要整批停下來，而不是對每個剩餘專案繼續徒勞地重試。"""
    pass


def _call_gemini(client: genai.Client, prompt: str,
                 system: str = "", max_tokens: int = 256,
                 json_mode: bool = True,
                 use_search: bool = False,
                 disable_thinking: bool = True,
                 _retry_count: int = 0) -> Optional[str]:
    # 遇到「無法判斷是分鐘還是每日限制」的模糊 429 錯誤時，用遞增等待時間重試幾次：
    # RPM/TPM（每分鐘）限制一定會在 1 分鐘內的滾動視窗內恢復，如果累積等待超過
    # 這個時間仍然失敗，就幾乎可以確定是 RPD（每日額度）用完，改成直接停止，
    # 而不是每次都只等 20 秒就再送出下一次請求繼續徒勞撞牆。
    _AMBIGUOUS_RETRY_WAITS = [20, 40, 70]   # 累積 130 秒，遠超過 RPM 滾動視窗的 1 分鐘

    cfg_kwargs: Dict[str, Any] = {"max_output_tokens": max_tokens}
    # ⚠️ Google Search Grounding 不支援 response_mime_type，必須分開處理
    if json_mode and not use_search:
        cfg_kwargs["response_mime_type"] = "application/json"
    # ⚠️ gemini-2.5-flash 預設會啟用「思考」，思考過程也會消耗 max_output_tokens 額度，
    # 若 max_tokens 設定較小，容易導致額度全部被思考用光、實際輸出變成空字串
    # （這種情況不會丟例外，只會安靜地失敗）。純分類/JSON 輸出的呼叫不需要思考，
    # 預設關閉以確保 token 額度留給真正的輸出。
    if disable_thinking:
        try:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else None
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                tools=tools,
                **cfg_kwargs,
            )
        )
        # Grounding 回應可能在多個 candidates/parts 中，需逐一拼接
        if resp.text:
            return resp.text.strip()
        # 備援：手動拼接所有 text parts
        parts = []
        try:
            for cand in (resp.candidates or []):
                for part in (cand.content.parts or []):
                    if hasattr(part, "text") and part.text:
                        parts.append(part.text)
        except Exception:
            pass
        if parts:
            return "".join(parts).strip()

        # 完全沒有文字內容：印出診斷原因，避免安靜失敗
        finish_reason = None
        try:
            finish_reason = resp.candidates[0].finish_reason
        except Exception:
            pass
        if finish_reason and str(finish_reason) not in ("STOP", "1"):
            print(f"  ⚠️  Gemini 回應為空（finish_reason={finish_reason}）。"
                  f"若為 MAX_TOKENS，請調高 max_tokens；若為 SAFETY，"
                  f"代表內容被安全機制擋下。")
        else:
            print("  ⚠️  Gemini 回應為空（無明確原因，可嘗試調高 max_tokens 重試）。")
        return None
    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "API key not valid" in err:
            print("  ❌  API Key 無效，請確認金鑰是否正確、是否為 AI Studio 發行的 "
                  "AIza... 開頭金鑰，或是否已被撤銷。前往 "
                  "https://aistudio.google.com/apikey 重新確認/產生。")
            return None
        elif "PERMISSION_DENIED" in err:
            print(f"  ❌  權限不足（PERMISSION_DENIED），請確認該金鑰對應的專案"
                  f"是否已啟用 Generative Language API：{e}")
            return None
        elif "NOT_FOUND" in err or ("404" in err and "model" in err.lower()):
            print(f"  ❌  模型「{GEMINI_MODEL}」已不開放使用（可能是新帳號限制或"
                  f"已下架）。目前程式已改用 gemini-flash-latest 自動指向最新穩定版，"
                  f"若仍出現此錯誤，代表 Google 可能又更換了模型世代，"
                  f"請至 https://ai.google.dev/gemini-api/docs/models 確認目前"
                  f"可用的模型名稱後更新 GEMINI_MODEL：{e}")
            return None
        elif "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            err_lower = err.lower()
            # 優先解析 Google 回傳的結構化資訊：
            #   quotaId 明確標示 PerMinute / PerDay（比關鍵字比對可靠）
            #   retryDelay 是 Google 自己算好的建議等待秒數（比亂猜的固定秒數精準）
            quota_id_match  = re.search(r"'quotaId':\s*'([^']+)'", err)
            retry_delay_match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", err)
            quota_id = quota_id_match.group(1) if quota_id_match else ""
            retry_delay = float(retry_delay_match.group(1)) + 2 if retry_delay_match else None  # +2秒緩衝

            is_per_day = ("PerDay" in quota_id or "per day" in err_lower
                          or "perday" in err_lower.replace(" ", ""))
            is_per_minute = ("PerMinute" in quota_id or "per minute" in err_lower
                             or "rpm" in err_lower or "tpm" in err_lower)

            if is_per_day:
                print("  🛑 已達【每日額度上限】（RPD），短暫等待沒有用，"
                      "要等到太平洋時間午夜 00:00（約台灣時間下午 3~4 點）才會重置。\n"
                      "     可到 https://aistudio.google.com/usage 查看即時用量與重置時間，"
                      "或考慮升級付費層級提高額度。")
                raise DailyQuotaExceededError(str(e))
            elif is_per_minute:
                wait = retry_delay if retry_delay else 20
                print(f"  ⚠️  已達【每分鐘額度上限】（{quota_id or 'RPM/TPM'}），"
                      f"等待 {wait:.0f} 秒後重試"
                      f"{'（採用 Google 建議的 retryDelay）' if retry_delay else ''}...")
                time.sleep(wait)
                if _retry_count < 3:
                    return _call_gemini(client, prompt, system, max_tokens, json_mode,
                                        use_search, disable_thinking, _retry_count + 1)
                print("  🛑 已重試多次仍持續被限流，這已超過 RPM 限制正常恢復的時間，"
                      "判定為每日額度用完，停止本次批次。")
                raise DailyQuotaExceededError(str(e))
            else:
                # 訊息本身沒有明確標示是分鐘還是每日限制，用遞增等待時間重試，
                # 累積等待時間遠超過 RPM 的 1 分鐘滾動視窗；若重試完仍失敗，
                # 判定為每日額度用完，直接停止而不是繼續無意義地重試。
                if _retry_count < len(_AMBIGUOUS_RETRY_WAITS):
                    wait = retry_delay if retry_delay else _AMBIGUOUS_RETRY_WAITS[_retry_count]
                    print(f"  ⚠️  Rate Limit（無法判斷是分鐘還是每日限制），"
                          f"第 {_retry_count + 1}/{len(_AMBIGUOUS_RETRY_WAITS)} 次重試，"
                          f"等待 {wait:.0f} 秒...\n"
                          f"     可到 https://ai.dev/rate-limit 查看即時用量")
                    time.sleep(wait)
                    return _call_gemini(client, prompt, system, max_tokens, json_mode,
                                        use_search, disable_thinking, _retry_count + 1)
                print(f"  🛑 已重試 {len(_AMBIGUOUS_RETRY_WAITS)} 次、累積等待超過 1 分鐘"
                      f"仍持續被限流——RPM/TPM 限制正常應該早就恢復了，判定為每日額度"
                      f"（RPD）用完，停止本次批次。\n"
                      f"     詳細訊息：{e}\n"
                      f"     可到 https://ai.dev/rate-limit 或 https://aistudio.google.com/usage"
                      f" 確認實際用量。")
                raise DailyQuotaExceededError(str(e))
        elif "INVALID_ARGUMENT" in err or "mime" in err.lower():
            print(f"  ⚠️  API 參數錯誤（可能 mime_type 衝突）：{e}")
            return None
        else:
            print(f"  ⚠️  Gemini 錯誤：{e}")
            return None


# ══════════════════════════════════════════════════════════════════════
# 方案定價解析器（純 Regex）
# ══════════════════════════════════════════════════════════════════════

class PlansParser:
    _MULTI_KEYWORDS  = re.compile(r'雙入|三入|四入|五入|六入|多入|團購|加購|組合|兩入|2入|3入|4入|5入')
    _PLAN_SEP        = re.compile(r'▼\s*【方案\s*\d+】\s*▼')
    _PRICE_LINE      = re.compile(r'NT\$?\s*([0-9,]+)')
    _EARLY_BIRD      = re.compile(r'早鳥|限時|預購特惠|超早鳥')
    _ADDON           = re.compile(r'加購|加碼|贈品|配件')
    # 組合包：標題中出現「＋」或「+」代表把多個不同品項綁在一起賣
    # （例如「實體書＋書籤＋電子版」），售價自然比單一品項高，
    # 不應該被當成核心／正常定價的比較對象。
    _BUNDLE          = re.compile(r'＋|\+')
    # 明確標示「單入」的方案（單本/單入/單件/1入/一份…），這是判斷核心售價的
    # 最優先依據——比起「排除早鳥/多入/組合後剩下的」這種間接推論更可靠，
    # 因為它是直接比對文字上明確寫的「這是單一件商品」。
    _SINGLE          = re.compile(r'單入|單本|單件|單一件|單份|1入|一入|一份', re.IGNORECASE)
    # 從標題推算數量（雙入=2，三入=3…）
    _QTY_MAP         = {'雙入': 2, '兩入': 2, '2入': 2, '三入': 3, '3入': 3,
                        '四入': 4, '4入': 4, '五入': 5, '5入': 5, '六入': 6, '6入': 6}

    def _detect_qty(self, title: str) -> int:
        for kw, qty in self._QTY_MAP.items():
            if kw in title:
                return qty
        return 1

    def parse(self, plans_text: str) -> Dict[str, Any]:
        if not plans_text.strip():
            return {"plans_count": 0, "plans_core_price": 0, "plans_min_price": 0,
                    "plans_max_price": 0, "plans_has_early_bird": 0, "parsed_plans": []}

        blocks = [b.strip() for b in self._PLAN_SEP.split(plans_text) if b.strip()]
        parsed = []
        for block in blocks:
            lines    = [l.strip() for l in block.splitlines() if l.strip()]
            title    = lines[0] if lines else ""
            is_multi  = bool(self._MULTI_KEYWORDS.search(title))
            is_early  = bool(self._EARLY_BIRD.search(title))
            is_addon  = bool(self._ADDON.search(title))
            is_bundle = bool(self._BUNDLE.search(title))
            is_single = bool(self._SINGLE.search(title))
            qty      = self._detect_qty(title)
            prices   = self._PRICE_LINE.findall(block)
            sale_p   = int(prices[0].replace(',', '')) if len(prices) >= 1 else 0
            orig_p   = int(prices[1].replace(',', '')) if len(prices) >= 2 else sale_p
            unit_p   = round(sale_p / qty) if qty > 1 and sale_p > 0 else sale_p
            if sale_p > 0:
                parsed.append({
                    "title":          title,
                    "is_multi":       is_multi,
                    "is_early_bird":  is_early,
                    "is_addon":       is_addon,
                    "is_bundle":      is_bundle,
                    "is_single":      is_single,
                    "qty":            qty,
                    "sale_price":     sale_p,
                    "unit_price":     unit_p,
                    "original_price": orig_p,
                })

        if not parsed:
            return {"plans_count": 0, "plans_core_price": 0, "plans_min_price": 0,
                    "plans_max_price": 0, "plans_has_early_bird": 0, "parsed_plans": []}

        # ── 核心售價選取邏輯 ────────────────────────────────────────
        # 優先順序（tier 0~3 篩出候選集合後，皆用「同標題分組→取組內最高價
        # →跨組取中位數」的方式選出核心價，而不是直接取全體最高價，
        # 避免選到離群的高價方案，改為選出「最合理可比價」的那個）：
        # 0. 標題明確標示「單入/單本/單件」的方案（排除早鳥/加購/組合包）
        #    → 這是最可靠的判斷依據，因為是直接比對文字上明確寫的
        #      「這是單一件商品」，而非用排除法間接推論。
        # 1. 若沒有明確標示單入的方案 → 退回排除法：非早鳥、非多入、非加購、
        #    非組合包 的方案
        # 2. 若全部都是多入 → 拆算單價後的方案（排除早鳥/加購/組合包）
        # 3. 若全部都是早鳥 → 早鳥單入方案（排除加購/組合包）
        # 4. 兜底：所有方案 unit_price 中位數（此時才會納入組合包，避免完全沒有結果）

        def _group_median(plans: List[dict]) -> Optional[int]:
            """
            同標題方案視為「同一商品的不同時期定價」，組內取最高價
            （早鳥視為異常低價，排除其影響）；不同標題（不同商品/規格）
            之間再取中位數，代表最合理、最適合拿去跟市場比價的那個值，
            而不是直接取全體最高價（可能選到離群的高價變體）。
            """
            groups: Dict[str, List[int]] = {}
            for p in plans:
                groups.setdefault(p["title"], []).append(p["unit_price"])
            if not groups:
                return None
            group_max = sorted(max(v) for v in groups.values())
            n = len(group_max)
            # 中位數：奇數取正中間；偶數取中間偏低者（避免偏向高估）
            return group_max[(n - 1) // 2]

        def _pick_core(plans: List[dict]) -> int:
            # tier-0：明確標示「單入」的方案（排除早鳥/加購/組合包）
            tier0 = [p for p in plans if p["is_single"] and not p["is_early_bird"]
                     and not p["is_addon"] and not p["is_bundle"]]
            v = _group_median(tier0)
            if v is not None:
                return v
            # tier-1: 標準單入（排除早鳥/多入/加購/組合包，用排除法推論）
            tier1 = [p for p in plans if not p["is_multi"] and not p["is_early_bird"]
                     and not p["is_addon"] and not p["is_bundle"]]
            v = _group_median(tier1)
            if v is not None:
                return v
            # tier-2: 標準多入（拆算單價，排除加購/組合包）
            tier2 = [p for p in plans if not p["is_early_bird"] and not p["is_addon"]
                     and not p["is_bundle"]]
            v = _group_median(tier2)
            if v is not None:
                return v
            # tier-3: 早鳥單入（排除加購/組合包）
            tier3 = [p for p in plans if not p["is_multi"] and not p["is_addon"]
                     and not p["is_bundle"]]
            v = _group_median(tier3)
            if v is not None:
                return v
            # tier-4: 兜底，此時才納入組合包，取全體中位數
            all_units = sorted(p["unit_price"] for p in plans)
            return all_units[len(all_units) // 2]

        core_price = _pick_core(parsed)
        all_p      = [p["sale_price"] for p in parsed]
        single_p   = [p["sale_price"] for p in parsed if not p["is_multi"]]

        return {
            "plans_count":         len(parsed),
            "plans_core_price":    core_price,           # ← 主要比較用，最合理可比價
            "plans_min_price":     min(single_p) if single_p else min(all_p),
            "plans_max_price":     max(all_p),
            "plans_has_early_bird": int(any(p["is_early_bird"] for p in parsed)),
            "parsed_plans":        parsed,
        }


# ══════════════════════════════════════════════════════════════════════
# 市場價格 Agent
# ══════════════════════════════════════════════════════════════════════

class MarketPriceAgent:
    """
    負責「市場行情」查詢，每個商品正常情況下最多 2 次 Gemini 呼叫：
      - Call 1：_analyze_keyword_and_price → 判斷含規格特徵的關鍵字＋核心售價
      - Call 2：_fetch_market_range        → 單次 Grounding 搜尋，內建
                精準搜尋→放寬品類→AI知識推斷 三層策略，由 Gemini 在
                同一次生成內自行遞進嘗試，不需分開呼叫三次 API

    規格比對原則：類似即可，不要求完全相同（材質/容量/尺寸/版本可以有差異），
    只排除完全不同類別的商品。

    保證一定有市場價：Grounding 找不到 → 補打一次 AI 知識推斷 → 若兩次都
    完全無回應（例如網路/API 異常），才會用核心售價 ±30% 做本地保底估算，
    絕不會回傳「無法取得」。
    """

    _ANALYZE_SYSTEM = """你是專業電商商品分析師，專門分析台灣群眾募資文案。
任務：從募資文案（與可能提供的方案售價資訊）中，直接判斷出：
1. keyword：一個消費者在蝦皮/PChome/momo搜尋時會用、且帶有「規格特徵」的關鍵字組合
   （例如材質、尺寸、容量、片數、版本等），目的是讓之後查詢到的市場價格，
   對應的是「同規格同類型」的商品，而不是泛用同類詞。
   - 不含品牌名、創作者名、行銷詞
   - 8~16字中文詞組（詞＋規格特徵），例如「304不鏽鋼 480ml 保溫瓶」而非只寫「保溫瓶」
2. core_price：若文案中提供標準單入方案售價則直接採用；若沒有，從文案自行判斷合理的
   單入/標準版新台幣整數售價，找不到則填0

規則：關鍵字與售價都必須直接對應文案中的商品，不可憑空創造。
回傳純 JSON（不含 markdown fence 或任何說明文字）：
{"keyword": "（含規格特徵的關鍵字）", "core_price": 數字}"""

    _PRICE_SEARCH_SYSTEM = """你是台灣電商價格分析師，任務是找出「指定規格商品」在台灣市場的真實售價區間，
且必須盡可能跨多個電商平台比對，確保這是市場的普遍行情，而非單一商家的個別報價。

🏬【請盡量涵蓋以下台灣主要電商/通路，不要只查一家就結束】
蝦皮購物、PChome 24h購物、momo購物網、Yahoo奇摩購物中心、樂天市場、
森森購物、生活市集、friDay購物、博客來（書籍類）、東森購物、名店街等。

🚨【搜尋策略 — 請在同一次回覆內依序自行嘗試，不需使用者再次提問】🚨
Google Search Grounding 不支援 site: 指令，請直接用自然語言搜尋：
1. 第一步：用完整關鍵字（含規格特徵）分別嘗試「蝦皮 價格」「momo 價格」
   「PChome 價格」等不同平台的自然語言搜尋，盡量找到至少 2 個不同平台的售價，
   目標是找到「類型類似、用途相同」的商品售價（例如同類型的保溫瓶、同類型的
   筆記本），不需要規格完全一模一樣（材質/容量/尺寸/版本可以有些微差異），
   只要是消費者會拿來比較的同類商品即可，這樣才找得到足夠的市場數據。
2. 若第一步找到的同平台/同類商品不夠多，放寬成上層品類關鍵字＋「購買」再搜尋，
   一樣盡量嘗試多個平台，目標是湊到至少 3 筆有效售價。
3. 若做完以上兩步仍然完全找不到任何售價，才憑你對台灣電商市場的知識，
   針對這個商品類型給出合理估算區間，並在 sources_note 標明「AI知識推斷」。

⚠️【最重要的規則：無論如何都必須給出價格區間，絕對不可以回傳「找不到」、
0、或空值。就算完全沒有搜尋結果，也一定要根據商品類型給出合理估算。】

排除：嘖嘖/flyingV 等募資平台售價、二手商品、海外代購、蝦皮個人賣家的非量產二手品、
以及完全不同類別的商品（例如關鍵字是保溫瓶卻查到保溫袋，或是書籍卻查到文具），
但同類型商品的規格差異（容量、尺寸、材質、版本）不需要完全相同，類似即可。
必須給出具體數字，不可回傳0。

data_quality 判斷標準：
- "high"：有 2 個以上不同平台、且商品類型相符的售價可交叉比對
- "medium"：只找到 1 個平台的相符售價，或跨平台但筆數很少
- "low"：僅為放寬品類後的估算，或是 AI 知識推斷

回傳純 JSON（不含 markdown fence 或多餘說明文字）：
{
  "searched_keyword": "實際用來得出結果的關鍵字",
  "search_tier": "exact"/"broadened"/"ai_estimate",
  "platforms": ["實際查到售價的平台名稱列表，例如 蝦皮購物, momo購物網"],
  "prices_found": [所有有效售價整數，單位新台幣，此欄位不可為空陣列],
  "price_min": Q25或最低合理價,
  "price_max": Q75或最高合理價,
  "data_quality": "high"/"medium"/"low",
  "sources_note": "資料來源、平台數量與搜尋策略說明"
}"""

    _AI_ESTIMATE_SYSTEM = """你是台灣電商市場價格專家，熟悉蝦皮、PChome、momo、博客來等平台的商品行情。
根據商品關鍵字（含規格特徵），憑你的訓練知識給出台灣市場的合理售價區間。

規則：
- 絕對必須給出具體數字，不可回傳 0、不可留空、不可說找不到
- 以台灣消費市場正常零售價為準（非特賣、非批發）
- 針對關鍵字所屬的商品類型估算即可，不需要規格完全對應，類似類型的市售
  商品價格即可作為估算基礎
- 偏離幅度允許 ±40%，寧可範圍寬一點也要給數字；就算商品很小眾冷門，
  也要用最相近的同類商品類型推估一個合理區間

回傳純 JSON（禁止 markdown fence 或任何說明文字）：
{"price_min": 整數, "price_max": 整數, "note": "估算依據一句話說明", "confidence": "high"/"medium"/"low"}"""

    def __init__(self, client: genai.Client, content_text: str):
        self.client = client
        self._cache: Dict[str, Any] = self._load_cache()

    # ── Call 1：一次判斷「含規格特徵的關鍵字」＋核心售價 ──────────────
    def _analyze_keyword_and_price(self, content_text: str,
                                   known_core_price: int = 0) -> Tuple[str, int]:
        # ── 內容雜湊快取：同一份文案（＋同樣的已知核心售價）重複執行時，
        #    不再重打 Gemini，適合 Colab 反覆測試/除錯的情境 ──────────
        cache_key = "kw:" + hashlib.md5(
            (content_text[:LLM_MAX_CHARS] + f"|{known_core_price}").encode("utf-8")
        ).hexdigest()
        if cache_key in self._cache:
            c = self._cache[cache_key]
            print(f"  [快取] 關鍵字判斷沿用先前結果：{c['keyword']}（NT${c['price']:,}）")
            return c["keyword"], c["price"]

        price_hint = (f"\n【已知標準單入方案售價】：NT${known_core_price:,}"
                      f"（keyword 需能代表這個價位對應的規格商品）"
                      if known_core_price > 0 else "")
        prompt = (
            f"【募資文案】：\n{content_text[:LLM_MAX_CHARS]}{price_hint}"
        )
        raw = _call_gemini(self.client, prompt,
                           system=self._ANALYZE_SYSTEM,
                           max_tokens=400, json_mode=True,
                           disable_thinking=True)
        keyword, price = "未分類商品", known_core_price
        if raw:
            r = _parse_json(raw)
            if r:
                keyword = str(r.get("keyword") or keyword).strip() or keyword
                if known_core_price <= 0:
                    try:
                        price = int(float(str(r.get("core_price", 0)).replace(',', '')))
                    except (ValueError, TypeError):
                        price = 0
        if keyword == "未分類商品":
            m = re.search(r'(?:單入|標準版|早鳥).*?NT\$?\s*([0-9,]+)', content_text)
            if m and known_core_price <= 0:
                price = int(m.group(1).replace(',', ''))

        if keyword != "未分類商品":
            self._cache[cache_key] = {"keyword": keyword, "price": price}
            self._save_cache()
        else:
            print("  ⚠️  關鍵字判斷失敗（未分類商品），本次結果不寫入快取，下次會重新嘗試。")
        return keyword, price

    def _get_keyword(self, text: str, candidates: List[str]) -> str:
        # 保留舊接口相容性（目前流程已不需要，由 _analyze_keyword_and_price 取代）
        return candidates[0] if candidates else "未分類商品"

    def _get_keyword_and_price(self, text: str,
                               candidates: List[str]) -> Tuple[str, int]:
        return self._analyze_keyword_and_price(text)

    def _filter_candidates(self, content_text: str, top_k: int = 3) -> List[str]:
        # 已移除 Embedding 篩選步驟以減少 API 呼叫次數，保留空清單相容舊呼叫點
        return []

    # ── 從原始文字中解析價格數字（純本地運算，不耗用 API 額度）───────
    @staticmethod
    def _parse_prices_from_raw(raw: str) -> Tuple[int, int, str, List[int]]:
        prices: List[int] = []
        json_match = re.search(r'\{[\s\S]*?\}', raw)
        if json_match:
            r = _parse_json(json_match.group(0)) or {}
            val = r.get("prices_found", [])
            if isinstance(val, list):
                for v in val:
                    try:
                        n = int(str(v).replace(',', ''))
                        if 50 < n < 500_000:
                            prices.append(n)
                    except Exception:
                        pass
            for field in ("price_min", "price_max"):
                try:
                    n = int(str(r.get(field, 0) or 0).replace(',', ''))
                    if 50 < n < 500_000:
                        prices.append(n)
                except Exception:
                    pass

        for m in re.finditer(
            r'(?:NT\$?|TWD|新台幣|售價[：: ]?|定價[：: ]?|原價[：: ]?|特價[：: ]?|約[：: ]?)'
            r'\s*([0-9][0-9,]{1,6})',
            raw
        ):
            try:
                n = int(m.group(1).replace(',', ''))
                if 50 < n < 500_000:
                    prices.append(n)
            except Exception:
                pass

        if len(prices) < 2:
            candidates = []
            for m in re.finditer(r'\b([1-9][0-9]{2,5})\b', raw):
                n = int(m.group(1))
                if 50 < n < 500_000:
                    candidates.append(n)
            if candidates:
                buckets: Dict[int, List[int]] = {}
                for n in candidates:
                    mag = 10 ** (len(str(n)) - 1)
                    buckets.setdefault(mag, []).append(n)
                best_mag = max(buckets, key=lambda k: len(buckets[k]))
                prices.extend(buckets[best_mag])

        prices = sorted(set(prices))
        if len(prices) >= 2:
            p_min = int(np.percentile(prices, 25))
            p_max = int(np.percentile(prices, 75))
            if p_min == p_max:
                p_min = int(p_min * 0.8)
                p_max = int(p_max * 1.2)
            quality = "high" if len(prices) >= 3 else "medium"
            return p_min, p_max, quality, prices
        elif len(prices) == 1:
            v = prices[0]
            return int(v * 0.8), int(v * 1.2), "medium", prices
        return 0, 0, "low", []

    # ── Call 2：單次 Grounding 呼叫，三層 fallback 策略寫入同一個 prompt ──
    def _fetch_market_range(self, keyword: str,
                            fallback_keywords: Optional[List[str]] = None,
                            core_price_hint: int = 0
                            ) -> Tuple[int, int, str, str, List[str]]:

        cache_key = f"v8:{keyword}"
        if cache_key in self._cache:
            c = self._cache[cache_key]
            plats = c.get("platforms", [])
            plat_str = f"（來源：{'、'.join(plats)}）" if plats else ""
            print(f"  [快取] {c.get('searched_keyword', keyword)} → {c['range_str']} [{c['quality']}]{plat_str}")
            return c["min"], c["max"], c["range_str"], c["quality"], plats

        market_min = market_max = 0
        data_quality = "low"
        searched_kw = keyword
        source_label = ""
        platforms: List[str] = []

        print(f"  🌐 [單次呼叫] Grounding 搜尋（內建 精準→放寬→AI推斷 三層策略，"
              f"要求跨多平台比對）：關鍵字【{keyword}】")
        grounding_prompt = (
            f"商品關鍵字（含規格特徵）：{keyword}\n"
            f"請依系統指示的策略，盡量跨蝦皮、PChome、momo、Yahoo購物中心、"
            f"樂天市場等至少 2 個不同平台，找出類型類似（不需規格完全相同，"
            f"容量/材質/版本可以有差異，只要是同類商品即可）的真實售價，"
            f"列出至少3筆售價、標明各筆售價來自哪個平台，並計算最低/最高合理售價。"
            f"無論如何都必須給出價格，不可回傳找不到。"
        )
        raw_g = _call_gemini(
            self.client, grounding_prompt,
            system=self._PRICE_SEARCH_SYSTEM,
            max_tokens=1536, json_mode=False, use_search=True,
            disable_thinking=False  # 搜尋策略需要推理判斷是否要放寬品類，保留思考
        )
        search_tier = "unknown"
        if raw_g:
            print(f"  📥 Grounding 原始回傳（前300字）：{raw_g[:300]!r}")
            r = _parse_json(re.search(r'\{[\s\S]*\}', raw_g).group(0)) if re.search(r'\{[\s\S]*\}', raw_g) else None
            if r:
                searched_kw = str(r.get("searched_keyword", keyword))
                search_tier = str(r.get("search_tier", "unknown"))
                plats = r.get("platforms", [])
                if isinstance(plats, list):
                    platforms = sorted({str(x).strip() for x in plats if str(x).strip()})
            market_min, market_max, data_quality, prices_found = self._parse_prices_from_raw(raw_g)
            # 若模型明確標示查到 2 個以上不同平台的售價，資料品質提升為 high
            # （跨平台交叉比對過，比單一來源可信）
            if len(platforms) >= 2 and market_max > 0:
                data_quality = "high"
            if market_max > 0:
                tier_label = {"exact": "精準搜尋", "broadened": "放寬品類搜尋",
                              "ai_estimate": "AI知識推斷"}.get(search_tier, "Grounding搜尋")
                source_label = tier_label
                plat_str = f"，來源平台：{'、'.join(platforms)}" if platforms else "，⚠️ 未標示來源平台"
                print(f"  ✅ 取得行情：NT${market_min:,}~NT${market_max:,}"
                      f"（{len(prices_found)}筆，策略：{tier_label}{plat_str}）")

        # ── 若 Grounding 沒給出價格（無論是完全無回應，還是回應了但沒有可用數字），
        #    都補打一次純知識推斷，確保一定能拿到價格 ──
        if market_max == 0:
            print("  🤖 [備援呼叫] Grounding 沒有取得有效價格，改用 Gemini 純知識推斷...")
            ai_prompt = (
                f"商品關鍵字（含規格特徵）：{keyword}\n\n"
                f"請根據你對台灣電商市場（蝦皮/PChome/momo）的知識，"
                f"針對這個商品類型（類似規格即可，不需要完全對應）估算合理售價區間"
                f"（新台幣）。\n絕對必須給出具體數字，不可為 0、不可說不確定。"
            )
            raw_ai = _call_gemini(
                self.client, ai_prompt,
                system=self._AI_ESTIMATE_SYSTEM,
                max_tokens=400, json_mode=True, use_search=False,
                disable_thinking=True
            )
            if raw_ai:
                print(f"  📥 AI推斷回傳：{raw_ai[:200]!r}")
                r_ai = _parse_json(raw_ai) or {}
                try:
                    p_min = int(str(r_ai.get("price_min", 0) or 0).replace(',', ''))
                    p_max = int(str(r_ai.get("price_max", 0) or 0).replace(',', ''))
                except Exception:
                    p_min = p_max = 0
                if p_max == 0:
                    p_min, p_max, _, _ = self._parse_prices_from_raw(raw_ai)
                if p_max > 0 and p_min >= 0:
                    if p_min == 0:
                        p_min = int(p_max * 0.6)
                    market_min, market_max = p_min, p_max
                    data_quality = r_ai.get("confidence", "low")
                    source_label = f"AI知識推斷（{r_ai.get('note', '估算')}）"
                    platforms = []  # AI 知識推斷沒有實際來源平台
                    print(f"  🤖 AI推斷行情：NT${market_min:,}~NT${market_max:,}（{data_quality}）")

        # ── 保底機制：就算 Grounding 跟 AI 推斷兩次呼叫都完全失敗（例如網路/
        #    API 異常），也絕對不留空——用已知的核心售價往外估算一個區間，
        #    確保「一定要有價格」這個要求無論如何都會被滿足。這種情況品質
        #    標為 low，且不會被誤認為真正查到的市場行情。──
        if market_max == 0:
            if core_price_hint > 0:
                market_min = int(core_price_hint * 0.7)
                market_max = int(core_price_hint * 1.3)
                data_quality = "low"
                source_label = "本地保底估算（Gemini 兩次呼叫皆無回應，以核心售價±30%推算）"
                print(f"  🛟 [保底估算] Gemini 完全無回應，改用核心售價推算："
                      f"NT${market_min:,}~NT${market_max:,}")
            else:
                # 連核心售價都不知道時的最後手段：給一個通用的保守估算區間
                market_min, market_max = 300, 3000
                data_quality = "low"
                source_label = "本地保底估算（無任何可用資訊，為通用區間，僅供參考）"
                print("  🛟 [保底估算] 無核心售價可參考，使用通用保守區間。")

        range_str = f"NT${market_min:,} ~ NT${market_max:,}"
        tag = f"【{source_label}】" if source_label else ""
        print(f"  📈 市場行情 {range_str} {tag}")
        self._cache[cache_key] = {
            "min": market_min, "max": market_max,
            "range_str": range_str, "quality": data_quality,
            "searched_keyword": searched_kw,
            "platforms": platforms,
        }
        self._save_cache()

        return market_min, market_max, range_str, data_quality, platforms

    def analyze(self, content_text: str,
                plans_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        known_core_price = 0
        price_source = "llm"
        if plans_info and plans_info.get("plans_core_price", 0) > 0:
            known_core_price = plans_info["plans_core_price"]
            price_source = "plans"

        # Call 1：一次判斷「含規格特徵的關鍵字」＋（必要時）核心售價
        keyword, core_price = self._analyze_keyword_and_price(content_text, known_core_price)
        if known_core_price > 0:
            core_price = known_core_price
        else:
            price_source = "llm" if core_price > 0 else "regex"

        # Call 2（單次）：Grounding 搜尋市場行情（同類型商品即可，盡量跨多平台）
        market_min, market_max, range_str, quality, platforms = self._fetch_market_range(
            keyword, core_price_hint=core_price
        )

        is_competitive = 0
        deviation = None
        if market_max > 0:
            is_competitive = 1 if market_min <= core_price <= market_max else 0
            median = (market_min + market_max) / 2
            deviation = round((core_price - median) / median, 4) if median else 0.0

        return {
            "product_keyword":       keyword,
            "project_core_price":    core_price,
            "price_source":          price_source,
            "market_price_min":      market_min,
            "market_price_max":      market_max,
            "market_range_str":      range_str,
            "market_data_quality":   quality,
            "market_platforms":      "、".join(platforms) if platforms else "",
            "market_platform_count": len(platforms),
            "price_deviation_ratio": deviation,
            "is_price_competitive":  is_competitive,
        }

    def _load_cache(self) -> dict:
        if not os.path.exists(CACHE_FILE):
            return {}
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                raw_cache = json.load(f)
        except Exception:
            return {}

        # 自動清除舊版本可能殘留的「失敗結果」快取（例如金鑰失效那次寫入的紀錄），
        # 避免程式一直讀到過期的失敗結果而不重試。
        cleaned = {}
        removed = 0
        for k, v in raw_cache.items():
            is_bad = (
                (isinstance(v, dict) and v.get("keyword") == "未分類商品") or
                (isinstance(v, dict) and str(v.get("range_str", "")).startswith("無法取得"))
            )
            if is_bad:
                removed += 1
                continue
            cleaned[k] = v
        if removed:
            print(f"  🧹 已自動清除 {removed} 筆過期的失敗快取紀錄，將重新嘗試。")
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cleaned, f, ensure_ascii=False)
        return cleaned

    def _save_cache(self):
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f, ensure_ascii=False)




# ══════════════════════════════════════════════════════════════════════
# 報告列印
# ══════════════════════════════════════════════════════════════════════

def _print_plans_table(parsed_plans: List[dict], core_price: int = 0):
    print("\n  ┌──────────────────────────────────────────────────────────────┐")
    print("  │                      方案定價明細                             │")
    print("  ├──────────────────────────┬───────┬───────┬───────┬──────────┤")
    print("  │ 方案名稱                 │  售價 │  單價 │  折扣 │ 標籤     │")
    print("  ├──────────────────────────┼───────┼───────┼───────┼──────────┤")
    for p in parsed_plans:
        title    = p["title"][:22].ljust(22)
        sale     = f"NT${p['sale_price']:,}".rjust(7)
        unit     = f"NT${p['unit_price']:,}".rjust(7) if p.get("qty", 1) > 1 else "      -"
        discount = (f"{round(p['sale_price']/p['original_price']*10,1)}折"
                    if p.get("original_price", 0) > p["sale_price"] else "  - ")
        tags = []
        if p.get("is_single"):     tags.append("單入")
        if p.get("is_early_bird"): tags.append("🐦早鳥")
        if p.get("is_multi"):      tags.append(f"×{p.get('qty',2)}")
        if p.get("is_addon"):      tags.append("加購")
        if p.get("is_bundle"):     tags.append("🎁組合")
        is_core = (core_price > 0 and p["unit_price"] == core_price
                   and not p.get("is_early_bird") and not p.get("is_addon")
                   and not p.get("is_bundle"))
        if is_core:                tags.append("★核心")
        tag_str = " ".join(tags).ljust(8)
        print(f"  │ {title} │ {sale} │ {unit} │ {discount} │ {tag_str} │")
    print("  └──────────────────────────┴───────┴───────┴───────┴──────────┘")


def _print_market_report(price_result: Dict[str, Any]):
    kw    = price_result["product_keyword"]
    cp    = price_result["project_core_price"]
    src   = price_result["price_source"]
    mmin  = price_result["market_price_min"]
    mmax  = price_result["market_price_max"]
    rstr  = price_result["market_range_str"]
    qual  = price_result["market_data_quality"]
    dev   = price_result["price_deviation_ratio"]
    comp  = price_result["is_price_competitive"]
    plats = price_result.get("market_platforms", "")

    verdict = "✅ 價格具競爭力" if comp else "⚠️  價格偏離市場行情"
    dev_str = f"{dev:+.1%}" if dev is not None else "N/A"

    bar_len = 32
    if mmax > mmin:
        pos        = max(0.0, min(1.0, (cp - mmin) / (mmax - mmin)))
        marker_pos = int(pos * bar_len)
        bar        = "─" * marker_pos + "▲" + "─" * (bar_len - marker_pos)
        bar_line   = f"  ║  NT${mmin:<7,}[{bar}]NT${mmax:<7,}║"
    else:
        bar_line   = f"  ║  市場行情：{rstr:<46}║"

    quality_label = {
        "high":   "高（≥2個平台交叉比對）",
        "medium": "中（單一平台或筆數較少）",
        "low":    "低（估算值）",
    }.get(qual, qual)
    plats_line = f"  ║  來源平台     {plats:<38}║" if plats else \
                 "  ║  來源平台     （未標示，可能為單一來源或估算值）      ║"

    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║              市場價格競爭力診斷報告                  ║
  ╠══════════════════════════════════════════════════════╣
  ║  產品關鍵字   {kw:<38}║
  ║  核心售價     NT${cp:<6,}  （來源：{src:<8}）           ║
  ║  市場行情     {rstr:<40}║
  ║  資料品質     {quality_label:<38}║
{plats_line}
  ║                                                      ║
{bar_line}
  ║                                                      ║
  ║  偏離幅度     {dev_str:<8}  {verdict:<27}║
  ╚══════════════════════════════════════════════════════╝""")


# ══════════════════════════════════════════════════════════════════════
# 主測試流程
# ══════════════════════════════════════════════════════════════════════

def run_test(plans_file: str = PLANS_FILE, content_file: str = CONTENT_FILE):
    print("\n" + "=" * 62)
    print(" 🧪 市場價格比對測試 v5.1（Gemini 呼叫最小化 + Colab 相容版）")
    print(f"    model   : {GEMINI_MODEL}")
    print(f"    plans   : {plans_file}")
    print(f"    content : {content_file}")
    print(f"    搜尋方式 : Google Search Grounding（無需 SerpAPI）")
    print("=" * 62)

    plans_text   = _read_file(plans_file)
    content_text = _read_file(content_file)
    if not content_text.strip():
        print("❌ content 檔案無法讀取，終止。")
        print("   （在 Colab 中可用左側檔案面板上傳，或執行："
              "from google.colab import files; files.upload()）")
        return

    api_key = _get_api_key()
    if not api_key:
        print("❌ 找不到 GEMINI_API_KEY。請用以下任一方式提供：\n"
              "   1) 本機：os.environ['GEMINI_API_KEY'] = 'AIza...'\n"
              "   2) Colab：左側 🔑 Secrets 面板新增 GEMINI_API_KEY 並開啟筆記本存取權\n"
              "   3) 直接依提示互動輸入金鑰")
        return
    client = genai.Client(api_key=api_key)

    # ── 步驟 1：解析 plans.txt ─────────────────────────────────────
    print("\n【步驟 1】解析 plans.txt 方案定價...")
    parser     = PlansParser()
    plans_info = parser.parse(plans_text)
    if plans_info["plans_count"] > 0:
        print(f"  ✅ 找到 {plans_info['plans_count']} 個方案")
        print(f"     核心售價（標準單入）  ：NT${plans_info['plans_core_price']:,}  ← 市場比對用")
        print(f"     最低活動價（含早鳥）  ：NT${plans_info['plans_min_price']:,}")
        print(f"     最高方案售價          ：NT${plans_info['plans_max_price']:,}")
        print(f"     含早鳥方案：{'是' if plans_info['plans_has_early_bird'] else '否'}")
        _print_plans_table(plans_info["parsed_plans"], plans_info["plans_core_price"])
    else:
        print("  ⚠️  未找到 plans.txt 或格式不符，將由 LLM 推斷價格。")

    # ── 步驟 2：規格化關鍵字判斷（Call 1）+ 單次市場行情搜尋（Call 2）──
    print("\n【步驟 2】Gemini 判斷含規格特徵的關鍵字 → 單次 Google Search 搜尋行情...")
    market_agent = MarketPriceAgent(client, content_text)
    price_result = market_agent.analyze(content_text, plans_info=plans_info)
    _print_market_report(price_result)

    # ── 步驟 3：摘要 ───────────────────────────────────────────────
    dev = price_result["price_deviation_ratio"]
    print("\n【診斷摘要】")
    print(f"  最終關鍵字  ：{price_result['product_keyword']}")
    print(f"  核心售價    ：NT${price_result['project_core_price']:,}  (來源：{price_result['price_source']})")
    print(f"  市場行情    ：{price_result['market_range_str']}")
    print(f"  資料品質    ：{price_result['market_data_quality']}")
    print(f"  來源平台    ：{price_result.get('market_platforms') or '（未標示）'}")
    print(f"  偏離幅度    ：{f'{dev:+.2%}' if dev is not None else 'N/A'}")
    print(f"  競爭力判定  ：{'✅ 具競爭力' if price_result['is_price_competitive'] else '⚠️  偏離市場'}\n")



# ══════════════════════════════════════════════════════════════════════
# 批次處理：走訪整個資料集資料夾，逐一分析每個專案並輸出成 CSV
# （原本獨立的 batch_process_zeczec.py，現在合併進同一支檔案）
# ══════════════════════════════════════════════════════════════════════

import csv
import glob



def _find_projects(root_dir: str) -> List[Dict[str, str]]:
    """
    走訪 root_dir 底下所有資料夾，只要該資料夾內有 *_content.txt，
    就視為一個「專案」，並自動配對同前綴的 _plans.txt。

    回傳每個專案的資訊 dict：
      project_id   : 專案資料夾名稱（例如 ES32_hohoka-01）
      prefix       : 檔名前綴（例如 ES32）
      content_path : content.txt 完整路徑
      plans_path   : plans.txt 完整路徑（找不到則為 None）
      category_path: 相對於 root_dir 的分類路徑（例如 預購式專案/教育/成功）
    """
    projects = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        content_files = [f for f in filenames if f.endswith("_content.txt")]
        if not content_files:
            continue
        for content_file in content_files:
            prefix = content_file[: -len("_content.txt")]
            plans_file = f"{prefix}_plans.txt"
            plans_path = os.path.join(dirpath, plans_file)
            if not os.path.exists(plans_path):
                # 找不到同前綴的 plans.txt，嘗試該目錄下唯一的 *_plans.txt 當備援
                candidates = glob.glob(os.path.join(dirpath, "*_plans.txt"))
                plans_path = candidates[0] if len(candidates) == 1 else None

            project_folder = os.path.basename(dirpath.rstrip(os.sep))
            category_path = os.path.relpath(dirpath, root_dir)

            projects.append({
                "project_id":    project_folder,
                "prefix":        prefix,
                "content_path":  os.path.join(dirpath, content_file),
                "plans_path":    plans_path,
                "category_path": category_path,
            })
    return projects


def _load_done_ids(output_csv: str) -> set:
    """讀取已存在的 CSV，取得已經處理過的 project_id 集合，用於續跑。"""
    done = set()
    if os.path.exists(output_csv):
        try:
            with open(output_csv, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("project_id"):
                        done.add(row["project_id"])
        except Exception as e:
            print(f"⚠️  讀取既有 CSV 失敗（將視為從頭開始）：{e}")
    return done


_CSV_FIELDS = [
    "project_id", "category_path", "status", "error_message",
    "plans_count", "plans_core_price", "plans_min_price", "plans_max_price",
    "plans_has_early_bird",
    "product_keyword", "project_core_price", "price_source",
    "market_price_min", "market_price_max", "market_range_str",
    "market_data_quality", "market_platforms", "market_platform_count",
    "price_deviation_ratio", "is_price_competitive",
]


def batch_run(root_dir: str,
              output_csv: str = "market_price_results.csv",
              max_projects: Optional[int] = None,
              sleep_between: float = 5.0) -> str:
    """
    走訪 root_dir 底下所有專案，逐一分析後輸出成 CSV。

    root_dir      : 資料集根目錄
    output_csv    : 輸出 CSV 路徑；若檔案已存在，會自動跳過已處理過的 project_id（續跑）
    max_projects  : 限制最多處理幾個專案（測試用，None 代表全部處理）
    sleep_between : 每個專案處理完後的間隔秒數，避免短時間內打太多次 API。
                    免費層級 RPM 額度可能很低（例如 5 RPM，等於平均每 12 秒
                    才能打 1 次），建議設定 5~10 秒以上，實際撞到限流時
                    _call_gemini 內部也會用 Google 建議的 retryDelay 自動等待重試。
    """
    projects = _find_projects(root_dir)
    print(f"📂 掃描到 {len(projects)} 個專案（含 *_content.txt 的資料夾）")

    done_ids = _load_done_ids(output_csv)
    if done_ids:
        print(f"↩️  偵測到既有結果 {len(done_ids)} 筆，將自動跳過已處理過的專案")

    todo = [p for p in projects if p["project_id"] not in done_ids]
    if max_projects is not None:
        todo = todo[:max_projects]
    print(f"🚀 本次將處理 {len(todo)} 個專案\n")

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("找不到 GEMINI_API_KEY，請先設定環境變數 / Colab Secrets / 互動輸入。")
    client = genai.Client(api_key=api_key)
    agent = MarketPriceAgent(client, content_text="")

    file_exists = os.path.exists(output_csv)
    with open(output_csv, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()

        success_count = 0
        fail_count = 0
        t0 = time.time()

        for i, proj in enumerate(todo, 1):
            pid = proj["project_id"]
            print(f"[{i}/{len(todo)}] 處理 {pid} ...")
            row = {k: "" for k in _CSV_FIELDS}
            row["project_id"] = pid
            row["category_path"] = proj["category_path"]

            try:
                if not proj["plans_path"]:
                    raise FileNotFoundError("找不到對應的 *_plans.txt")

                content_text = _read_file(proj["content_path"])
                plans_text = _read_file(proj["plans_path"])
                if not content_text.strip():
                    raise ValueError("content.txt 內容為空")

                plans_info = PlansParser().parse(plans_text)
                for k in ("plans_count", "plans_core_price", "plans_min_price",
                          "plans_max_price", "plans_has_early_bird"):
                    row[k] = plans_info.get(k, "")

                result = agent.analyze(content_text, plans_info=plans_info)
                for k in ("product_keyword", "project_core_price", "price_source",
                          "market_price_min", "market_price_max", "market_range_str",
                          "market_data_quality", "market_platforms",
                          "market_platform_count", "price_deviation_ratio",
                          "is_price_competitive"):
                    row[k] = result.get(k, "")

                row["status"] = "success" if result.get("market_price_max", 0) else "no_market_data"
                success_count += 1
                print(f"  ✅ {result.get('product_keyword')} | "
                      f"核心 NT${result.get('project_core_price', 0):,} | "
                      f"行情 {result.get('market_range_str')}")

            except DailyQuotaExceededError as e:
                row["status"] = "error"
                row["error_message"] = f"每日額度用完：{e}"
                writer.writerow(row)
                f.flush()
                print(f"\n{'=' * 50}")
                print("🛑 Gemini API 每日額度已用完，批次處理提前停止。")
                print(f"   已完成 {i-1}/{len(todo)} 筆（{success_count} 成功 / {fail_count} 失敗），"
                      f"結果已存到 {output_csv}。")
                print("   要等到太平洋時間午夜 00:00（約台灣時間下午 3~4 點）額度重置後，"
                      "重新執行同一段程式碼即可（會自動跳過已完成的專案，接續處理）。")
                print(f"{'=' * 50}")
                return output_csv

            except Exception as e:
                row["status"] = "error"
                row["error_message"] = str(e)
                fail_count += 1
                print(f"  ❌ 失敗：{e}")

            writer.writerow(row)
            f.flush()  # 每筆立即寫入，中途中斷也不會遺失前面已完成的結果
            time.sleep(sleep_between)

    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f"✅ 批次處理完成：成功 {success_count} / 失敗 {fail_count}"
          f"（共 {len(todo)} 筆，耗時 {elapsed/60:.1f} 分鐘）")
    print(f"📄 結果已輸出到：{output_csv}")
    print(f"{'=' * 50}")
    return output_csv


# ══════════════════════════════════════════════════════════════════════
# 執行入口：可以單一測試一個專案，也可以批次跑整個資料集
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── 模式 A：單一專案測試 ──────────────────────────────────────
    # run_test(
    #     plans_file   = PLANS_FILE,
    #     content_file = CONTENT_FILE,
    # )

    # ── 模式 B：批次處理整個資料集資料夾，輸出成 CSV ──────────────
    batch_run(
        root_dir      = "ZecZec_Dataset_E&G",   # 換成你實際的資料集根目錄
        output_csv    = "market_price_results.csv",
        max_projects  = None,   # 測試時可先設 5，跑通後再改回 None 跑全部
        sleep_between = 5.0,
    )
