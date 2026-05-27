# 檔案名稱：test_market_price.py
# 版本：v4.0（三層策略，確保一定有市場價輸出）
# 說明：獨立測試腳本，直接讀取 *_plans.txt + *_content.txt，
#       執行完整市場價格比對流程並印出診斷報告。
#       ✅ 不需要 SerpAPI key
#       第1層：Grounding 搜尋 → 第2層：放寬品類 → 第3層：AI純知識推斷
#       ✅ Embedding 向量篩選候選關鍵字
#
# 使用方式：
#   os.environ["GEMINI_API_KEY"] = "AIza..."
#   run_test("PF2_plans.txt", "PF2_content.txt")

import subprocess, sys

def _pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg, imp in [("google-genai", "google.genai"), ("numpy", "numpy"),
                 ("scikit-learn", "sklearn"), ("requests", "requests")]:
    try:
        __import__(imp)
    except ImportError:
        _pip_install(pkg)

import os, re, json, time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
from typing import Optional, Dict, Any, Tuple, List

# ══════════════════════════════════════════════════════════════════════
# ⚙️  設定區
# ══════════════════════════════════════════════════════════════════════

PLANS_FILE   = "PF1_plans.txt"
CONTENT_FILE = "PF1_content.txt"

os.environ["GEMINI_API_KEY"] = "AIzaSyApjbcyyxR5vlP4aDdE0ba3Wd3fd5cf3L8"

CACHE_FILE      = "market_price_cache.json"
LLM_MAX_CHARS   = 3000
GEMINI_MODEL    = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-004"

# KEYWORD_BANK 已移除：改由 content.txt 動態萃取（見 _extract_keywords_from_content）

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

def _call_gemini(client: genai.Client, prompt: str,
                 system: str = "", max_tokens: int = 256,
                 json_mode: bool = True,
                 use_search: bool = False) -> Optional[str]:
    cfg_kwargs: Dict[str, Any] = {"max_output_tokens": max_tokens}
    # ⚠️ Google Search Grounding 不支援 response_mime_type，必須分開處理
    if json_mode and not use_search:
        cfg_kwargs["response_mime_type"] = "application/json"
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
        return "".join(parts).strip() if parts else None
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower():
            print("  ⚠️  Rate Limit，等待 15 秒...")
            time.sleep(15)
        elif "INVALID_ARGUMENT" in err or "mime" in err.lower():
            print(f"  ⚠️  API 參數錯誤（可能 mime_type 衝突）：{e}")
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
            is_multi = bool(self._MULTI_KEYWORDS.search(title))
            is_early = bool(self._EARLY_BIRD.search(title))
            is_addon = bool(self._ADDON.search(title))
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
                    "qty":            qty,
                    "sale_price":     sale_p,
                    "unit_price":     unit_p,
                    "original_price": orig_p,
                })

        if not parsed:
            return {"plans_count": 0, "plans_core_price": 0, "plans_min_price": 0,
                    "plans_max_price": 0, "plans_has_early_bird": 0, "parsed_plans": []}

        # ── 核心售價選取邏輯 ────────────────────────────────────────
        # 優先順序：
        # 1. 非早鳥、非多入、非加購 的單入標準方案（取 unit_price 最高者，代表正常定價）
        # 2. 若全部都是早鳥 → 非多入早鳥方案中取最高價（早鳥最貴的通常最接近正常定價）
        # 3. 若全部都是多入 → 取 unit_price 最高者
        # 4. 兜底：所有方案 unit_price 中位數

        def _pick_core(plans: List[dict]) -> int:
            # tier-1: 標準單入
            tier1 = [p for p in plans if not p["is_multi"] and not p["is_early_bird"] and not p["is_addon"]]
            if tier1:
                return max(p["unit_price"] for p in tier1)
            # tier-2: 標準多入（拆算單價）
            tier2 = [p for p in plans if not p["is_early_bird"] and not p["is_addon"]]
            if tier2:
                return max(p["unit_price"] for p in tier2)
            # tier-3: 早鳥單入
            tier3 = [p for p in plans if not p["is_multi"] and not p["is_addon"]]
            if tier3:
                return max(p["unit_price"] for p in tier3)
            # tier-4: 全體中位數
            all_units = sorted(p["unit_price"] for p in plans)
            return all_units[len(all_units) // 2]

        core_price = _pick_core(parsed)
        all_p      = [p["sale_price"] for p in parsed]
        single_p   = [p["sale_price"] for p in parsed if not p["is_multi"]]

        return {
            "plans_count":         len(parsed),
            "plans_core_price":    core_price,           # ← 主要比較用，非早鳥標準單入價
            "plans_min_price":     min(single_p) if single_p else min(all_p),
            "plans_max_price":     max(all_p),
            "plans_has_early_bird": int(any(p["is_early_bird"] for p in parsed)),
            "parsed_plans":        parsed,
        }


# ══════════════════════════════════════════════════════════════════════
# 市場價格 Agent
# ══════════════════════════════════════════════════════════════════════

class MarketPriceAgent:

    _EXTRACT_KW_SYSTEM = """你是專業電商商品分類專家。
任務：從以下募資文案中，找出實際販售的商品，萃取5~10個消費者在蝦皮/PChome搜尋時會用的通用關鍵字。
規則（嚴格遵守）：
- 關鍵字必須直接對應文案中提到的商品或商品類型，不可憑空創造
- 不含品牌名、創作者名、行銷詞、形容詞
- 每個關鍵字為2~8字的中文名詞
- 優先提取主商品名稱，再列出材質/使用情境的組合詞
- 若文案只有一種商品，仍需列出該商品的不同描述方式（例：勵志卡片、心靈卡牌、插畫小卡）
回傳純 JSON，不含任何說明文字：{"keywords": ["詞1", "詞2", ...]}"""

    _KW_SYSTEM = """你是專業電商數據分析師。
任務：從【候選關鍵字清單】中，選出最能代表募資商品、且消費者在電商平台搜尋時最常用的關鍵字。
規則：
- 必須從候選清單中選出，不可自行創造新詞
- 選擇最通用、搜尋量最高的詞（不含品牌名、行銷詞）
- 若候選清單中多個詞都符合，選最精準的一個
回傳純 JSON：{"keyword": "（從清單中選出的詞）"}"""

    _KW_PRICE_SYSTEM = """你是專業電商數據分析師，專門分析台灣群眾募資文案。
任務：從【候選關鍵字清單】中選出最符合文案商品的關鍵字，並找出核心售價。
規則：
- 關鍵字必須從候選清單中選出，不可自行創造新詞
- 選擇最通用、消費者搜尋電商平台最常用的詞（不含品牌名、行銷詞）
- 售價：取單入/標準版方案的新台幣整數售價，排除雙入組/團購方案，找不到填0
回傳純 JSON：{"keyword": "（從清單中選出的詞）", "core_price": 數字}"""

    _PRICE_SEARCH_SYSTEM = """你是台灣電商價格分析師。
任務：根據提供的【候選搜尋詞清單】，在台灣市場找出真實售價。

🚨【重要搜尋策略】🚨
Google Search Grounding 不支援 site: 指令，請直接用自然語言搜尋：
- 優先搜尋「{關鍵字} 台灣 售價」或「{關鍵字} 蝦皮 價格」
- 若無結果，改用「{關鍵字} 購買」或同類商品名稱
- 出版品/書籍：加上「博客來」、「誠品」、「金石堂」
- 禁止加 site: 指令（會導致無結果）

搜尋策略（依序嘗試，直到找到足夠價格）：
1. 最精準詞 + 「台灣 售價」
2. 較廣泛同類詞 + 「購買」
3. 功能相近替代品類 + 「價格」

排除：嘖嘖/flyingV 募資平台、二手商品、海外代購。

回傳純 JSON（不含 markdown fence 或多餘說明）：
{
  "searched_keyword": "實際用來搜尋的詞",
  "prices_found": [所有有效售價整數，單位新台幣],
  "price_min": Q25或最低合理價（找不到填0）,
  "price_max": Q75或最高合理價（找不到填0）,
  "data_quality": "high"/"medium"/"low",
  "sources_note": "資料來源與搜尋策略說明"
}"""

    def __init__(self, client: genai.Client, content_text: str):
        self.client  = client
        self._cache: Dict[str, Any] = self._load_cache()
        print("  🔄 從 content 文案動態萃取關鍵字庫...")
        self.keyword_bank = self._extract_keywords_from_content(content_text)
        print(f"  📚 萃取到 {len(self.keyword_bank)} 個關鍵字：{self.keyword_bank}")
        print("  🔄 初始化 Embedding 關鍵字庫向量...")
        self.bank_vecs = self._embed_batch(self.keyword_bank) if self.keyword_bank else np.zeros((0, 768))

    def _extract_keywords_from_content(self, content_text: str) -> List[str]:
        prompt = (
            f"請閱讀以下募資文案，判斷主要商品類型，"
            f"萃取5~10個消費者在蝦皮/PChome搜尋時會用的通用關鍵字（不含品牌名、折扣詞）。\n\n"
            f"【募資文案】：\n{content_text[:LLM_MAX_CHARS]}\n\n"
            f'回傳純 JSON 格式（不要加任何說明）：{{"keywords": ["詞1", "詞2", ...]}}'
        )
        for attempt, json_mode in enumerate((True, False), 1):
            raw = _call_gemini(self.client, prompt,
                               system=self._EXTRACT_KW_SYSTEM,
                               max_tokens=512, json_mode=json_mode)
            if raw:
                r = _parse_json(raw)
                if r and isinstance(r.get("keywords"), list):
                    kws = [str(k).strip() for k in r["keywords"] if str(k).strip()]
                    if kws:
                        return kws
                print(f"  ⚠️  萃取第{attempt}次 parse 失敗，raw={raw[:120]!r}")
            else:
                print(f"  ⚠️  萃取第{attempt}次 Gemini 無回應")
        print("  ⚠️  關鍵字萃取失敗，以空白字庫繼續（Gemini 將自行生成）")
        return []

    def _embed_batch(self, texts: List[str],
                     task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        try:
            resp = self.client.models.embed_content(
                model=EMBEDDING_MODEL, contents=texts,
                config=types.EmbedContentConfig(task_type=task)
            )
            return np.array([e.values for e in resp.embeddings])
        except Exception as e:
            print(f"  ⚠️  Embedding 失敗：{e}")
            return np.zeros((len(texts), 768))

    def _embed_query(self, text: str) -> np.ndarray:
        try:
            resp = self.client.models.embed_content(
                model=EMBEDDING_MODEL, contents=text[:1000],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            return np.array(resp.embeddings[0].values)
        except Exception as e:
            print(f"  ⚠️  Query Embedding 失敗：{e}")
            return np.zeros(768)

    def _filter_candidates(self, content_text: str, top_k: int = 3) -> List[str]:
        if not self.keyword_bank:
            print("  ❌  關鍵字萃取失敗，無法繼續。請確認 content 文案內容。")
            return []
        qvec = self._embed_query(content_text)
        if not self.bank_vecs.any():
            return self.keyword_bank[:top_k]
        sims    = cosine_similarity([qvec], self.bank_vecs)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        best    = float(sims[top_idx[0]])
        if best < 0.25:
            print(f"  ℹ️  字庫相似度低（{best:.3f}），回傳全部關鍵字供 Gemini 從文案中選擇。")
            return self.keyword_bank
        candidates = [self.keyword_bank[i] for i in top_idx]
        print(f"  🔍 向量候選詞：{candidates}（最高相似度 {best:.3f}）")
        return candidates

    def _get_keyword(self, text: str, candidates: List[str]) -> str:
        cand_str = "、".join(candidates)
        prompt = (f"【候選關鍵字清單】：{cand_str}\n\n"
                  f"【募資文案】：\n{text[:LLM_MAX_CHARS]}")
        raw = _call_gemini(self.client, prompt, system=self._KW_SYSTEM, max_tokens=64)
        if raw:
            r = _parse_json(raw)
            if r and r.get("keyword"):
                return str(r["keyword"])
        return candidates[0]

    def _get_keyword_and_price(self, text: str,
                               candidates: List[str]) -> Tuple[str, int]:
        cand_str = "、".join(candidates)
        prompt = (f"【候選關鍵字清單】：{cand_str}\n\n"
                  f"【募文案】：\n{text[:LLM_MAX_CHARS]}")
        raw = _call_gemini(self.client, prompt,
                           system=self._KW_PRICE_SYSTEM, max_tokens=128)
        if raw:
            r = _parse_json(raw)
            if r:
                kw = str(r.get("keyword", candidates[0] if candidates else "未分類商品"))
                try:
                    price = int(float(str(r.get("core_price", 0)).replace(',', '')))
                except (ValueError, TypeError):
                    price = 0
                return kw, price
        m = re.search(r'(?:單入|標準版|早鳥).*?NT. ([0-9,]+)', text)
        price = int(m.group(1).replace(',', '')) if m else 0
        return (candidates[0] if candidates else ""), price

    # ── 純知識推斷：不需要網路，直接問 Gemini ──────────────────────
    _AI_ESTIMATE_SYSTEM = """你是台灣電商市場價格專家，熟悉蝦皮、PChome、momo、博客來等平台的商品行情。
根據商品關鍵字，憑你的訓練知識給出台灣市場的合理售價區間。

規則：
- 必須給出具體數字，不可回傳 0
- 以台灣消費市場正常零售價為準（非特賣、非批發）
- 若商品冷門或不確定，以同類型商品的常見售價估算，並在 note 說明
- 偏離幅度允許 ±40%，寧可範圍寬一點也要給數字

回傳純 JSON（禁止 markdown fence 或任何說明文字）：
{"price_min": 整數, "price_max": 整數, "note": "估算依據一句話說明", "confidence": "high"/"medium"/"low"}"""

    def _fetch_market_range(self, keyword: str,
                            fallback_keywords: Optional[List[str]] = None
                            ) -> Tuple[int, int, str, str]:

        seen: set = {keyword}
        all_kws: List[str] = [keyword]
        for kw in (fallback_keywords or []):
            if kw not in seen:
                seen.add(kw)
                all_kws.append(kw)

        cache_key = f"v5:{'|'.join(all_kws)}"
        if cache_key in self._cache:
            c = self._cache[cache_key]
            print(f"  [快取] {c.get('searched_keyword', keyword)} → {c['range_str']} [{c['quality']}]")
            return c["min"], c["max"], c["range_str"], c["quality"]

        kw_list_str = "、".join(all_kws)

        # ── 從原始數字中解析價格 ────────────────────────────────────
        def _parse_prices_from_raw(raw: str) -> Tuple[int, int, str, List[int]]:
            """從任意文字中用 regex 強制抽出價格數字"""
            prices: List[int] = []
            # 先找 JSON 物件中的 prices_found / price_min / price_max
            json_match = re.search(r'\{[\s\S]*?\}', raw)
            if json_match:
                r = _parse_json(json_match.group(0)) or {}
                for field in ("prices_found",):
                    val = r.get(field, [])
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

            # 再從全文抓 NT$/元/售價 後面的數字
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

            # 最後從全文掃所有 3~6 位數（取最密集的數量級）
            if len(prices) < 2:
                candidates = []
                for m in re.finditer(r'\b([1-9][0-9]{2,5})\b', raw):
                    n = int(m.group(1))
                    if 50 < n < 500_000:
                        candidates.append(n)
                if candidates:
                    # 找出眾數數量級（百位 / 千位 / 萬位）
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

        market_min = market_max = 0
        data_quality = "low"
        searched_kw  = keyword
        prices_found: List[int] = []
        source_label = ""

        # ══ 第一層：Google Search Grounding（有網路時最準確）══════════
        print(f"  🌐 [第1層] Google Search Grounding，候選詞：【{kw_list_str}】")
        grounding_prompt = (
            f"請搜尋台灣電商（蝦皮、PChome、momo、博客來）上「{kw_list_str}」的售價。\n"
            f"列出至少3筆實際售價（新台幣整數），計算最低與最高合理售價。\n"
            f"排除：嘖嘖/flyingV 募資價、二手商品、海外代購。\n"
            f"回傳純 JSON（禁止 markdown fence）：\n"
            f'{{"searched_keyword":"{keyword}","prices_found":[價格1,價格2,...],'
            f'"price_min":數字,"price_max":數字,"data_quality":"high"/"medium"/"low",'
            f'"sources_note":"來源說明"}}'
        )
        raw_g = _call_gemini(
            self.client, grounding_prompt,
            system=self._PRICE_SEARCH_SYSTEM,
            max_tokens=800, json_mode=False, use_search=True
        )
        if raw_g:
            print(f"  📥 Grounding 原始回傳（前300字）：{raw_g[:300]!r}")
            market_min, market_max, data_quality, prices_found = _parse_prices_from_raw(raw_g)
            if market_max > 0:
                searched_kw  = keyword
                source_label = "Grounding搜尋"
                print(f"  ✅ Grounding 取得行情：NT${market_min:,}~NT${market_max:,}（{len(prices_found)}筆）")

        # ══ 第二層：Grounding 放寬品類重試 ════════════════════════════
        if market_max == 0:
            print(f"  ⚠️  [第2層] Grounding 無結果，放寬品類重試...")
            wider_prompt = (
                f"商品關鍵字：{kw_list_str}\n"
                f"若找不到完全一樣的商品，請用同類型或上層品類搜尋（例：特定書籍→同類書籍定價）。\n"
                f"在台灣市場找出至少2筆參考售價（新台幣），並說明使用的替代搜尋詞。\n"
                f"回傳純 JSON（禁止 markdown fence）：\n"
                f'{{"searched_keyword":"實際搜尋詞","prices_found":[價格1,價格2,...],'
                f'"price_min":數字,"price_max":數字,"data_quality":"high"/"medium"/"low",'
                f'"sources_note":"說明替代品類"}}'
            )
            raw_g2 = _call_gemini(
                self.client, wider_prompt,
                system=self._PRICE_SEARCH_SYSTEM,
                max_tokens=800, json_mode=False, use_search=True
            )
            if raw_g2:
                print(f"  📥 第2層回傳（前300字）：{raw_g2[:300]!r}")
                market_min, market_max, data_quality, prices_found = _parse_prices_from_raw(raw_g2)
                if market_max > 0:
                    source_label = "Grounding放寬搜尋"
                    print(f"  ✅ 第2層取得行情：NT${market_min:,}~NT${market_max:,}（{len(prices_found)}筆）")

        # ══ 第三層：純 AI 知識推斷（不需網路，一定有結果）════════════
        if market_max == 0:
            print(f"  🤖 [第3層] Grounding 失敗，改用 Gemini 純知識推斷...")
            ai_prompt = (
                f"商品關鍵字：{kw_list_str}\n\n"
                f"請根據你對台灣電商市場（蝦皮/PChome/momo）的知識，"
                f"估算此類商品的合理售價區間（新台幣）。\n"
                f"必須給出具體數字，不可為 0。若不確定，以同類商品估算並說明。"
            )
            raw_ai = _call_gemini(
                self.client, ai_prompt,
                system=self._AI_ESTIMATE_SYSTEM,
                max_tokens=256, json_mode=True, use_search=False
            )
            if raw_ai:
                print(f"  📥 AI推斷回傳：{raw_ai[:200]!r}")
                r_ai = _parse_json(raw_ai) or {}
                try:
                    p_min = int(str(r_ai.get("price_min", 0) or 0).replace(',', ''))
                    p_max = int(str(r_ai.get("price_max", 0) or 0).replace(',', ''))
                except Exception:
                    p_min = p_max = 0

                # 若 JSON 欄位為 0，仍嘗試從原始文字抽數字
                if p_max == 0:
                    p_min, p_max, _, _ = _parse_prices_from_raw(raw_ai)

                if p_max > 0 and p_min >= 0:
                    if p_min == 0:
                        p_min = int(p_max * 0.6)
                    market_min   = p_min
                    market_max   = p_max
                    data_quality = r_ai.get("confidence", "low")
                    source_label = f"AI知識推斷（{r_ai.get('note','估算')}）"
                    prices_found = [p_min, p_max]
                    print(f"  🤖 AI推斷行情：NT${market_min:,}~NT${market_max:,}（{data_quality}）")
                    if r_ai.get("note"):
                        print(f"  📌 {r_ai['note']}")

        # ── 最終輸出 ────────────────────────────────────────────────
        if market_max > 0:
            range_str = f"NT${market_min:,} ~ NT${market_max:,}"
            tag = f"【{source_label}】" if source_label else ""
            print(f"  📈 市場行情 {range_str} {tag}")
        else:
            # 完全兜底：三層全失敗也不回傳 0，改回 "無法取得"
            range_str = "無法取得（Gemini API 未回應）"
            print("  ❌  三層策略均未取得行情，請確認 Gemini API Key 是否有效。")

        self._cache[cache_key] = {
            "min": market_min, "max": market_max,
            "range_str": range_str, "quality": data_quality,
            "searched_keyword": searched_kw,
        }
        self._save_cache()
        return market_min, market_max, range_str, data_quality

    def analyze(self, content_text: str,
                plans_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        candidates = self._filter_candidates(content_text)

        if plans_info and plans_info.get("plans_core_price", 0) > 0:
            core_price   = plans_info["plans_core_price"]   # ← 標準單入非早鳥價
            price_source = "plans"
            keyword      = self._get_keyword(content_text, candidates)
        else:
            keyword, core_price = self._get_keyword_and_price(content_text, candidates)
            price_source        = "llm" if core_price > 0 else "regex"

        market_min, market_max, range_str, quality = self._fetch_market_range(keyword, self.keyword_bank)

        is_competitive = 0
        deviation      = None
        if market_max > 0:
            is_competitive = 1 if market_min <= core_price <= market_max else 0
            median         = (market_min + market_max) / 2
            deviation      = round((core_price - median) / median, 4) if median else 0.0

        return {
            "product_keyword":       keyword,
            "project_core_price":    core_price,
            "price_source":          price_source,
            "market_price_min":      market_min,
            "market_price_max":      market_max,
            "market_range_str":      range_str,
            "market_data_quality":   quality,
            "price_deviation_ratio": deviation,
            "is_price_competitive":  is_competitive,
        }

    def _load_cache(self) -> dict:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

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
        if p.get("is_early_bird"): tags.append("🐦早鳥")
        if p.get("is_multi"):      tags.append(f"×{p.get('qty',2)}")
        if p.get("is_addon"):      tags.append("加購")
        is_core = core_price > 0 and p["unit_price"] == core_price and not p.get("is_early_bird") and not p.get("is_addon")
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
        "high":   "高（3筆以上）",
        "medium": "中（1~2筆）",
        "low":    "低（估算值）",
    }.get(qual, qual)

    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║              市場價格競爭力診斷報告                  ║
  ╠══════════════════════════════════════════════════════╣
  ║  產品關鍵字   {kw:<38}║
  ║  核心售價     NT${cp:<6,}  （來源：{src:<8}）           ║
  ║  市場行情     {rstr:<40}║
  ║  資料品質     {quality_label:<38}║
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
    print(" 🧪 市場價格比對測試 v3.0")
    print(f"    model   : {GEMINI_MODEL}")
    print(f"    plans   : {plans_file}")
    print(f"    content : {content_file}")
    print(f"    搜尋方式 : Google Search Grounding（無需 SerpAPI）")
    print("=" * 62)

    plans_text   = _read_file(plans_file)
    content_text = _read_file(content_file)
    if not content_text.strip():
        print("❌ content 檔案無法讀取，終止。"); return

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("❌ 找不到 GEMINI_API_KEY。請設定：os.environ['GEMINI_API_KEY'] = 'AIza...'"); return
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

    # ── 步驟 2：Embedding 篩選 + Gemini 裁決 + Google Search 行情 ──
    print("\n【步驟 2】向量篩選 → Gemini 裁決關鍵字 → Google Search 搜尋行情...")
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
    print(f"  偏離幅度    ：{f'{dev:+.2%}' if dev is not None else 'N/A'}")
    print(f"  競爭力判定  ：{'✅ 具競爭力' if price_result['is_price_competitive'] else '⚠️  偏離市場'}\n")


if __name__ == "__main__":
    run_test(
        plans_file   = PLANS_FILE,
        content_file = CONTENT_FILE,
    )