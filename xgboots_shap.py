"""
完整流程：XGBoost → SHAP → LLM 逐特徵解釋 → HTML 互動報告
===========================================================
支援三種 LLM 後端（依優先順序自動選擇）：
  1. Ollama  — 本地免費，不需 API Key（推薦）
  2. Gemini  — 雲端 Google API
  3. 跳出錯誤（兩者都不可用時）

執行方式：
  # 只用 Ollama（推薦）
  python xgboost_shap_gemini.py

  # 指定 Ollama 模型（預設 gemma3:4b）
  python xgboost_shap_gemini.py --model llama3.2

  # 用 Gemini
  set GEMINI_API_KEY=你的金鑰          # Windows
  export GEMINI_API_KEY="你的金鑰"     # Mac/Linux
  python xgboost_shap_gemini.py --backend gemini

  # 指定分析的專案編號（預設 5）
  python xgboost_shap_gemini.py --project 12

Ollama 安裝與模型下載說明：
  請見 README 區段，或執行後查看自動印出的安裝指引。
===========================================================
"""

import os, sys, json, warnings, textwrap, argparse, time, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
import urllib.request, urllib.error

# ── 選用套件 ──────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as gtypes
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════
# 0. 維度 / 特徵常數
# ══════════════════════════════════════════════════════════════════
DIMENSION_MAP = {
    "專案執行力":    ["目標金額", "募資天數"],
    "價格競爭力":    ["回饋方案層數", "回饋方案金額中位數", "定價區間合理性"],
    "文案說服力":    ["文案字數", "文案感情字詞比例", "文案規格字詞比例", "FAQ更新比例", "FAQ題數"],
    "影像吸引力":    ["每百字多媒體密度", "有影片", "實際操作情境影片"],
    "產品市場契合度": ["平台內同類商品近期成功率", "Google趨勢斜率", "爆款相似度"],
}
ALL_FEATURES = [f for feats in DIMENSION_MAP.values() for f in feats]

FEATURE_META = {
    # feature: (說明, 單位, 最佳範圍低, 最佳範圍高)
    "目標金額":                ("募資目標金額",            "新台幣元",  100_000,  1_000_000),
    "募資天數":                ("募資活動持續天數",          "天",        25,       40),
    "回饋方案層數":            ("回饋方案檔位數量",          "層",        5,        8),
    "回饋方案金額中位數":      ("所有方案金額的中位數",       "新台幣元",  800,      2_500),
    "定價區間合理性":          ("相對市場價格的合理程度",    "0–1分",    0.7,      1.0),
    "文案字數":                ("文案總字數（含OCR）",       "字",        3_500,    5_000),
    "文案感情字詞比例":        ("情感/故事性字詞佔比",       "比例",      0.12,     0.20),
    "文案規格字詞比例":        ("規格/數據性字詞佔比",       "比例",      0.15,     0.28),
    "FAQ更新比例":             ("活動期間FAQ更新比例",       "比例",      0.5,      1.0),
    "FAQ題數":                 ("FAQ總題數",                "題",        15,       25),
    "每百字多媒體密度":        ("每百字的圖片/影片數",       "個",        2.5,      4.0),
    "有影片":                  ("專案頁是否含影片",          "0或1",      1,        1),
    "實際操作情境影片":        ("是否有真人使用情境影片",    "0或1",      1,        1),
    "平台內同類商品近期成功率": ("同類商品近90天成功率",     "比例",      0.5,      1.0),
    "Google趨勢斜率":          ("近3個月搜尋趨勢斜率",      "-1到1",     0.1,      1.0),
    "爆款相似度":              ("與歷史爆款的相似程度",      "比例",      0.6,      1.0),
}

# ══════════════════════════════════════════════════════════════════
# 1. 資料生成
# ══════════════════════════════════════════════════════════════════
def make_dataset(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "目標金額":                rng.uniform(50_000, 5_000_000, n),
        "募資天數":                rng.integers(15, 60, n).astype(float),
        "回饋方案層數":            rng.integers(1, 10, n).astype(float),
        "回饋方案金額中位數":      rng.uniform(500, 10_000, n),
        "定價區間合理性":          rng.uniform(0, 1, n),
        "文案字數":                rng.integers(500, 5000, n).astype(float),
        "文案感情字詞比例":        rng.uniform(0, 0.3, n),
        "文案規格字詞比例":        rng.uniform(0, 0.4, n),
        "FAQ更新比例":             rng.uniform(0, 1, n),
        "FAQ題數":                 rng.integers(0, 20, n).astype(float),
        "每百字多媒體密度":        rng.uniform(0, 5, n),
        "有影片":                  rng.integers(0, 2, n).astype(float),
        "實際操作情境影片":        rng.integers(0, 2, n).astype(float),
        "平台內同類商品近期成功率": rng.uniform(0, 1, n),
        "Google趨勢斜率":          rng.uniform(-1, 1, n),
        "爆款相似度":              rng.uniform(0, 1, n),
    })[ALL_FEATURES]
    logit = (
        df["平台內同類商品近期成功率"] * 2.5
        + df["有影片"] * 1.8
        + df["爆款相似度"] * 1.2
        + df["文案感情字詞比例"] * 2.0
        - df["目標金額"] / 2_500_000
        + df["定價區間合理性"] * 0.8
        + rng.normal(0, 0.6, n)
    )
    return df, (logit > 1.2).astype(int)

# ══════════════════════════════════════════════════════════════════
# 2. XGBoost
# ══════════════════════════════════════════════════════════════════
def train_model(X, y):
    print("▶ [Step 1] 訓練 XGBoost ...")
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    m.fit(X, y, eval_set=[(X, y)], verbose=False)
    acc = ((m.predict_proba(X)[:,1] >= 0.5).astype(int) == y).mean()
    print(f"   訓練準確率: {acc:.3f}")
    return m

# ══════════════════════════════════════════════════════════════════
# 3. SHAP
# ══════════════════════════════════════════════════════════════════
def run_shap(model, X):
    print("▶ [Step 2] 計算 SHAP 值 ...")
    exp = shap.TreeExplainer(model)
    sv  = exp(X)
    return exp, sv

def build_summary(sv, X, idx):
    mat       = sv.values
    base_val  = float(sv.base_values[idx])
    proj_shap = dict(zip(X.columns, mat[idx].tolist()))
    proj_vals = dict(zip(X.columns, X.iloc[idx].tolist()))
    global_imp= dict(zip(X.columns, np.abs(mat).mean(axis=0).tolist()))
    pred_prob = float(expit(base_val + sum(proj_shap.values())) * 100)
    base_prob = float(expit(base_val) * 100)
    return dict(
        global_importance=global_imp, project_shap=proj_shap,
        project_values=proj_vals, base_value=base_val,
        predicted_prob=pred_prob, base_prob=base_prob,
    )

# ══════════════════════════════════════════════════════════════════
# 4. Shared Prompt 生成（Ollama & Gemini 共用）
# ══════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = textwrap.dedent("""\
你是一位在台灣擁有十年實戰經驗的群眾募資顧問，同時精通機器學習可解釋性。
你的任務是根據 XGBoost + SHAP 數據，為專案每一個特徵寫出「可直接執行的優化建議」。

【絕對禁止的萬用句】以下說法一律禁止出現：
  ✗ 「參考優秀案例」「參考成功專案」「參考業界標竿」
  ✗ 「提升品質」「加強內容」「持續優化」（沒有具體數字的空話）
每一條建議必須包含：①現況數字 ②目標數字或範圍 ③具體操作步驟（至少一個動詞行動）

【各特徵的業界標竿數字（撰寫建議時必須引用對應數字）】

目標金額：
  - 台灣平台成功率最高區間：10–100 萬（成功率>72%）；300 萬以上成功率驟降至 <35%
  - 建議公式：初始目標 = 最低開模/印刷成本（MVP）× 1.2
  - 若超標：拆分為「初始目標 + Stretch Goals」，Stretch 設在 100%、200%、500%

募資天數：
  - 最佳區間：25–40 天；黃金節奏：前 3 天衝 30%、中段維熱度、最後 72 小時二次爆發
  - 若 <20 天：延長至 28 天；若 >50 天：縮短至 40 天並排程 3 次限時活動

回饋方案層數：
  - 最佳：5–8 層；必備結構：超早鳥(-20%)、標準、組合包、企業採購
  - 若 <4 層：新增「超早鳥限量 50 份」與「雙入組合包」兩層

回饋方案金額中位數：
  - 台灣甜蜜點：800–2,500 元（最高下單頻率）
  - 若 >3,000：增加 1,200–1,800 元標準方案；若 <600：加配件組合拉高客單

定價區間合理性（0–1）：
  - 0.7 以上合理；0.5 以下嚴重偏離市場
  - 修正：以蝦皮同類商品前 20 名做價格錨點，調整至市場均價 ±15% 內

文案字數：
  - 高成功率中位數：3,500–5,000 字
  - 若 <2,000：依序補充 ①品牌起源故事 300 字 ②每款使用情境 200 字 ③製程說明 400 字
  - 若 >6,000：加錨點目錄，每段加粗重點句

文案感情字詞比例（0–0.3）：
  - 最佳：0.12–0.20；若 <0.08：每 200 字插入 1 個情感錨點句；若 >0.25：改用「數字+情境」取代純感性

文案規格字詞比例（0–0.4）：
  - 最佳：0.15–0.28；若 <0.10：為每個功能補充規格表；若 >0.35：精簡至關鍵 5 項

FAQ更新比例（0–1）：
  - 頭 7 天新增 ≥5 題，之後每週 2–3 題；高更新比例提升後段轉化率 8–15%
  - 比例 <0.3：每天花 15 分鐘回覆留言並整理成 FAQ

FAQ題數：
  - 最低門檻：10 題；理想：15–25 題
  - 必備分類：①常見疑慮 ②規格比較 ③使用方法 ④出貨時程 ⑤適用族群

每百字多媒體密度（0–5）：
  - 最佳：2.5–4.0；若 <1.5：每個功能段落配 1 張情境實拍圖
  - 若 >5.0：壓縮圖片至 WebP 並移除重複展示圖

有影片（0/1）：
  - 有影片的專案成功率比無影片高 40–60%
  - 若 =0：最低要求 60–90 秒開箱影片，手機+自然光+穩定器即可，展示「使用前 vs 使用後」

實際操作情境影片（0/1）：
  - 情境影片比純產品展示影片轉化率高 25–35%
  - 若 =0：在現有影片 30 秒起加入真實用戶使用片段，或補拍 30 秒情境 B-Roll

平台內同類商品近期成功率（0–1）：
  - <0.3：市場寒冬，建議延後 1–2 個月至旺季（11–12 月、3–4 月）或強化差異化
  - >0.6：市場熱絡，加速推進並在文案強調與同類的差異點

Google趨勢斜率（-1 到 1）：
  - 正斜率（>0.2）：在文案標題加入當前熱搜關鍵字
  - 負斜率（<-0.2）：搭配 KOL 開箱或媒體報導拉抬搜尋量；募資前 2 週投放相關社群貼文

爆款相似度（0–1）：
  - >0.7：在文案引用「同類型已有 X 萬人支持」作為社會認同
  - <0.4：提取平台前 3 名爆款的標題關鍵字、主視覺色系、首段文案結構融入本案

【JSON 輸出格式（嚴格遵守，不可有任何 markdown 或前言）】
{
  "overall": {
    "predicted_prob": <float>,
    "base_prob": <float>,
    "summary": "<根據數據說明本案整體優劣勢，點出最關鍵的 1 個風險與 1 個優勢，40–60字>"
  },
  "dimensions": [
    {
      "name": "<維度名稱>",
      "shap_total": <float>,
      "features": [
        {
          "name": "<特徵名稱>",
          "shap": <float>,
          "raw_value": <float>,
          "status": "<建議立即修改 | 建議進行優化 | 請繼續保持>",
          "current_desc": "<將 raw_value 轉為白話，說明現況數值的意義，20–40字>",
          "suggestion": "<必須含①現況數字 ②目標數字/範圍 ③明確動詞行動，60–120字，禁止萬用句>"
        }
      ]
    }
  ]
}

status 判斷邏輯（依 SHAP 值）：
  SHAP < -0.3  → "建議立即修改"
  -0.3 ≤ SHAP < 0  → "建議進行優化"
  SHAP ≥ 0  → "請繼續保持"
""")


def build_user_prompt(summary, project_idx):
    """組裝給 LLM 的 user prompt，攜帶完整 SHAP 數字脈絡。"""
    base_prob = summary["base_prob"]
    pred_prob = summary["predicted_prob"]
    delta     = pred_prob - base_prob

    pos_shaps  = {k: v for k, v in summary["project_shap"].items() if v > 0}
    neg_shaps  = {k: v for k, v in summary["project_shap"].items() if v < 0}
    pos_total  = sum(pos_shaps.values()) or 1
    neg_total  = sum(neg_shaps.values()) or -1

    gi_rank = {k: i+1 for i, k in enumerate(
        sorted(summary["global_importance"], key=lambda k: summary["global_importance"][k], reverse=True)
    )}

    lines = [
        f"=== 專案 #{project_idx} 完整 SHAP 數據包 ===",
        f"預測成功率: {pred_prob:.1f}%",
        f"市場基礎勝率: {base_prob:.1f}%",
        f"自身條件貢獻: {delta:+.1f}% ({'高於' if delta>=0 else '低於'}市場基準)",
        "",
        "【所有特徵（依|SHAP|降序，含業界最佳範圍參考）】",
    ]

    for f, sv in sorted(summary["project_shap"].items(), key=lambda x: abs(x[1]), reverse=True):
        rv  = summary["project_values"].get(f, 0)
        rnk = gi_rank.get(f, "?")
        meta= FEATURE_META.get(f, ("", "", None, None))
        lo, hi = meta[2], meta[3]
        range_str = f"（業界最佳:{lo}–{hi} {meta[1]}）" if lo is not None else ""
        pct = sv/pos_total*100 if sv>=0 else sv/neg_total*100
        pct_str = f"+{pct:.1f}%正向" if sv>=0 else f"{pct:.1f}%負向"
        lines.append(
            f"  {f:<20} SHAP={sv:+.5f}  實際值={rv:<10.4g}  "
            f"重要性排名第{rnk}  {pct_str}  {range_str}"
        )

    urgent = [(f,v) for f,v in sorted(summary["project_shap"].items(),
               key=lambda x: x[1]) if v < -0.3]
    if urgent:
        lines += ["", "【⚠️ 緊急處理清單（SHAP < -0.3，對成功率傷害最大）】"]
        for f, sv in urgent:
            rv   = summary["project_values"].get(f, 0)
            meta = FEATURE_META.get(f, ("","",None,None))
            lo, hi = meta[2], meta[3]
            gap  = f" → 目標需達到 {lo}–{hi}" if lo is not None else ""
            lines.append(f"  ⚠️ {f}：現況={rv:.4g}{gap}，SHAP={sv:+.4f}")

    lines += ["", "【各維度明細】"]
    for dim, feats in DIMENSION_MAP.items():
        dim_shap = sum(summary["project_shap"].get(f,0) for f in feats)
        lines.append(f"▌{dim}  SHAP加總={dim_shap:+.4f}")
        for f in feats:
            sv = summary["project_shap"].get(f, 0)
            rv = summary["project_values"].get(f, 0)
            lines.append(f"    {f}: 實際值={rv:.4g}  SHAP={sv:+.4f}")
        lines.append("")

    lines += [
        "【撰寫規則（最後提醒）】",
        "1. suggestion 必須含 ①現況數字 ②目標數字/範圍 ③明確動詞行動",
        "2. 禁止：「參考優秀案例」「持續優化」「提升品質」",
        "3. 每條建議要讓創作者「今天下班前就知道要做什麼」",
        "4. 嚴格按 JSON 格式，不可有任何前言或結尾說明",
    ]
    return "\n".join(lines)


def parse_llm_json(text: str) -> dict:
    """去除 markdown fences 後解析 JSON；失敗時嘗試提取第一個 {...} 區塊。"""
    text = text.strip()
    # 去掉 ```json ... ``` 包裝
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 嘗試抓出第一個完整 JSON 物件
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise

# ══════════════════════════════════════════════════════════════════
# 5A. Ollama 後端
# ══════════════════════════════════════════════════════════════════
OLLAMA_URL = "http://localhost:11434"

def ollama_is_running() -> bool:
    """偵測 Ollama 服務是否在跑。"""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def ollama_list_models() -> list[str]:
    """列出已下載的 Ollama 模型名稱清單。"""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def ollama_generate(model: str, system: str, user: str,
                    temperature: float = 0.3, timeout: int = 300) -> str:
    """
    呼叫 Ollama /api/chat（非串流），回傳模型輸出文字。
    使用標準 urllib，不需額外安裝套件。
    """
    payload = json.dumps({
        "model":   model,
        "stream":  False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data["message"]["content"]
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama 請求失敗：{e}") from e


def call_ollama_structured(summary, project_idx, model="gemma3:4b") -> dict:
    """
    完整 Ollama 呼叫流程：
    1. 確認服務在跑
    2. 確認模型已下載
    3. 呼叫模型（最多重試 2 次，處理 JSON 解析失敗）
    4. 解析並回傳結構化 JSON
    """
    print(f"▶ [Step 4] 呼叫 Ollama（模型: {model}）產生逐特徵解釋 ...")

    if not ollama_is_running():
        raise RuntimeError(
            "Ollama 服務未啟動！請先執行：ollama serve\n"
            "（或在 Windows 開啟 Ollama 應用程式）"
        )

    available = ollama_list_models()
    # 允許「modelname」或「modelname:tag」兩種格式比對
    base_name = model.split(":")[0]
    matched = [m for m in available if m == model or m.startswith(base_name+":")]
    if not matched:
        raise RuntimeError(
            f"模型 [{model}] 尚未下載！\n"
            f"已有模型：{available or '（無）'}\n"
            f"請執行：ollama pull {model}"
        )
    actual_model = matched[0]
    print(f"   使用模型: {actual_model}")

    user_prompt = build_user_prompt(summary, project_idx)

    for attempt in range(1, 3):
        print(f"   生成中（第 {attempt} 次）... 視模型大小約需 30–120 秒")
        raw = ollama_generate(actual_model, SYSTEM_PROMPT, user_prompt)
        try:
            result = parse_llm_json(raw)
            print("   ✅ JSON 解析成功")
            return result
        except (json.JSONDecodeError, KeyError) as e:
            print(f"   ⚠️  JSON 解析失敗（第 {attempt} 次）：{e}")
            if attempt == 1:
                # 第二次在 prompt 末尾強調只輸出 JSON
                user_prompt += "\n\n重要：只輸出合法 JSON，不可有任何說明文字！"
            else:
                raise RuntimeError(
                    f"Ollama 模型 [{actual_model}] 兩次均無法輸出合法 JSON。\n"
                    "建議換用指令遵循能力更好的模型，例如：\n"
                    "  ollama pull llama3.2\n"
                    "  python xgboost_shap_gemini.py --model llama3.2"
                ) from e

# ══════════════════════════════════════════════════════════════════
# 5B. Gemini 後端（保留原有邏輯）
# ══════════════════════════════════════════════════════════════════
def call_gemini_structured(summary, project_idx, api_key) -> dict:
    print("▶ [Step 4] 呼叫 Gemini 產生逐特徵解釋 ...")
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-genai 套件未安裝，請執行：pip install google-genai")

    MODEL_FALLBACKS = [
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
    ]
    client      = genai.Client(api_key=api_key)
    user_prompt = build_user_prompt(summary, project_idx)
    last_exc    = None

    for model in MODEL_FALLBACKS:
        for attempt in range(1, 4):
            try:
                print(f"   嘗試 {model}（第 {attempt} 次）...")
                resp = client.models.generate_content(
                    model  = model,
                    config = gtypes.GenerateContentConfig(
                        system_instruction = SYSTEM_PROMPT,
                        temperature        = 0.3,
                        max_output_tokens  = 3000,
                    ),
                    contents = user_prompt,
                )
                result = parse_llm_json(resp.text)
                print(f"   ✅ 成功（{model}）")
                return result
            except (json.JSONDecodeError, KeyError) as e:
                last_exc = e
                time.sleep(2)
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    m = re.search(r"(\d+)s", err)
                    wait = int(m.group(1)) + 5 if m else 65
                    if attempt < 3:
                        print(f"   ⏳ 配額超限，等待 {wait} 秒...")
                        time.sleep(wait)
                    last_exc = e
                else:
                    last_exc = e
                    break

    raise RuntimeError(f"所有 Gemini 模型均失敗：{last_exc}")

# ══════════════════════════════════════════════════════════════════
# 6. SHAP 圖表
# ══════════════════════════════════════════════════════════════════
def save_shap_plots(sv, X, idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, fn in [
        ("beeswarm",  lambda: shap.plots.beeswarm(sv, max_display=16, show=False)),
        ("waterfall", lambda: shap.plots.waterfall(sv[idx], max_display=16, show=False)),
        ("bar",       lambda: shap.plots.bar(sv, max_display=16, show=False)),
    ]:
        plt.figure(figsize=(10, 6))
        fn()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_{name}.png"), dpi=130, bbox_inches="tight")
        plt.close()
    print(f"   SHAP 圖表已存至: {out_dir}/")

# ══════════════════════════════════════════════════════════════════
# 7. HTML 報告
# ══════════════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>募資潛力分析報告 #{idx}</title>
<style>
  :root{{
    --gold:#e8a000;--gold-light:#ffd460;--gold-bg:#fffae8;
    --green:#2ecc71;--yellow:#f39c12;--red:#e74c3c;
    --text:#1a1a2e;--muted:#666;--radius:14px;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif;
        background:#f5f0e8;color:var(--text);padding:24px 16px 60px}}
  h1{{text-align:center;font-size:1.5rem;font-weight:800;margin-bottom:6px}}
  .subtitle{{text-align:center;color:var(--muted);font-size:.9rem;margin-bottom:20px}}
  .score-card{{background:var(--gold);border-radius:var(--radius);padding:20px 24px;
               margin-bottom:24px;display:flex;align-items:center;gap:20px;flex-wrap:wrap}}
  .score-num{{font-size:3rem;font-weight:900;color:#fff;line-height:1}}
  .score-label{{font-size:.8rem;color:rgba(255,255,255,.8);margin-top:2px}}
  .score-base{{font-size:.95rem;color:rgba(255,255,255,.9)}}
  .score-summary{{flex:1;min-width:200px;color:#fff;font-size:.92rem;line-height:1.6}}
  .tabs{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}}
  .tab{{padding:10px 18px;border-radius:30px;border:2px solid transparent;
        font-size:.88rem;font-weight:700;cursor:pointer;transition:all .2s;
        background:#fff;color:var(--text)}}
  .tab.active{{background:var(--gold);color:#fff;border-color:var(--gold)}}
  .tab:hover:not(.active){{border-color:var(--gold)}}
  .panel{{display:none}}.panel.active{{display:block}}
  .feat-card{{background:#fff;border-radius:var(--radius);padding:18px 20px;
              margin-bottom:14px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
  .feat-header{{display:flex;align-items:center;gap:12px;margin-bottom:10px;flex-wrap:wrap}}
  .badge{{padding:4px 12px;border-radius:20px;font-size:.78rem;font-weight:700;white-space:nowrap}}
  .badge-red{{background:var(--red);color:#fff}}
  .badge-yellow{{background:var(--yellow);color:#fff}}
  .badge-green{{background:var(--green);color:#fff}}
  .feat-name{{font-size:1.05rem;font-weight:800}}
  .feat-current{{font-size:.85rem;color:var(--muted);margin-bottom:8px;
                 border-left:3px solid #ddd;padding-left:10px}}
  .feat-suggest{{font-size:.88rem;line-height:1.65;background:#f9f9f9;
                 border-radius:8px;padding:10px 14px}}
  .shap-bar-wrap{{display:flex;align-items:center;gap:8px;margin:10px 0 8px}}
  .shap-bar-bg{{flex:1;height:6px;background:#eee;border-radius:3px;overflow:hidden}}
  .shap-bar-fill{{height:100%;border-radius:3px}}
  .shap-val{{font-size:.78rem;width:58px;text-align:right;font-variant-numeric:tabular-nums}}
  .dim-header{{background:var(--gold-bg);border:2px solid var(--gold-light);
               border-radius:var(--radius);padding:14px 18px;margin-bottom:16px;
               display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
  .dim-name-big{{font-size:1.1rem;font-weight:800}}
  .dim-total{{font-size:.82rem;color:var(--muted)}}
  .engine-badge{{display:inline-block;background:#1a1a2e;color:#ffd460;
                 font-size:.72rem;font-weight:700;padding:3px 10px;
                 border-radius:20px;margin-left:8px}}
  footer{{text-align:center;color:var(--muted);font-size:.78rem;margin-top:40px}}
</style>
</head>
<body>
<h1>🚀 專案募資潛力分析報告</h1>
<p class="subtitle">
  專案編號 #<strong>{idx}</strong>
  <span class="engine-badge">🤖 {engine}</span>
</p>
<div class="score-card">
  <div>
    <div class="score-num">{pred_prob}%</div>
    <div class="score-label">預測成功率</div>
  </div>
  <div>
    <div class="score-base">市場基礎勝率 <strong>{base_prob}%</strong></div>
    <div class="score-base">自身條件貢獻 <strong>{delta:+.1f}%</strong></div>
  </div>
  <div class="score-summary">{summary_text}</div>
</div>
<div class="tabs">{tab_html}</div>
{panels_html}
<footer>由 XGBoost + SHAP + {engine} 自動生成 ｜ 僅供參考，請結合實際市場判斷</footer>
<script>
document.querySelectorAll('.tab').forEach(t => {{
  t.addEventListener('click', () => {{
    document.querySelectorAll('.tab,.panel').forEach(e => e.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('panel-' + t.dataset.dim).classList.add('active');
  }});
}});
</script>
</body></html>"""

STATUS_BADGE = {
    "建議立即修改": '<span class="badge badge-red">建議立即修改</span>',
    "建議進行優化": '<span class="badge badge-yellow">建議進行優化</span>',
    "請繼續保持":   '<span class="badge badge-green">請繼續保持</span>',
}
SHAP_COLOR = {
    "建議立即修改": "#e74c3c",
    "建議進行優化": "#f39c12",
    "請繼續保持":   "#2ecc71",
}

def render_html(report: dict, project_idx: int, engine: str, out_path: str):
    overall = report["overall"]
    delta   = overall["predicted_prob"] - overall["base_prob"]

    tab_parts, panel_parts = [], []
    for i, dim in enumerate(report["dimensions"]):
        active = "active" if i == 0 else ""
        safe   = dim["name"].replace(" ", "_")
        tab_parts.append(
            f'<button class="tab {active}" data-dim="{safe}">{dim["name"]}</button>'
        )
        sign    = "📈" if dim["shap_total"] >= 0 else "📉"
        dim_hdr = (f'<div class="dim-header">'
                   f'<span class="dim-name-big">{sign} {dim["name"]}</span>'
                   f'<span class="dim-total">維度 SHAP 加總：{dim["shap_total"]:+.4f}</span>'
                   f'</div>')

        max_abs = max(abs(f["shap"]) for f in dim["features"]) or 1
        cards   = []
        for feat in dim["features"]:
            sv    = feat["shap"]
            pct   = abs(sv) / max_abs * 100
            color = SHAP_COLOR.get(feat.get("status",""), "#999")
            badge = STATUS_BADGE.get(feat.get("status",""), "")
            bar   = (f'<div class="shap-bar-wrap">'
                     f'<div class="shap-bar-bg"><div class="shap-bar-fill" '
                     f'style="width:{pct:.1f}%;background:{color}"></div></div>'
                     f'<span class="shap-val" style="color:{color}">{sv:+.4f}</span>'
                     f'</div>')
            cards.append(
                f'<div class="feat-card">'
                f'<div class="feat-header">{badge}'
                f'<span class="feat-name">{feat["name"]}</span></div>'
                f'{bar}'
                f'<div class="feat-current">原設定：{feat.get("current_desc","")}</div>'
                f'<div class="feat-suggest">修改建議：{feat.get("suggestion","")}</div>'
                f'</div>'
            )
        panel_parts.append(
            f'<div class="panel {active}" id="panel-{safe}">'
            f'{dim_hdr}{"".join(cards)}</div>'
        )

    html = HTML_TEMPLATE.format(
        idx         = project_idx,
        engine      = engine,
        pred_prob   = f"{overall['predicted_prob']:.1f}",
        base_prob   = f"{overall['base_prob']:.1f}",
        delta       = delta,
        summary_text= overall.get("summary", ""),
        tab_html    = "\n".join(tab_parts),
        panels_html = "\n".join(panel_parts),
    )
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(html)
    print(f"   HTML 報告已存至: {out_path}")

# ══════════════════════════════════════════════════════════════════
# 8. 主流程
# ══════════════════════════════════════════════════════════════════
def print_ollama_guide(model: str):
    print("""
╔══════════════════════════════════════════════════════════════╗
║            📦  Ollama 安裝與模型下載完整指引                 ║
╚══════════════════════════════════════════════════════════════╝

【Step 1】安裝 Ollama
  Windows / Mac：
    前往 https://ollama.com/download 下載安裝程式，一路 Next 即可

  Linux：
    curl -fsSL https://ollama.com/install.sh | sh

【Step 2】啟動 Ollama 服務
  Windows：安裝後會自動在背景執行（系統匣會有 Ollama 圖示）
  Mac：    開啟 Ollama.app，或執行：ollama serve
  Linux：  ollama serve   （或 systemctl start ollama）

  確認服務正在運行：
    curl http://localhost:11434/api/tags

【Step 3】下載推薦模型（擇一即可）

  ┌─────────────────┬──────────┬──────────────────────────────────┐
  │ 模型名稱         │ 大小     │ 說明                             │
  ├─────────────────┼──────────┼──────────────────────────────────┤
  │ gemma3:4b       │ ~3 GB    │ 推薦入門，速度快，中文尚可       │
  │ llama3.2        │ ~2 GB    │ Meta 出品，指令遵循能力強        │
  │ qwen2.5:7b      │ ~5 GB    │ 阿里巴巴，中文能力最佳           │
  │ mistral         │ ~4 GB    │ 歐系模型，英文推理能力強         │
  │ llama3.1:8b     │ ~5 GB    │ 較大，建議 16GB RAM 以上         │
  └─────────────────┴──────────┴──────────────────────────────────┘

  下載命令（以 gemma3:4b 為例）：
    ollama pull gemma3:4b

  查看已下載的模型：
    ollama list

【Step 4】執行本程式
  # 使用預設模型（gemma3:4b）
  python xgboost_shap_gemini.py

  # 指定其他模型
  python xgboost_shap_gemini.py --model qwen2.5:7b

  # 指定分析的專案編號
  python xgboost_shap_gemini.py --model llama3.2 --project 10

【常見問題】
  Q：輸出不是合法 JSON？
  A：換用指令遵循能力更好的模型：ollama pull llama3.2

  Q：生成太慢？
  A：改用更小的模型，或確認 GPU 是否啟用：
     nvidia-smi   # 查看 GPU；Ollama 會自動偵測並使用 GPU

  Q：RAM 不夠？
  A：4B 模型需約 4–6 GB RAM；若不足，改用 gemma3:1b（~1 GB）
""")
    print(f"  目前指定模型：{model}")
    print(f"  請先執行：ollama pull {model}")
    print()


def main():
    parser = argparse.ArgumentParser(description="XGBoost + SHAP + LLM 募資分析")
    parser.add_argument("--project", type=int,   default=5,
                        help="要分析的專案索引（預設 5）")
    parser.add_argument("--backend", type=str,   default="ollama",
                        choices=["ollama", "gemini"],
                        help="LLM 後端：ollama（預設）或 gemini")
    parser.add_argument("--model",   type=str,   default="gemma3:4b",
                        help="Ollama 模型名稱（預設 gemma3:4b）")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="Ollama 服務位址（預設 http://localhost:11434）")
    args = parser.parse_args()

    global OLLAMA_URL
    OLLAMA_URL = args.ollama_url

    PROJECT_IDX = args.project
    OUT_DIR     = "shap_output"
    API_KEY     = os.environ.get("GEMINI_API_KEY", "")

    print("=" * 60)
    print("  XGBoost → SHAP → LLM 逐特徵解釋 完整流程")
    print(f"  後端: {args.backend.upper()}" +
          (f"  模型: {args.model}" if args.backend == "ollama" else ""))
    print("=" * 60)

    # Step 0–2：資料、訓練、SHAP
    print("▶ [Step 0] 生成模擬資料 ...")
    X, y = make_dataset()
    print(f"   樣本數: {len(X)}, 成功率: {y.mean():.1%}")

    model_xgb = train_model(X, y)
    _, sv     = run_shap(model_xgb, X)

    print("▶ [Step 3] SHAP 圖表 + 摘要 ...")
    os.makedirs(OUT_DIR, exist_ok=True)
    save_shap_plots(sv, X, PROJECT_IDX, OUT_DIR)
    summary = build_summary(sv, X, PROJECT_IDX)
    json.dump(summary,
              open(f"{OUT_DIR}/project_{PROJECT_IDX}_shap.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # Step 4：LLM 解釋
    engine_label = ""
    report       = None

    if args.backend == "ollama":
        try:
            report       = call_ollama_structured(summary, PROJECT_IDX, args.model)
            engine_label = f"Ollama/{args.model}"
        except RuntimeError as e:
            print(f"\n❌ Ollama 失敗：{e}\n")
            print_ollama_guide(args.model)
            sys.exit(1)

    elif args.backend == "gemini":
        if not API_KEY:
            print("❌ 使用 Gemini 後端需設定環境變數 GEMINI_API_KEY")
            print("   Windows: set GEMINI_API_KEY=你的金鑰")
            print("   Mac/Linux: export GEMINI_API_KEY='你的金鑰'")
            sys.exit(1)
        try:
            report       = call_gemini_structured(summary, PROJECT_IDX, API_KEY)
            engine_label = "Gemini"
        except RuntimeError as e:
            print(f"\n❌ Gemini 失敗：{e}")
            sys.exit(1)

    # Step 5：HTML 報告
    json.dump(report,
              open(f"{OUT_DIR}/project_{PROJECT_IDX}_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("▶ [Step 5] 產生 HTML 互動報告 ...")
    html_path = f"{OUT_DIR}/project_{PROJECT_IDX}_report.html"
    render_html(report, PROJECT_IDX, engine_label, html_path)

    print(f"\n✅ 完成！開啟以下檔案查看報告：")
    print(f"   {html_path}")


if __name__ == "__main__":
    main()