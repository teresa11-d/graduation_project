"""
feature_importance_analysis.py
================================
募資平台特徵重要性分析 (支援動態多 CSV 檔案合併)
- 相關性熱度圖 (Heatmap)
- Random Forest 特徵重要性橫條圖
- XGBoost 特徵重要性橫條圖
- RF vs XGBoost 比較圖

備註：
trust_word_density (信任詞密度) 與 risk_word_count (風險誇大詞數) 
為待開發特徵，目前以註解形式保留於程式碼中。

使用方式:
  # 模擬資料測試
  python feature_importance_analysis.py --mode simulate

  # 讀取多個真實 CSV 檔案 (接續檔名即可，支援無限多個)
  python feature_importance_analysis.py --mode csv --csv_files projects_summary.csv my_result.csv extra_features.csv
"""

import argparse
import os
import urllib.request
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
from matplotlib import rcParams
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ── 字型設定 ──────────────────────────────────────────────────────────────────
def setup_chinese_font():
    preferred_fonts = [
        "PingFang TC", "Noto Sans CJK TC", "Microsoft JhengHei", 
        "Taipei Sans TC Beta", "Arial Unicode MS", "SimHei", 
        "Heiti TC", "WenQuanYi Micro Hei"
    ]
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in preferred_fonts:
        if font in available_fonts: return font
    for font in available_fonts:
        if any(keyword in font.lower() for keyword in ['cjk tc', 'jhenghei', 'pingfang', 'hei', 'ming', 'cjk jp']):
            return font
    font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
    font_path = "NotoSansCJKtc-Regular.otf"
    if not os.path.exists(font_path):
        try: urllib.request.urlretrieve(font_url, font_path)
        except Exception: return "sans-serif"
    try:
        fm.fontManager.addfont(font_path)
        return fm.FontProperties(fname=font_path).get_name()
    except Exception: return "sans-serif"

CJK_FONT = setup_chinese_font()
rcParams["font.family"] = CJK_FONT
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 150
rcParams["savefig.bbox"] = "tight"

# ── 顏色主題 ──────────────────────────────────────────────────────────────────
COLOR_RF  = "#3B6D11"
COLOR_XGB = "#185FA5"
COLOR_NEG = "#A32D2D"
COLOR_POS = "#0F6E56"
BG        = "#FAFAF8"
GRID      = "#E8E6DF"

# ── 特徵顯示名稱（依照 PDF 要求增刪）────────────────────────────────────────
FEATURE_LABELS = {
    # 原始特徵
    "target_amount_log":        "目標金額 (log)",
    "duration_days":            "募資天數",
    "price_tier_count":         "回饋方案層級數",
    "has_social_links":         "社群連結數",
    "story_ratio":              "故事段落比例",
    "spec_ratio":               "規格段落比例",
    "risk_ratio":               "風險說明比例",
    
    # 待開發的文字特徵 (目前以註解呈現)
    # "trust_word_density":       "信任詞密度",
    # "risk_word_count":          "風險誇大詞數",

    "img_text_ratio":           "圖文密度比",
    "has_video":                "有主影片",
    "video_duration":           "影片長度 (秒)",
    "category_recent_success":  "同類近期成功率",
    "market_saturation":        "市場飽和度",
    "google_trend_slope":       "Google 趨勢斜率",
    "blockbuster_similarity":   "爆款相似度",
    
    # 從 CSV 解析的新增特徵
    "折扣層數":                 "折扣層數",
    "FAQ總題數":                "FAQ總題數",
    "FAQ更新頻率":              "FAQ更新頻率",
    "price_min":                "最低方案價格",
    "price_max":                "最高方案價格",
    "price_avg":                "平均方案價格",
    "main_cat_encoded":         "主分類(編碼)",
    "sub_cat_encoded":          "次分類(編碼)",
    "word_count":               "文案總字數",
    "image_count":              "圖片數量",
    "video_count":              "影片數量",
    "media_per_100_words":      "多媒體密度(每百字)",
}

FEATURE_COLS = list(FEATURE_LABELS.keys())
TARGET_COL   = "success_label"
OUTPUT_DIR   = "output_plots"


# ══════════════════════════════════════════════════════════════════════════════
# 1. 資料準備與特徵工程
# ══════════════════════════════════════════════════════════════════════════════

def process_price_list(price_str):
    if pd.isna(price_str) or not isinstance(price_str, str):
        return pd.Series({"price_min": np.nan, "price_max": np.nan, "price_avg": np.nan})
    prices = [float(p.strip()) for p in price_str.split('|') if p.strip().replace('.','',1).isdigit()]
    if not prices:
        return pd.Series({"price_min": np.nan, "price_max": np.nan, "price_avg": np.nan})
    return pd.Series({"price_min": min(prices), "price_max": max(prices), "price_avg": np.mean(prices)})

def simulate_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """產生模擬資料"""
    rng = np.random.default_rng(seed)
    n_pos = int(n * 0.4)
    n_neg = n - n_pos

    def make_rows(size, success):
        s = int(success)
        return {
            "target_amount_log": rng.normal(11.5 - 1.2 * (1 - s), 1.2, size),
            "duration_days":     rng.integers(14, 90, size).astype(float),
            "price_tier_count":  np.clip(rng.normal(4 + s, 1.5, size), 1, 12),
            "has_social_links":  rng.binomial(5, 0.3 + 0.3 * s, size).astype(float),
            "story_ratio":       np.clip(rng.normal(0.3 + 0.1 * s, 0.1, size), 0, 1),
            "spec_ratio":        np.clip(rng.normal(0.25 + 0.08 * s, 0.1, size), 0, 1),
            "risk_ratio":        np.clip(rng.normal(0.05 + 0.05 * s, 0.03, size), 0, 0.3),
            
            # "trust_word_density": np.clip(rng.normal(0.8 + 0.6 * s, 0.4, size), 0, 5),
            # "risk_word_count":   np.clip(rng.normal(3 - 2 * s, 1.5, size), 0, 15),

            "img_text_ratio":    np.clip(rng.normal(0.15 + 0.08 * s, 0.08, size), 0, 1),
            "has_video":         rng.binomial(1, 0.5 + 0.3 * s, size).astype(float),
            "video_duration":    np.where(rng.binomial(1, 0.5 + 0.3 * s, size), rng.normal(130 + 20 * s, 40, size), np.nan),
            "category_recent_success": np.clip(rng.normal(0.35 + 0.2 * s, 0.12, size), 0, 1),
            "market_saturation": rng.integers(10, 200, size).astype(float),
            "google_trend_slope": rng.normal(0.5 * s - 0.1, 0.3, size),
            "blockbuster_similarity": np.clip(rng.normal(0.4 + 0.2 * s, 0.15, size), 0, 1),
            
            "折扣層數": np.clip(rng.normal(5 + 2 * s, 2, size), 1, 15).astype(int),
            "FAQ總題數": np.clip(rng.normal(10 + 5 * s, 8, size), 0, 50).astype(int),
            "FAQ更新頻率": np.clip(rng.normal(0.3 + 0.2 * s, 0.2, size), 0, 1),
            "price_min": np.clip(rng.normal(800 - 200 * s, 300, size), 100, 5000),
            "price_max": np.clip(rng.normal(5000 + 2000 * s, 2000, size), 1000, 20000),
            "price_avg": np.clip(rng.normal(2500 + 500 * s, 1000, size), 500, 10000),
            "main_cat_encoded": rng.integers(0, 5, size),
            "sub_cat_encoded": rng.integers(0, 15, size),
            "word_count": np.clip(rng.normal(1500 + 800 * s, 600, size), 300, 8000).astype(int),
            "image_count": np.clip(rng.normal(30 + 15 * s, 15, size), 5, 100).astype(int),
            "video_count": np.clip(rng.normal(0.5 + 0.8 * s, 1.0, size), 0, 5).astype(int),
            
            TARGET_COL: np.full(size, s),
        }

    df = pd.concat([
        pd.DataFrame(make_rows(n_pos, 1)),
        pd.DataFrame(make_rows(n_neg, 0)),
    ], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    df["media_per_100_words"] = (df["image_count"] + df["video_count"]) / (df["word_count"] / 100)
    return df

def load_multiple_csvs(csv_paths: list) -> pd.DataFrame:
    """自動尋找共通 ID 並合併多個 CSV 檔案"""
    if not csv_paths:
        raise ValueError("請提供至少一個 CSV 檔案路徑")

    merged_df = None
    # 常見的專案識別碼欄位名稱，程式會自動尋找這些欄位來進行對齊合併
    id_candidates = ['專案編號', 'project', 'project_id', 'id', '專案ID', 'Project ID']

    for path in csv_paths:
        print(f"  > 正在讀取並合併: {path}")
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            print(f"    [錯誤] 無法讀取 {path}: {e}")
            continue

        # 尋找是否包含已知 ID 欄位
        found_id_col = None
        for col in id_candidates:
            if col in df.columns:
                found_id_col = col
                break

        if found_id_col:
            # 統一改名為標準內部 ID 以便合併
            df = df.rename(columns={found_id_col: '_common_project_id_'})
            df = df.set_index('_common_project_id_')

        if merged_df is None:
            merged_df = df
        else:
            if df.index.name == '_common_project_id_' and merged_df.index.name == '_common_project_id_':
                # 如果兩個表都有 ID，用 Outer Join 自動對齊
                # 排除重複出現的非 ID 欄位，以已存在的資料表為主
                overlapping = set(merged_df.columns).intersection(set(df.columns))
                if overlapping:
                    df = df.drop(columns=list(overlapping))
                merged_df = merged_df.join(df, how='outer')
            else:
                # 若找不到共通 ID，退回暴力橫向拼接 (並給出警告)
                print(f"    [警告] 在此檔案找不到共通的專案 ID 欄位，將直接並排合併，請確認資料順序一致！")
                merged_df = pd.concat([merged_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    # 處理特徵工程
    if merged_df is not None:
        if merged_df.index.name == '_common_project_id_':
            merged_df = merged_df.reset_index(drop=True)

        if '募資狀態' in merged_df.columns: 
            merged_df[TARGET_COL] = (merged_df['募資狀態'] == '成功').astype(int)
        else: 
            merged_df[TARGET_COL] = 0
            
        if '方案價格列表' in merged_df.columns:
            price_features = merged_df['方案價格列表'].apply(process_price_list)
            merged_df = pd.concat([merged_df, price_features], axis=1)
            
        for new_col, ori_col in [("main_cat_encoded", "主分類"), ("sub_cat_encoded", "次分類")]:
            if ori_col in merged_df.columns: 
                merged_df[new_col] = merged_df[ori_col].astype('category').cat.codes
            else: 
                merged_df[new_col] = np.nan

        # 填補 CSV 中可能缺失的待開發特徵，避免後續篩選時報錯
        for col in FEATURE_COLS:
            if col not in merged_df.columns:
                merged_df[col] = np.nan
                
        merged_df = merged_df[FEATURE_COLS + [TARGET_COL]].copy()
        merged_df = merged_df.dropna(subset=[TARGET_COL])
        merged_df[TARGET_COL] = merged_df[TARGET_COL].astype(int)

    return merged_df

def preprocess(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_COLS)
    return X_imp, y

# ══════════════════════════════════════════════════════════════════════════════
# 2. 模型訓練與繪圖
# ══════════════════════════════════════════════════════════════════════════════

def train_rf(X_train, y_train, seed=42):
    model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_xgb(X_train, y_train, seed=42):
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, use_label_encoder=False, eval_metric="logloss", random_state=seed, verbosity=0)
        model.fit(X_train, y_train)
        return model
    except ImportError:
        print("[警告] xgboost 未安裝。")
        return None

def get_importance_df(model, feature_names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
        "label": [FEATURE_LABELS.get(f, f) for f in feature_names],
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df

def _setup_ax(ax, title: str, xlabel: str = ""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, loc="left")
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, color="#555")
    ax.tick_params(labelsize=9, colors="#444")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)

def plot_heatmap(df: pd.DataFrame, save_path: str):
    cols = FEATURE_COLS + [TARGET_COL]
    # 如果全都是 NaN (例如還沒資料的特徵)，corr() 會出錯，預先篩選掉
    valid_cols = [c for c in cols if c in df.columns and df[c].notna().any()]
    corr = df[valid_cols].corr()
    n = len(corr)
    
    fig, ax = plt.subplots(figsize=(18, 16))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rw_g", [COLOR_NEG, "#FFFFFF", COLOR_POS], N=256)
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Pearson r", fontsize=11)
    
    labels = [FEATURE_LABELS.get(c, c) for c in corr.columns]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            if pd.notna(val) and abs(val) > 0.15 and i != j:
                color = "white" if abs(val) > 0.6 else "#222"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
                
    if TARGET_COL in corr.columns:
        idx = list(corr.columns).index(TARGET_COL)
        for spine in ["top", "right", "bottom", "left"]: ax.spines[spine].set_visible(False)
        rect = plt.Rectangle((idx - 0.5, -0.5), 1, n, linewidth=2, edgecolor="#854F0B", facecolor="none")
        ax.add_patch(rect)
    ax.set_title("特徵相關性熱度圖（橘框 = 成功標籤）", fontsize=15, fontweight="bold", pad=16, loc="left")
    plt.tight_layout()
    plt.savefig(save_path, facecolor=BG)
    plt.close()

def plot_bar_importance(imp_df: pd.DataFrame, title: str, color: str, save_path: str, top_n: int = 20):
    data = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.42)))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax, title, xlabel="Feature Importance（MDI）")
    bars = ax.barh(data["label"], data["importance"], color=color, alpha=0.85, height=0.6, edgecolor="none")
    for bar, val, rank in zip(bars, data["importance"], data["rank"].iloc[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=8, color="#333")
        ax.text(-0.002, bar.get_y() + bar.get_height() / 2, f"#{rank}", va="center", ha="right", fontsize=7.5, color="#999", fontweight="bold")
    ax.set_xlim(-0.012, data["importance"].max() * 1.18)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    plt.savefig(save_path, facecolor=BG)
    plt.close()

def plot_comparison(rf_df: pd.DataFrame, xgb_df: pd.DataFrame, save_path: str, top_n: int = 25):
    features = rf_df["feature"].head(top_n).tolist()
    labels   = rf_df["label"].head(top_n).tolist()
    xgb_map = dict(zip(xgb_df["feature"], xgb_df["importance"]))
    rf_vals  = rf_df["importance"].head(top_n).values
    xgb_vals = np.array([xgb_map.get(f, 0.0) for f in features])
    
    rf_norm  = rf_vals / rf_vals.max() if rf_vals.max() > 0 else rf_vals
    xgb_norm = xgb_vals / xgb_vals.max() if xgb_vals.max() > 0 else xgb_vals
    y = np.arange(len(features))
    bar_h = 0.36
    
    fig, ax = plt.subplots(figsize=(11, max(7, len(features) * 0.45)))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax, "RF vs XGBoost 特徵重要性對比 (Top 25)", xlabel="正規化重要性")
    ax.barh(y + bar_h / 2, rf_norm, height=bar_h, color=COLOR_RF, alpha=0.82, label="Random Forest", edgecolor="none")
    ax.barh(y - bar_h / 2, xgb_norm, height=bar_h, color=COLOR_XGB, alpha=0.82, label="XGBoost", edgecolor="none")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.18)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.6)
    for i, (r, x) in enumerate(zip(rf_norm, xgb_norm)):
        diff = abs(r - x)
        if diff > 0.2:
            ax.text(max(r, x) + 0.02, i, f"Δ{diff:.2f}", va="center", fontsize=7.5, color="#854F0B", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, facecolor=BG)
    plt.close()

def plot_success_corr_bar(df: pd.DataFrame, save_path: str):
    cols = [c for c in FEATURE_COLS if c in df.columns and df[c].notna().any()]
    corr_vals = df[cols].corrwith(df[TARGET_COL]).sort_values()
    labels = [FEATURE_LABELS.get(c, c) for c in corr_vals.index]
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in corr_vals.values]
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(cols) * 0.35)))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax, "各特徵與募資成功的相關係數（Pearson r）", xlabel="Pearson r")
    ax.barh(labels, corr_vals.values, color=colors, alpha=0.85, height=0.6, edgecolor="none")
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="-")
    
    for i, val in enumerate(corr_vals.values):
        offset = 0.005 if val >= 0 else -0.005
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, f"{val:.3f}", va="center", ha=ha, fontsize=8, color="#333")
    
    x_min, x_max = corr_vals.min(), corr_vals.max()
    if pd.notna(x_min) and pd.notna(x_max) and x_min != x_max:
        ax.set_xlim(x_min * 1.3, x_max * 1.3)
        
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=COLOR_POS, alpha=0.85, label="正相關（有助成功）"),
        Patch(color=COLOR_NEG, alpha=0.85, label="負相關（不利成功）"),
    ], fontsize=9, loc="lower right", framealpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, facecolor=BG)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 4. 主流程
# ══════════════════════════════════════════════════════════════════════════════

def run(mode: str = "simulate", csv_paths: list = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"► 成功設定圖表字型: {CJK_FONT}")

    if mode == "simulate":
        print("► 產生綜合模擬資料（500 筆）...")
        df = simulate_data(n=500)
        df.to_csv(os.path.join(OUTPUT_DIR, "simulated_full_features.csv"), index=False, encoding="utf-8-sig")
        print(f"  模擬資料已儲存至 {OUTPUT_DIR}/simulated_full_features.csv")
    else:
        if not csv_paths:
            print("[錯誤] 未提供 CSV 檔案路徑。請使用 --csv_files 參數。")
            return
        print(f"► 開始處理多重 CSV 合併作業...")
        df = load_multiple_csvs(csv_paths)

    print(f"  合併後資料維度: {df.shape}，包含有效特徵數: {len(FEATURE_COLS)}，成功率: {df[TARGET_COL].mean():.1%}")

    X, y = preprocess(df)
    
    if len(X) < 10:
        print("\n[提示] 資料筆數過少，全數作為訓練集使用。")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("\n► 繪製相關性熱度圖...")
    plot_heatmap(df, os.path.join(OUTPUT_DIR, "01_heatmap.png"))
    print("\n► 繪製成功標籤相關係數圖...")
    plot_success_corr_bar(df, os.path.join(OUTPUT_DIR, "02_success_corr.png"))

    print("\n► 訓練 Random Forest...")
    rf = train_rf(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    rf_imp = get_importance_df(rf, FEATURE_COLS)
    plot_bar_importance(rf_imp, f"Random Forest 特徵重要性（Accuracy {rf_acc:.3f}）", COLOR_RF, os.path.join(OUTPUT_DIR, "03_rf_importance.png"))

    print("\n► 訓練 XGBoost...")
    xgb = train_xgb(X_train, y_train)
    if xgb is not None:
        xgb_acc = xgb.score(X_test, y_test)
        xgb_imp = get_importance_df(xgb, FEATURE_COLS)
        plot_bar_importance(xgb_imp, f"XGBoost 特徵重要性（Accuracy {xgb_acc:.3f}）", COLOR_XGB, os.path.join(OUTPUT_DIR, "04_xgb_importance.png"))
        plot_comparison(rf_imp, xgb_imp, os.path.join(OUTPUT_DIR, "05_comparison.png"))

        print("\n" + "=" * 65)
        print("📊 綜合特徵重要性摘要（Top 10）")
        print("=" * 65)
        print(f"{'排名':<5} {'特徵名稱':<25} {'RF 權重':>10} {'XGBoost 權重':>12}")
        print("-" * 65)
        xgb_map = dict(zip(xgb_imp["feature"], xgb_imp["importance"]))
        for _, row in rf_imp.head(10).iterrows():
            xgb_val = xgb_map.get(row["feature"], 0.0)
            print(f"#{row['rank']:<4} {row['label']:<25} {row['importance']:>10.4f} {xgb_val:>12.4f}")
        print("=" * 65)

    print(f"\n✅ 分析完成！全部圖表已儲存至 ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="募資平台特徵重要性分析")
    parser.add_argument("--mode", choices=["simulate", "csv"], default="simulate", help="simulate=模擬；csv=讀取真實 CSV")
    
    # 這裡改成 nargs='+' 允許傳入 1~N 個檔案路徑
    parser.add_argument("--csv_files", nargs='+', default=["projects_summary.csv", "my_result.csv"], 
                        help="傳入多個 CSV 檔案，以空格分隔 (例: a.csv b.csv c.csv)")
    
    args = parser.parse_args()
    run(mode=args.mode, csv_paths=args.csv_files)