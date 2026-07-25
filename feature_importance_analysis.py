"""
feature_importance_analysis_v3.py
===================================
募資平台特徵重要性分析 — 對照實際 CSV 欄位完整重寫合併邏輯
(並整合完整視覺化圖表：Heatmap、相關係數圖、重要性比較圖)

合併策略：
  主表 = projects_summary（短碼），提供標籤與基本特徵
  其餘所有檔案為長碼，取底線前綴（ES1_rogerems → ES1）對應主表 project_id
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib import rcParams
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")

# ── 字型設定 ──────────────────────────────────────────────────────────────────
def setup_chinese_font():
    preferred = ["Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "SimHei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in preferred:
        if font in available:
            return font
    return "sans-serif"

rcParams["font.family"]        = setup_chinese_font()
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"]         = 150

COLOR_RF  = "#3B6D11"
COLOR_XGB = "#185FA5"
COLOR_NEG = "#A32D2D"
COLOR_POS = "#0F6E56"
BG        = "#FAFAF8"
GRID      = "#E8E6DF"

# ── 特徵欄位定義（對應實際欄位名稱）────────────────────────────────────────────
FEATURE_LABELS = {
    # projects_summary
    "price_tier_count":        "回饋方案層級數",
    "折扣層數":                  "折扣層數",
    "FAQ總題數":                 "FAQ總題數",
    "FAQ更新頻率":               "FAQ更新頻率",
    "price_min":               "最低方案價格",
    "price_max":               "最高方案價格",
    "price_avg":               "平均方案價格",
    "main_cat_encoded":        "主分類(編碼)",
    "sub_cat_encoded":         "次分類(編碼)",
    # image_text_ratio_result
    "word_count":              "文案總字數",
    "image_count":             "圖片數量",
    "video_count":             "影片數量",
    "media_per_100_words":     "多媒體密度(每百字)",
    # video_features_result
    "has_video":               "有主影片",
    "video_duration":          "影片長度(秒)",
    "usage_scene_ratio":       "實際操作情境佔比",
    # feat_out_text（尚未產生，佔位）
    "feat_text_story_ratio":   "故事段落比例",
    "feat_text_spec_ratio":    "規格段落比例",
    "feat_text_risk_ratio":    "風險說明比例",
    "feat_has_social_link":    "社群連結數",
    "feat_adverb_density":     "副詞密度",
    "feat_adj_nv_ratio":       "形/(名+動)比例",
    "feat_punct_intensity":    "標點符號強度",
    "feat_numeral_density":    "數字密度",
    "feat_entity_density":     "實體密度",
    "feat_avg_sentence_len":   "平均句長",
    "feat_type_token_ratio":   "詞彙多樣性(TTR)",
    # PMF
    "category_recent_success": "同類近期成功率",
    "market_saturation":       "市場飽和度",
    "google_trend_slope":      "Google趨勢斜率",
    "blockbuster_similarity":  "爆款相似度",
}

FEATURE_COLS = list(FEATURE_LABELS.keys())
TARGET_COL   = "success_label"
OUTPUT_DIR   = "output_plots"


# ══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════════════

def read_csv(path: str, name: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        print(f"  ⚠️  [{name}] 找不到檔案：{path}")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  📄 [{name}] {df.shape[0]} 列 × {df.shape[1]} 欄  |  欄位：{df.columns.tolist()}")
    return df

def to_prefix(series: pd.Series) -> pd.Series:
    """長碼 ES1_rogerems → 短碼前綴 ES1"""
    return series.astype(str).str.strip().str.split("_").str[0]

def left_join_by_prefix(df_main: pd.DataFrame,
                        df_sub: pd.DataFrame,
                        sub_id_col: str,
                        name: str,
                        exclude_cols: list = None) -> pd.DataFrame:
    if df_sub.empty:
        return df_main

    df_sub = df_sub.copy()
    df_sub[sub_id_col] = df_sub[sub_id_col].astype(str).str.strip()
    df_sub["_prefix"]  = to_prefix(df_sub[sub_id_col])

    drop = list(set((exclude_cols or []) + [sub_id_col]))
    df_sub = df_sub.drop(columns=[c for c in drop if c in df_sub.columns], errors="ignore")

    if df_sub["_prefix"].duplicated().any():
        num_cols = df_sub.select_dtypes(include="number").columns.tolist()
        df_sub   = df_sub.groupby("_prefix")[num_cols].mean().reset_index()

    df_sub = df_sub.rename(columns={"_prefix": "project_id"})
    merged  = pd.merge(df_main, df_sub, on="project_id", how="left")

    new_cols = [c for c in merged.columns if c not in df_main.columns]
    if new_cols:
        matched = merged[new_cols[0]].notna().sum()
        print(f"  ✅ [{name}] 對齊 {matched} / {len(merged)} 筆"
              f"（{len(merged)-matched} 筆無此模態資料）")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 主資料載入與前處理
# ══════════════════════════════════════════════════════════════════════════════

def load_data(projects_path, img_ratio_path, video_path,
              pmf_path, feat_text_path=None) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("📥  多模態資料合併")
    print("=" * 60)

    df_proj = read_csv(projects_path, "projects_summary")
    if df_proj.empty:
        raise FileNotFoundError("projects_summary.csv 是必要檔案，請確認路徑。")

    df_proj["project_id"] = df_proj["專案編號"].astype(str).str.strip()
    df_proj[TARGET_COL] = (df_proj["募資狀態"] == "成功").astype(int)
    n_s, n_f = df_proj[TARGET_COL].sum(), (df_proj[TARGET_COL] == 0).sum()
    print(f"\n  🎯 標籤：成功 {n_s} / 失敗 {n_f}（來源：projects_summary 募資狀態）")

    df_proj["main_cat_encoded"] = df_proj["主分類"].astype("category").cat.codes
    df_proj["sub_cat_encoded"]  = df_proj["次分類"].astype("category").cat.codes

    if "方案價格列表" in df_proj.columns:
        def parse_prices(val):
            try:
                prices = [float(x.strip()) for x in str(val).split("|")
                          if x.strip().replace(".", "").replace("-", "").isdigit()]
                if prices:
                    return len(prices), min(prices), max(prices), sum(prices) / len(prices)
            except Exception:
                pass
            return (np.nan,) * 4

        parsed = df_proj["方案價格列表"].apply(parse_prices)
        df_proj["price_tier_count"] = parsed.apply(lambda x: x[0])
        df_proj["price_min"]        = parsed.apply(lambda x: x[1])
        df_proj["price_max"]        = parsed.apply(lambda x: x[2])
        df_proj["price_avg"]        = parsed.apply(lambda x: x[3])

    base_cols = ["project_id", TARGET_COL,
                 "main_cat_encoded", "sub_cat_encoded",
                 "折扣層數", "FAQ總題數", "FAQ更新頻率",
                 "price_tier_count", "price_min", "price_max", "price_avg"]
    base_cols = [c for c in base_cols if c in df_proj.columns]
    df = df_proj[base_cols].copy()
    print(f"  📊 主表建立：{len(df)} 筆專案\n")

    df_img = read_csv(img_ratio_path, "image_text_ratio")
    if not df_img.empty:
        keep = ["project", "word_count", "image_count", "video_count", "media_per_100_words"]
        keep = [c for c in keep if c in df_img.columns]
        df = left_join_by_prefix(df, df_img[keep], "project", "image_text_ratio")

    df_vid = read_csv(video_path, "video_features")
    if not df_vid.empty and "project" in df_vid.columns:
        vid_keep = ["project", "total_duration", "usage_scene_ratio"]
        vid_keep = [c for c in vid_keep if c in df_vid.columns]
        df_vid   = df_vid[vid_keep].copy()
        df_vid["project"] = df_vid["project"].astype(str).str.strip()
        df_vid["_prefix"] = to_prefix(df_vid["project"])

        if 'total_duration' in df_vid.columns:
            df_vid['total_duration'] = pd.to_numeric(df_vid['total_duration'], errors='coerce')
        if 'usage_scene_ratio' in df_vid.columns:
            df_vid['usage_scene_ratio'] = pd.to_numeric(df_vid['usage_scene_ratio'], errors='coerce')
        agg = {}
        if "total_duration"    in df_vid.columns: agg["total_duration"]    = "sum"
        if "usage_scene_ratio" in df_vid.columns: agg["usage_scene_ratio"] = "mean"
        df_vid_agg = df_vid.groupby("_prefix").agg(agg).reset_index()
        df_vid_agg = df_vid_agg.rename(columns={"_prefix": "project_id", "total_duration": "video_duration"})
        df_vid_agg["has_video"] = 1

        df = pd.merge(df, df_vid_agg, on="project_id", how="left")
        df["has_video"] = df["has_video"].fillna(0).astype(int)

    df_pmf = read_csv(pmf_path, "PMF") if pmf_path else pd.DataFrame()
    if not df_pmf.empty and "project_id" in df_pmf.columns:
        pmf_keep = ["project_id", "category_recent_success", "market_saturation", "google_trend_slope", "blockbuster_similarity"]
        pmf_keep = [c for c in pmf_keep if c in df_pmf.columns]
        df = left_join_by_prefix(df, df_pmf[pmf_keep], "project_id", "PMF")

    if feat_text_path and os.path.exists(feat_text_path):
        df_feat = read_csv(feat_text_path, "feat_out_text")
        if not df_feat.empty and "project_id" in df_feat.columns:
            num_cols = df_feat.select_dtypes(include="number").columns.tolist()
            df = left_join_by_prefix(df, df_feat[["project_id"] + num_cols], "project_id", "feat_out_text")
    else:
        print("  ⏳ [feat_out_text] 尚未產生，文本特徵以 NaN 佔位。")

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[["project_id"] + FEATURE_COLS + [TARGET_COL]].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df

def preprocess(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    empty = X.columns[X.isna().all()].tolist()
    if empty:
        X[empty] = 0
    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=FEATURE_COLS,
    )
    return X_imp, y


# ══════════════════════════════════════════════════════════════════════════════
# 模型訓練
# ══════════════════════════════════════════════════════════════════════════════

def train_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

def train_xgb(X_train, y_train):
    try:
        from xgboost import XGBClassifier
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=neg / max(pos, 1),
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train)
        return model
    except ImportError:
        print("  xgboost 未安裝，略過。")
        return None

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    try: auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except: auc = float("nan")
    print(f"  [{name}]  Accuracy={acc:.3f}  F1(macro)={f1:.3f}  AUC-ROC={auc:.3f}")
    return {"f1": f1, "auc": auc}

def get_xgb_gain(model, feature_names):
    raw = model.get_booster().get_score(importance_type="gain")
    imp = np.array([raw.get(f, raw.get(f"f{i}", 0)) for i, f in enumerate(feature_names)])
    return imp / imp.sum() if imp.sum() > 0 else imp


# ══════════════════════════════════════════════════════════════════════════════
# 綜合繪圖功能
# ══════════════════════════════════════════════════════════════════════════════

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
    corr = df.corr().fillna(0)
    n = len(corr)
    
    fig, ax = plt.subplots(figsize=(18, 16))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    
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
            if abs(val) > 0.15 and i != j:
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
    print(f"  💾 {save_path}")

def plot_success_corr_bar(df: pd.DataFrame, save_path: str):
    cols = [c for c in FEATURE_COLS if c in df.columns]
    corr_vals = df[cols].corrwith(df[TARGET_COL]).fillna(0).sort_values()
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
        
    ax.legend(handles=[
        Patch(color=COLOR_POS, alpha=0.85, label="正相關（有助成功）"),
        Patch(color=COLOR_NEG, alpha=0.85, label="負相關（不利成功）"),
    ], fontsize=9, loc="lower right", framealpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, facecolor=BG)
    plt.close()
    print(f"  💾 {save_path}")

def plot_importance(importances, feature_names, title, color, save_path, top_n=25):
    df_imp = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
        "label":      [FEATURE_LABELS.get(f, f) for f in feature_names],
    }).sort_values("importance", ascending=False)

    data = df_imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.44)))
    fig.patch.set_facecolor(BG)
    _setup_ax(ax, title, xlabel="Feature Importance")
    bars = ax.barh(data["label"], data["importance"], color=color, alpha=0.85, edgecolor="none")
    for bar, val in zip(bars, data["importance"]):
        ax.text(val, bar.get_y() + bar.get_height() / 2,
                f"  {val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, facecolor=BG)
    plt.close()
    print(f"  💾 {save_path}")
    return df_imp.reset_index(drop=True)

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
    print(f"  💾 {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 執行入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects",  default="projects_summary.csv")
    parser.add_argument("--img_ratio", default="image_text_ratio_result.csv")
    parser.add_argument("--video",     default="video_features_result.csv")
    parser.add_argument("--pmf",       default="Project_PMF_Simple_for_simluation.csv")
    parser.add_argument("--feat_text", default="feat_out_text.csv")
    parser.add_argument("--top_n",     type=int, default=25)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 載入並處理資料
    df = load_data(args.projects, args.img_ratio, args.video,
                   args.pmf, args.feat_text)

    X, y = preprocess(df)
    
    # 用於全域相關性分析的 DataFrame (合併 X 與 y)
    df_plot = pd.concat([X, y], axis=1)

    print("\n" + "=" * 60)
    print("📈 繪製進階特徵分析圖表")
    print("=" * 60)
    plot_heatmap(df_plot, os.path.join(OUTPUT_DIR, "01_heatmap.png"))
    plot_success_corr_bar(df_plot, os.path.join(OUTPUT_DIR, "02_success_corr.png"))

    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    # 2. Random Forest
    print("\n🌲 訓練 Random Forest")
    rf   = train_rf(X_train, y_train)
    rf_m = evaluate(rf, X_test, y_test, "RF")
    rf_df = plot_importance(
        rf.feature_importances_, FEATURE_COLS,
        f"Random Forest 特徵重要性｜F1={rf_m['f1']:.3f}  AUC={rf_m['auc']:.3f}",
        COLOR_RF, os.path.join(OUTPUT_DIR, "03_rf_importance.png"), args.top_n,
    )

    # 3. XGBoost
    print("\n⚡ 訓練 XGBoost")
    xgb = train_xgb(X_train, y_train)
    if xgb:
        xgb_m  = evaluate(xgb, X_test, y_test, "XGB")
        xgb_df = plot_importance(
            get_xgb_gain(xgb, FEATURE_COLS), FEATURE_COLS,
            f"XGBoost 特徵重要性 (Gain)｜F1={xgb_m['f1']:.3f}  AUC={xgb_m['auc']:.3f}",
            COLOR_XGB, os.path.join(OUTPUT_DIR, "04_xgb_importance.png"), args.top_n,
        )

        print("\n📊 繪製雙模型重要性對比圖")
        plot_comparison(rf_df, xgb_df, os.path.join(OUTPUT_DIR, "05_comparison.png"), args.top_n)

        # 綜合排名 CSV
        combined = rf_df[["feature", "label", "importance"]].rename(
            columns={"importance": "rf_importance"}
        ).merge(
            xgb_df[["feature", "importance"]].rename(
                columns={"importance": "xgb_gain_importance"}),
            on="feature", how="outer",
        ).fillna(0)
        combined["avg_importance"] = (
            combined["rf_importance"] + combined["xgb_gain_importance"]) / 2
        combined = combined.sort_values("avg_importance", ascending=False)
        out_csv  = os.path.join(OUTPUT_DIR, "feature_importance_combined.csv")
        combined.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n📄 綜合排名 → {out_csv}")

    print("\n🎉 完成！圖表請至 output_plots 資料夾查看。")