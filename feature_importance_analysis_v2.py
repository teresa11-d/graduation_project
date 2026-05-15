"""
feature_importance_analysis_v4.py
================================
募資平台特徵重要性分析 (強制欄位索引對齊 + 模糊比對版)
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
    preferred_fonts = ["PingFang TC", "Noto Sans CJK TC", "Microsoft JhengHei", "SimHei"]
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in preferred_fonts:
        if font in available_fonts: return font
    return "sans-serif"

CJK_FONT = setup_chinese_font()
rcParams["font.family"] = CJK_FONT
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 150

COLOR_RF, COLOR_XGB, COLOR_NEG, COLOR_POS, BG, GRID = "#3B6D11", "#185FA5", "#A32D2D", "#0F6E56", "#FAFAF8", "#E8E6DF"

# ── 特徵顯示名稱 ──────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "target_amount_log": "目標金額 (log)", "duration_days": "募資天數", "price_tier_count": "回饋方案層級數",
    "折扣層數": "折扣層數", "FAQ總題數": "FAQ總題數", "FAQ更新頻率": "FAQ更新頻率",
    "price_min": "最低方案價格", "price_max": "最高方案價格", "price_avg": "平均方案價格",
    "main_cat_encoded": "主分類(編碼)", "sub_cat_encoded": "次分類(編碼)",
    "story_ratio": "故事段落比例", "spec_ratio": "規格段落比例", "risk_ratio": "風險說明比例", "has_social_links": "社群連結數",
    "feat_adverb_density": "副詞密度", "feat_adj_nv_ratio": "形/(名+動)比例", "feat_punct_intensity": "標點符號強度",
    "feat_numeral_density": "數字密度", "feat_entity_density": "實體密度", "feat_avg_sentence_len": "平均句長",
    "feat_type_token_ratio": "詞彙多樣性(TTR)", "img_text_ratio": "圖文密度比", "word_count": "文案總字數",
    "image_count": "圖片數量", "video_count": "影片數量", "media_per_100_words": "多媒體密度(每百字)",
    "has_video": "有主影片", "video_duration": "影片長度(秒)", "usage_scene_ratio": "實際操作情境佔比", 
    "category_recent_success": "同類近期成功率", "market_saturation": "市場飽和度", "google_trend_slope": "Google趨勢斜率", "blockbuster_similarity": "爆款相似度"
}

FEATURE_COLS = list(FEATURE_LABELS.keys())
TARGET_COL, OUTPUT_DIR = "success_label", "output_plots"

# ══════════════════════════════════════════════════════════════════════════════
# 1. 智慧資料合併引擎 (導入絕對位置鎖定)
# ══════════════════════════════════════════════════════════════════════════════

def set_id_by_index(df, index_pos, name=""):
    """強制將指定位置的欄位重新命名為 project_id，解決名稱不一致問題"""
    if df.empty: return df
    
    if len(df.columns) > index_pos:
        original_col_name = df.columns[index_pos]
        df.rename(columns={original_col_name: 'project_id'}, inplace=True)
        # 確保轉為字串並去除多餘空白
        df['project_id'] = df['project_id'].astype(str).str.strip()
        print(f"  📌 [{name}] 已強制將第 {index_pos+1} 欄 '{original_col_name}' 設為合併鍵值。")
    else:
        print(f"  ❌ [{name}] 錯誤：找不到第 {index_pos+1} 欄，該檔案欄位數不足。")
    return df

def robust_merge(df_main, df_sub, name):
    """強大的模糊比對合併功能"""
    if df_main.empty or df_sub.empty: return df_main
    
    # 1. 精準比對
    exact_matches = set(df_main['project_id']).intersection(set(df_sub['project_id']))
    if len(exact_matches) > 0:
        print(f"  ✅ [{name}] 精準對齊了 {len(exact_matches)} 筆專案！")
        return pd.merge(df_main, df_sub, on='project_id', how='left')
        
    # 2. 模糊比對 (切除底線 _ 後的字串，例如 ES1_rogerems -> ES1)
    print(f"  ⚠️ [{name}] 無法精準匹配，啟動「底線前綴」模糊比對...")
    df_main['_prefix'] = df_main['project_id'].str.split('_').str[0]
    df_sub['_prefix'] = df_sub['project_id'].str.split('_').str[0]
    
    prefix_matches = set(df_main['_prefix']).intersection(set(df_sub['_prefix']))
    if len(prefix_matches) > 0:
        print(f"  ✅ [{name}] 模糊比對成功！救回並對齊了 {len(prefix_matches)} 筆專案！")
        df_sub = df_sub.drop(columns=['project_id'])
        df_res = pd.merge(df_main, df_sub, on='_prefix', how='left')
        return df_res.drop(columns=['_prefix'])
        
    # 3. 徹底失敗
    print(f"  ❌ [{name}] 比對徹底失敗！(主檔與此檔的 ID 完全不相干)")
    df_main = df_main.drop(columns=['_prefix'], errors='ignore')
    return pd.merge(df_main, df_sub.drop(columns=['_prefix'], errors='ignore'), on='project_id', how='left')

def load_from_csvs(projects_path, result_path, video_path, feat_text_path, img_ratio_path):
    print("\n" + "="*60 + "\n📥 啟動多模態資料合併系統 (絕對欄位對齊)\n" + "="*60)
    
    # ⭐ 依照您的指示：主檔取第 4 欄 (index=3)，其他取第 1 欄 (index=0)
    df_proj = set_id_by_index(pd.read_csv(projects_path, encoding="utf-8-sig") if os.path.exists(projects_path) else pd.DataFrame(), 3, "主專案檔")
    df_res  = set_id_by_index(pd.read_csv(result_path, encoding="utf-8-sig") if os.path.exists(result_path) else pd.DataFrame(), 0, "文案結構檔")
    
    df = robust_merge(df_proj, df_res, "文案結構檔") if not df_proj.empty and not df_res.empty else (df_proj if not df_proj.empty else df_res)

    if video_path and os.path.exists(video_path):
        df_vid = set_id_by_index(pd.read_csv(video_path, encoding="utf-8-sig"), 0, "影片特徵檔")
        if 'project_id' in df_vid.columns:
            df_vid_agg = df_vid.groupby('project_id').agg({'total_duration': 'sum', 'usage_scene_ratio': 'mean'}).reset_index() if 'total_duration' in df_vid.columns else df_vid
            if 'total_duration' in df_vid_agg.columns: df_vid_agg.rename(columns={'total_duration': 'video_duration'}, inplace=True)
            df = robust_merge(df, df_vid_agg, "影片特徵檔")

    if feat_text_path and os.path.exists(feat_text_path):
        df_feat = set_id_by_index(pd.read_csv(feat_text_path, encoding="utf-8-sig"), 0, "進階文本檔")
        df_feat.rename(columns={'feat_text_story_ratio': 'story_ratio', 'feat_text_spec_ratio': 'spec_ratio', 'feat_text_risk_ratio': 'risk_ratio', 'feat_has_social_link': 'has_social_links'}, inplace=True)
        if 'project_id' in df_feat.columns:
            df_feat_agg = df_feat.groupby('project_id').mean(numeric_only=True).reset_index()
            df = robust_merge(df, df_feat_agg, "進階文本檔")

    if img_ratio_path and os.path.exists(img_ratio_path):
        df_img = set_id_by_index(pd.read_csv(img_ratio_path, encoding="utf-8-sig"), 0, "圖文比例檔")
        if 'project_id' in df_img.columns:
            df_img_agg = df_img.groupby('project_id').mean(numeric_only=True).reset_index()
            df = robust_merge(df, df_img_agg, "圖文比例檔")

    # 目標變數設定
    if '募資狀態' in df.columns: df[TARGET_COL] = (df['募資狀態'] == '成功').astype(int)
    else: df[TARGET_COL] = 0
        
    for new_col, ori_col in [("main_cat_encoded", "主分類"), ("sub_cat_encoded", "次分類")]:
        if ori_col in df.columns: df[new_col] = df[ori_col].astype('category').cat.codes
        else: df[new_col] = np.nan

    for col in FEATURE_COLS:
        if col not in df.columns: df[col] = np.nan
            
    df = df[FEATURE_COLS + [TARGET_COL]].dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df

def preprocess(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]
    
    for col in X.columns: X[col] = pd.to_numeric(X[col], errors='coerce')
        
    empty_cols = X.columns[X.isna().all()].tolist()
    if empty_cols:
        print(f"\n⚠️ [防呆警告] 發現 {len(empty_cols)} 個特徵徹底空缺 (原檔案無此欄位或合併失敗)，自動補 0 確保模型運作。")
        X[empty_cols] = 0
        
    X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=FEATURE_COLS)
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
        model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric="logloss", random_state=seed, verbosity=0)
        model.fit(X_train, y_train)
        return model
    except: return None

def _setup_ax(ax, title: str, xlabel: str = ""):
    ax.set_facecolor(BG); ax.set_title(title, fontsize=13, fontweight="bold", pad=12, loc="left")
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, color="#555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, linestyle="--")

def plot_bar_importance(model, feature_names, title, color, save_path, top_n=20):
    imp = model.feature_importances_ if hasattr(model, 'feature_importances_') else list(model.get_booster().get_score(importance_type='gain').values())
    df = pd.DataFrame({"feature": feature_names, "importance": imp, "label": [FEATURE_LABELS.get(f, f) for f in feature_names]}).sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    data = df.head(top_n).iloc[::-1]
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.42))); fig.patch.set_facecolor(BG)
    _setup_ax(ax, title, xlabel="Feature Importance")
    bars = ax.barh(data["label"], data["importance"], color=color, alpha=0.85, edgecolor="none")
    for bar, val in zip(bars, data["importance"]):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.4f}", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(save_path, facecolor=BG); plt.close()
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 3. 執行入口
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", default="projects_summary.csv")
    parser.add_argument("--result", default="my_result.csv")
    parser.add_argument("--video", default="video_features_result.csv")
    parser.add_argument("--feat_text", default="feat_out_text.csv")
    parser.add_argument("--img_ratio", default="image_text_ratio_result.csv")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_from_csvs(args.projects, args.result, args.video, args.feat_text, args.img_ratio)
    print(f"\n✅ 最終資料維度: {df.shape}，包含特徵數: {len(FEATURE_COLS)}，成功率: {df[TARGET_COL].mean():.1%}")

    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) if len(X) >= 10 else (X, X, y, y)

    rf = train_rf(X_train, y_train)
    rf_imp = plot_bar_importance(rf, FEATURE_COLS, f"Random Forest 特徵重要性 (Acc: {rf.score(X_test, y_test):.3f})", COLOR_RF, os.path.join(OUTPUT_DIR, "01_rf_importance.png"), 25)

    xgb = train_xgb(X_train, y_train)
    if xgb:
        xgb_imp = plot_bar_importance(xgb, FEATURE_COLS, f"XGBoost 特徵重要性 (Acc: {xgb.score(X_test, y_test):.3f})", COLOR_XGB, os.path.join(OUTPUT_DIR, "02_xgb_importance.png"), 25)

    print("\n🎉 分析完成！請至 output_plots 資料夾查看圖表。")