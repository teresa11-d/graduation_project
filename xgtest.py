import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.special import expit

# ==========================================
# 1. 定義 5 大維度與對應特徵
# ==========================================
dimension_map = {
    '1. 專案執行力': ['目標金額', '募資天數'],
    '2. 價格競爭力': ['回饋方案層數', '回饋方案金額中位數', '定價區間合理性'],
    '3. 文案說服力': ['文案字數', '文案感情字詞比例', '文案規格字詞比例', 'FAQ更新比例', 'FAQ題數'],
    '4. 影像吸引力': ['每百字多媒體密度', '有影片', '實際操作情境影片'],
    '5. 市場契合度(PMF)': ['平台內同類商品近期成功率', 'Google趨勢斜率', '爆款相似度']
}

# ==========================================
# 2. 生成模擬資料並訓練 XGBoost
# ==========================================
np.random.seed(42)
n_samples = 1000
df_features = pd.DataFrame({
    # 1. 專案執行力
    '目標金額':              np.random.uniform(50000, 5000000, n_samples),
    '募資天數':              np.random.randint(15, 60, n_samples),
    # 2. 價格競爭力
    '回饋方案層數':          np.random.randint(1, 10, n_samples),          # CSV 中的折扣價格層數
    '回饋方案金額中位數':    np.random.uniform(500, 10000, n_samples),
    '定價區間合理性':        np.random.uniform(0, 1, n_samples),           # 外部資料
    # 3. 文案說服力
    '文案字數':              np.random.randint(500, 5000, n_samples),       # 含 OCR 圖片內字數
    '文案感情字詞比例':      np.random.uniform(0, 0.3, n_samples),
    '文案規格字詞比例':      np.random.uniform(0, 0.4, n_samples),
    'FAQ更新比例':           np.random.uniform(0, 1, n_samples),
    'FAQ題數':               np.random.randint(0, 20, n_samples),
    # 4. 影像吸引力
    '每百字多媒體密度':      np.random.uniform(0, 5, n_samples),
    '有影片':                np.random.randint(0, 2, n_samples),
    '實際操作情境影片':      np.random.randint(0, 2, n_samples),
    # 5. 市場契合度(PMF)
    '平台內同類商品近期成功率': np.random.uniform(0, 1, n_samples),
    'Google趨勢斜率':        np.random.uniform(-1, 1, n_samples),           # 外部資料
    '爆款相似度':            np.random.uniform(0, 1, n_samples),
})

# 模擬成功標籤 (加入隱含邏輯供模型學習)
simulated_y_logic = (
    (df_features['平台內同類商品近期成功率'] * 2)
    + (df_features['有影片'] * 1.5)
    - (df_features['目標金額'] / 3000000)
    + np.random.normal(0, 0.5, n_samples)
)
y = (simulated_y_logic > 1.0).astype(int)

# 訓練模型
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(df_features, y)

# ==========================================
# 3. 系統前置運算 (計算全域權重與分數極值)
# ==========================================
feature_names = df_features.columns.tolist()

# 全域特徵重要性 (各特徵佔整體模型的比例)
global_importances = dict(zip(feature_names, model.feature_importances_))

# 所有歷史專案的 log-odds 貢獻度 (用於 0-100 分的縮放基準)
dmatrix_all = xgb.DMatrix(df_features)
all_contribs = model.get_booster().predict(dmatrix_all, pred_contribs=True)

# 各特徵在歷史資料中的 0-100 縮放基準
feat_min_max = {}
for i, f in enumerate(feature_names):
    vals = all_contribs[:, i]
    feat_min_max[f] = {'min': vals.min(), 'max': vals.max()}

# 各維度在歷史資料中的 0-100 縮放基準
dim_min_max = {}
for dim, feats in dimension_map.items():
    feat_indices = [feature_names.index(f) for f in feats]
    dim_sums = all_contribs[:, feat_indices].sum(axis=1)
    dim_min_max[dim] = {'min': dim_sums.min(), 'max': dim_sums.max()}


def minmax_score(raw, lo, hi):
    """將 raw log-odds 縮放至 0–100，並夾住邊界。"""
    if hi > lo:
        return max(0.0, min(100.0, (raw - lo) / (hi - lo) * 100))
    return 50.0


def status_label(score):
    if score >= 80:
        return "🟢", "表現極佳"
    elif score >= 60:
        return "🟢", "表現良好"
    elif score >= 40:
        return "🟡", "需加強"
    else:
        return "🔴", "高風險"


# ==========================================
# 4. 分析單一專案並產出報表
# ==========================================
project_idx = 5
single_contribs = all_contribs[project_idx]

bias_log_odds  = single_contribs[-1]
feature_log_odds = single_contribs[:-1]

base_prob      = expit(bias_log_odds) * 100
final_prob     = expit(bias_log_odds + np.sum(feature_log_odds)) * 100
total_prob_diff = final_prob - base_prob

# 將 log-odds 貢獻轉換為勝率影響百分比
sum_log_odds = np.sum(feature_log_odds)
if sum_log_odds != 0:
    feature_impact_pct = (feature_log_odds / sum_log_odds) * total_prob_diff
else:
    feature_impact_pct = np.zeros_like(feature_log_odds)

impact_dict = dict(zip(feature_names, feature_impact_pct))

# --- 列印報表 ---
print("=========================================")
print(f" 🚀 專案募資潛力分析報告 (專案編號: #{project_idx})")
print("=========================================")
print(f"🎯 預測整體成功率： {final_prob:.1f}%")
print(f"*(基礎市場勝率 {base_prob:.1f}%，專案自身條件 {total_prob_diff:+.1f}%)*\n")
print("📊 【五大維度健康度與優化指引】")

for dim, feats in dimension_map.items():
    # 維度佔總模型影響力權重 (%)
    dim_weight = sum(global_importances[f] for f in feats) * 100

    # 維度 0-100 健康度分數
    feat_indices  = [feature_names.index(f) for f in feats]
    dim_raw_score = single_contribs[feat_indices].sum()
    dim_score     = minmax_score(dim_raw_score,
                                 dim_min_max[dim]['min'],
                                 dim_min_max[dim]['max'])

    dot, label = status_label(dim_score)

    # ── 維度標題列：「🟢 1. 專案執行力（18%）  68 分  （良好）」
    print(f"\n{dot} {dim}（{dim_weight:.0f}%）  {dim_score:.0f} 分  （{label}）")

    # ── 細項特徵列：依影響力排序，絕對值 ≥ 0.5% 才顯示
    feats_sorted = sorted(feats, key=lambda x: impact_dict[x], reverse=True)

    for f in feats_sorted:
        impact = impact_dict[f]
        if abs(impact) < 0.5:
            continue

        # 特徵自身 0-100 分 (基於其 log-odds 貢獻)
        feat_raw   = single_contribs[feature_names.index(f)]
        feat_score = minmax_score(feat_raw,
                                  feat_min_max[f]['min'],
                                  feat_min_max[f]['max'])

        f_dot, _ = status_label(feat_score)

        if impact > 0:
            suggestion = ""
        elif impact < -3:
            suggestion = " 🚨 *(強烈建議優化)*"
        else:
            suggestion = " *(建議微調)*"

        # 格式：「  ├─ 🟡 目標金額        48 分  *(建議微調)*」
        print(f"   ├─ {f_dot} {f:<12}  {feat_score:>5.0f} 分{suggestion}")

print("\n=========================================")
print("💡 系統建議：請優先針對帶有 🚨 標記的項目進行改善，即可有效提升成功率！")


# ==========================================
# 5. 詳細特徵權重報表
# ==========================================
print("\n\n" + "=" * 55)
print(" 📐 模型特徵權重詳細報表（全域）")
print("=" * 55)

# 取得三種重要性指標
imp_weight  = model.get_booster().get_score(importance_type='weight')   # 該特徵被用於分割的次數
imp_gain    = model.get_booster().get_score(importance_type='gain')     # 每次分割帶來的平均增益
imp_cover   = model.get_booster().get_score(importance_type='cover')    # 每次分割覆蓋的平均樣本數

# 整合成 DataFrame 並歸一化為百分比
all_feats = feature_names
df_imp = pd.DataFrame({
    '特徵': all_feats,
    '分割次數(Weight)': [imp_weight.get(f, 0) for f in all_feats],
    '平均增益(Gain)':   [imp_gain.get(f, 0)   for f in all_feats],
    '覆蓋樣本(Cover)':  [imp_cover.get(f, 0)  for f in all_feats],
})

# 百分比欄位
for col in ['分割次數(Weight)', '平均增益(Gain)', '覆蓋樣本(Cover)']:
    total = df_imp[col].sum()
    df_imp[col + '%'] = (df_imp[col] / total * 100).round(1) if total > 0 else 0

# 依平均增益排序（最能降低損失的特徵排最前）
df_imp = df_imp.sort_values('平均增益(Gain)', ascending=False).reset_index(drop=True)

# 找出特徵所屬維度
def find_dim(feat):
    for d, fs in dimension_map.items():
        if feat in fs:
            return d
    return '—'

df_imp['所屬維度'] = df_imp['特徵'].apply(find_dim)

print(f"\n{'排名':<4} {'特徵':<14} {'所屬維度':<20} {'分割次數%':>9} {'平均增益%':>9} {'覆蓋樣本%':>9}")
print("-" * 72)
for i, row in df_imp.iterrows():
    rank = i + 1
    bar_g = "█" * int(row["平均增益(Gain)%"] / 2)   # 每 2% 一格
    bar_w = "█" * int(row["分割次數(Weight)%"] / 2)
    print(f"{rank:<4} {row['特徵']:<14} {row['所屬維度']:<20} "
          f"{row['分割次數(Weight)%']:>8.1f}% "
          f"{row['平均增益(Gain)%']:>8.1f}% "
          f"{row['覆蓋樣本(Cover)%']:>8.1f}%")

print("\n── 依維度彙總（平均增益%）──")
dim_summary = df_imp.groupby('所屬維度')[['分割次數(Weight)%','平均增益(Gain)%','覆蓋樣本(Cover)%']].sum()
dim_summary = dim_summary.sort_values('平均增益(Gain)%', ascending=False)
for dim, row in dim_summary.iterrows():
    bar = "█" * int(row['平均增益(Gain)%'] / 2)
    print(f"  {dim:<22} 增益 {row['平均增益(Gain)%']:>5.1f}%  {bar}")

print("\n說明：")
print("  分割次數(Weight) — 該特徵出現在決策節點的總次數，反映「使用頻率」")
print("  平均增益(Gain)   — 每次分割平均降低多少損失，反映「實際貢獻度」（最重要）")
print("  覆蓋樣本(Cover)  — 每次分割平均涵蓋多少樣本，反映「影響範圍」")