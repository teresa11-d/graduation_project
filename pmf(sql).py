"""
PMF 完整特徵計算系統 - XGBoost 訓練用 (SQLite 升級版)
========================================
優化項目：
1. 完全捨棄 Pandas DataFrame 切片，改用 SQLite 原生語法查詢
2. 將【特徵1：近期成功率】與【特徵2：市場飽和度】合併為單一 SQL 查詢，效能翻倍
3. 【特徵4：爆款篩選】直接透過 SQL 的數學運算 (募資金額 / 目標金額 >= 1.5) 精準撈取
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
import warnings
import traceback # 用來印出詳細錯誤追蹤
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


import time
import random

# ========================================
# 工具類：Google Trends 爬蟲 (🚀 終極真言偵錯版 + pytrends-modern)
# ========================================
try:
    from pytrends_modern import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    print("⚠️  pytrends-modern 未安裝，Google Trends 功能不可用")


CACHE_FILE = 'trends_cache.json'

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

class GoogleTrendsFetcher:
    def __init__(self):
        self.cache = load_cache()   # ← 從磁碟讀快取
        self.consecutive_failures = 0
        self.base_wait = 20         # 基礎等待秒數

    def fetch_trend_slope(self, keywords_str, verbose=False):
        if not keywords_str or pd.isna(keywords_str):
            return np.nan

        keywords_str = str(keywords_str).strip()

        # 命中快取直接回傳，不打 API
        if keywords_str in self.cache:
            print(f"  💾 快取命中：{keywords_str[:30]}")
            return self.cache[keywords_str]

        keywords_list = list(set([kw.strip() for kw in keywords_str.split(',')]))[:3]
        print(f"  🔍 查詢 Trends: {keywords_list}")

        # 指數退避等待
        wait_time = self.base_wait * (2 ** min(self.consecutive_failures, 4))
        jitter = random.uniform(0, wait_time * 0.3)
        actual_wait = wait_time + jitter
        print(f"  ⏳ 等待 {actual_wait:.0f} 秒（連續失敗 {self.consecutive_failures} 次）")
        time.sleep(actual_wait)

        try:
            from pytrends_modern import TrendReq
            pytrends = TrendReq(hl='zh-TW', tz=480, retries=2, backoff_factor=2)
            pytrends.build_payload(keywords_list, cat=0, timeframe='today 12-m', geo='TW')
            df = pytrends.interest_over_time()

            if df.empty:
                self.cache[keywords_str] = None   # None 代表「查過但無資料」
                save_cache(self.cache)
                self.consecutive_failures = 0
                return np.nan

            slopes = []
            for i, keyword in enumerate(keywords_list):
                if i > 0:
                    wait = random.uniform(15, 30)
                    print(f"  ⏳ 等待 {wait:.0f} 秒再查下一個關鍵字...")
                    time.sleep(wait)
                
                try:
                    pytrends = TrendReq(hl='zh-TW', tz=480, retries=2, backoff_factor=2)
                    pytrends.build_payload([keyword], cat=0, timeframe='today 12-m', geo='TW')
                    df = pytrends.interest_over_time()
                    
                    if not df.empty and keyword in df.columns:
                        values = df[keyword].values
                        if np.sum(values) > 0:
                            slope = self._calculate_slope(values)
                            if slope is not None:
                                slopes.append(slope)
                                print(f"  ✅ [{keyword}] 斜率: {slope:.4f}")
                except Exception as e:
                    print(f"  ❌ [{keyword}] 失敗: {str(e)[:50]}")
                    continue

            result = round(float(np.mean(slopes)), 4) if slopes else np.nan
            self.cache[keywords_str] = result
            save_cache(self.cache)             # ← 成功就立刻存檔
            self.consecutive_failures = 0      # 重置失敗計數
            return result

        except Exception as e:
            self.consecutive_failures += 1
            err_str = str(e)

            if '429' in err_str:
                print(f"  🚫 429 被封鎖！連續第 {self.consecutive_failures} 次失敗")
                # 被封鎖時直接睡更久再重試一次
                extra_wait = 120 * self.consecutive_failures
                print(f"  😴 額外等待 {extra_wait} 秒...")
                time.sleep(extra_wait)
            else:
                print(f"  ❌ 非 429 錯誤: {err_str[:60]}")

            # 快取為 None，代表「本次嘗試失敗，下次可重試」
            # 如果你希望跳過失敗項目不重試，改成 self.cache[keywords_str] = None
            return np.nan
# ========================================
# 核心類：PMF 特徵計算器 (🚀 終極 SQL 偵錯版)
# ========================================
class PMFFeatureCalculator:
    """透過 SQLite 計算 PMF 特徵，具備詳細偵錯機制"""
    
    def __init__(self, df_main, db_path):
        self.df_main = df_main
        self.db_path = db_path
        
        # 測試資料庫連線
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"找不到 SQLite 資料庫：{self.db_path}")
            
        self.trends_fetcher = GoogleTrendsFetcher()
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print(f"✅ AI 模型與 SQLite 資料庫連線初始化完成\n")
    
    # ========== 特徵1 & 特徵2：SQL 查詢 ==========
    def calculate_market_features(self, project_start_date, category):
        target_date = pd.to_datetime(project_start_date).strftime('%Y-%m-%d')
        
        # 【注意】欄位名稱使用雙引號包裝是最安全的 SQL 寫法
        query = f"""
        SELECT 
            COUNT(*) AS platform_total,
            SUM(CASE WHEN "次分類" = '{category}' THEN 1 ELSE 0 END) AS category_total,
            SUM(CASE WHEN "次分類" = '{category}' AND "募資狀態" = '成功' THEN 1 ELSE 0 END) AS category_success
        FROM projects
        WHERE "開始日期" >= date('{target_date}', '-1 year') 
          AND "開始日期" < '{target_date}';
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                row = cursor.fetchone()
                
            platform_total = row[0] or 0
            category_total = row[1] or 0
            category_success = row[2] or 0
            
            # 成功率：沒有資料就回傳 NaN
            if category_total > 0:
                category_recent_success = round(category_success / category_total, 3)
            else:
                category_recent_success = np.nan
                
            # 飽和度：沒有資料就回傳 NaN
            if platform_total > 0:
                ratio = category_total / platform_total
                market_saturation = round(1 / (1 + np.exp(-10 * (ratio - 0.5))), 3)
            else:
                market_saturation = np.nan
                
            return category_recent_success, market_saturation
            
        except sqlite3.Error as e:
            print(f"\n[偵錯] 特徵 1,2 SQL 執行失敗！")
            print(f"錯誤訊息: {e}")
            print(f"執行的語法為:\n{query}")
            return np.nan, np.nan

    # ========== 特徵4：爆款相似度 (SQL 嚴謹型別版) ==========
    def calculate_google_trend_slope(self, trend_keywords):
        if pd.isna(trend_keywords): return np.nan
        return self.trends_fetcher.fetch_trend_slope(str(trend_keywords))
    
    # ========== 特徵4：爆款相似度 (SQL 嚴謹型別版) ==========
    def calculate_blockbuster_similarity(self, long_tail_keywords, category):
        if pd.isna(long_tail_keywords): return np.nan
        
        try:
            product_embedding = self.model.encode([str(long_tail_keywords)])[0]
        except:
            return np.nan
        
        # 【關鍵修正】：
        # 1. 欄位名稱用雙引號 `"達標率(%)"`
        # 2. 強制轉型 CAST(... AS REAL) 確保 SQLite 把它當作浮點數處理
        # 3. 條件改為 > 1.5 (大於 150%)
        query = f"""
        SELECT "專案編號", "達標率(%)"
        FROM projects
        WHERE "次分類" = '{category}' 
          AND "募資狀態" = '成功'
          AND CAST("達標率(%)" AS REAL) > 1.5;
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 把查詢結果讀出來
                bestseller_df = pd.read_sql_query(query, conn)
                bestseller_ids = bestseller_df['專案編號'].tolist()
                
        except sqlite3.Error as e:
            print(f"\n[偵錯] 特徵 4 (爆款) SQL 執行失敗！")
            print(f"錯誤訊息: {e}")
            print(f"執行的語法為:\n{query}")
            return np.nan
            
        # 如果沒有找到大於 150% 的專案
        if not bestseller_ids:
            return np.nan  
            
        # 從 df_main 中找出對應的關鍵字
        bestsellers_data = self.df_main[self.df_main['Project_Name'].str.split('_').str[0].isin(bestseller_ids)]
        
        similarities = []
        for _, row in bestsellers_data.iterrows():
            kw = str(row.get('trend_keywords', ''))
            if pd.isna(kw) or not kw.strip(): continue
                
            try:
                b_emb = self.model.encode([kw])[0]
                sim = cosine_similarity([product_embedding], [b_emb])[0][0]
                similarities.append(sim)
            except:
                continue
        
        if not similarities: return np.nan
        return round(float(np.max(similarities)), 3)
    
    # ========== 計算所有特徵 ==========
    def calculate_all_features(self, row, project_date_map):
        project_name = row.get('Project_Name')
        category = row.get('Category')
        outcome = row.get('Outcome')
        long_tail_keywords = row.get('Long-tail_Keywords')
        trend_keywords = row.get('trend_keywords')
        
        # 【關鍵修正】：把長碼 "ES1_rogerems" 切割，只保留底線前面的 "ES1"
        short_id = str(project_name).split('_')[0]
        
        # 用短碼去映射表取 開始日期
        start_date = project_date_map.get(short_id)
        
        if pd.isna(start_date) or start_date is None: 
            return None
        
        # 計算特徵
        f1, f2 = self.calculate_market_features(start_date, category)
        f3 = self.calculate_google_trend_slope(trend_keywords)
        f4 = self.calculate_blockbuster_similarity(long_tail_keywords, category)
        
        return {
            'project_id': project_name,
            'category': category,
            'outcome': outcome,
            'outcome_binary': 1 if outcome == '成功' else 0,
            'category_recent_success': f1,
            'market_saturation': f2,
            'google_trend_slope': f3,
            'blockbuster_similarity': f4,
        }

# ========================================
# 主程式
# ========================================
def main():
    print("=" * 80)
    print("PMF 特徵計算系統 - ⚡️ SQLite 極速驅動版")
    print("=" * 80)
    
    # 檔案路徑設定
    dataset_csv_path = 'Project_PMF_Dataset.csv'
    sqlite_db_path = 'zeczec_history.db'  # 你之前用 csv_to_sqlite 建立的資料庫
    
    # Step 1：讀取主要特徵庫
    print("\n[Step 1] 讀取主要專案資料")
    if not os.path.exists(dataset_csv_path):
        print(f"❌ 找不到檔案：{dataset_csv_path}")
        return
        
    df_main = pd.read_csv(dataset_csv_path)
    print(f"✅ Project_PMF_Dataset 讀取成功 ({len(df_main)} 筆)\n")
    
    # Step 2：建立日期映射表 (直接從 SQL 撈出 ID 與 日期)
    print("[Step 2] 從資料庫建立專案日期映射表")
    try:
        with sqlite3.connect(sqlite_db_path) as conn:
            date_df = pd.read_sql_query("SELECT 專案編號, 開始日期 FROM projects;", conn)
            project_date_map = dict(zip(date_df['專案編號'], date_df['開始日期']))
            print(f"✅ 成功撈取 {len(project_date_map)} 筆專案日期\n")
    except Exception as e:
        print(f"❌ 讀取 SQLite 失敗，請確認資料庫是否存在。錯誤: {e}")
        return

    # Step 3：初始化計算器
    print("[Step 3] 初始化特徵計算引擎")
    calculator = PMFFeatureCalculator(df_main, sqlite_db_path)
    
    # Step 4：計算特徵
    PROGRESS_FILE = 'pmf_progress.json'

    def load_progress():
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        return {'done_ids': [], 'results': []}

    def save_progress(done_ids, results):
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({'done_ids': done_ids, 'results': results}, f, ensure_ascii=False)

# ---- 在 Step 4 替換原本的 for loop ----
    progress = load_progress()
    done_ids = set(progress['done_ids'])
    results = progress['results']

    print(f"  📌 已完成 {len(done_ids)} 筆，從第 {len(done_ids)+1} 筆繼續\n")

    for idx, row in df_main.iterrows():
        project_id = row.get('Project_Name')

        # 跳過已完成的
        if project_id in done_ids:
            print(f"[{idx+1}/{len(df_main)}] {project_id[:35]:<35s} ⏭️  (已快取)")
            continue

        print(f"[{idx+1}/{len(df_main)}] {project_id[:35]:<35s}", end=' ')

        try:
            features = calculator.calculate_all_features(row, project_date_map)
            if features is None:
                print("❌ (無對應歷史日期)")
                done_ids.add(project_id)   # 標記為已處理（雖然失敗）
            else:
                results.append(features)
                done_ids.add(project_id)
                print("✅")
        except Exception as e:
            print(f"❌ ({str(e)[:30]})")

    # 每筆完成後立即存進度
    save_progress(list(done_ids), results)
            
    # Step 5：存檔與統計
    print(f"\n[Step 5] 儲存結果與數據檢查\n")
    df_features = pd.DataFrame(results)
    output_path = './PMF_Features_for_XGBoost.csv'
    df_features.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"🎉 已將 {len(df_features)} 筆計算結果存至：{output_path}\n")
    
    feature_cols = ['category_recent_success', 'market_saturation', 
                    'google_trend_slope', 'blockbuster_similarity']
    
    print("【特徵統計分佈】")
    print(df_features[feature_cols].describe())
    
    print("\n【準備進入 XGBoost 階段】")

if __name__ == '__main__':
    main()