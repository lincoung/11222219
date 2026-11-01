import math

def analyze_concrete_strength(group_name, ages, strength_matrix, target_fc):
    """
    功能（Process）：處理混凝土抗壓強度數據，計算 10 組試體在各齡期的平均值、
                  評估達標狀態，並利用對數成長模型反推預測方程式係數。
    參數（Input 接收）：
      - group_name: 試體組別名稱 (str)
      - ages: 測試齡期清單，如 [7, 28, 91] (list)
      - strength_matrix: 10組試體的強度二維清單 (10 rows x 3 cols) (list of lists)
      - target_fc: 設計抗壓強度目標 (float)
    回傳（return）：
      - 包含分析結果與反推係數的結構化字典 (dict)
    """
    analysis_results = {}
    avg_strengths = []
    detailed_records = []
    
    # 1. 處理過程 (Process)：使用 for 迴圈與 list 計算各齡期 10 組試體的平均抗壓強度
    for i in range(len(ages)):
        age = ages[i]
        # 萃取該齡期底下所有 10 個試體的數據
        trials = [trial[i] for trial in strength_matrix]
        mean_str = sum(trials) / len(trials)
        avg_strengths.append(mean_str)
        
        ratio = (mean_str / target_fc) * 100
        
        # 使用 if / elif / else 進行條件判斷
        if ratio >= 120:
            status = "強度表現極佳，超越設計標準"
        elif ratio >= 100:
            status = "已達到設計抗壓強度"
        elif ratio >= 80:
            status = "強度發展良好，接近達標"
        else:
            status = "早期強度發展中，需留意"
            
        detailed_records.append({
            "age": age,
            "trial_count": len(trials),
            "mean_strength": round(mean_str, 2),
            "ratio_pct": round(ratio, 1),
            "status": status
        })
        
    # 2. 數學模型處理 (Process)：使用 math 模組與純 Python 進行對數成長模型反推
    # 公式形式: f(t) = b * ln(t) + a
    n = len(ages)
    x_log = [math.log(a) for a in ages]
    y_log = avg_strengths
    
    sum_x = sum(x_log)
    sum_y = sum(y_log)
    sum_xy = sum(x * y for x, y in zip(x_log, y_log))
    sum_xx = sum(x ** 2 for x in x_log)
    
    b_coeff = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    a_coeff = (sum_y - b_coeff * sum_x) / n
    
    # 3. 使用 dict 組織回傳結果 (Output 結構化)
    analysis_results["group_name"] = group_name
    analysis_results["records"] = detailed_records
    analysis_results["model_formula"] = f"f(t) = {b_coeff:.4f} * ln(t) + {a_coeff:.4f}"
    
    return analysis_results

# ==========================================
# 準備測試資料 (齡期：7, 28, 91 天；各 10 組試體；M1, M2, M3 三種配比)
# ==========================================
test_ages = [7, 28, 91]       # 輸入 (Input)：齡期資料 (list)
design_target_fc = 30.0      # 輸入 (Input)：設計強度目標 (MPa)

# M1 組：傳統天然砂控制組（10組試體在 7天、28天、91天的實測數據）
group_M1_data = [
    [20.2, 37.6, 44.6],
    [20.5, 38.0, 45.0],
    [19.8, 37.1, 44.1],
    [20.8, 38.5, 45.5],
    [20.0, 37.4, 44.4],
    [20.3, 37.8, 44.8],
    [20.6, 38.2, 45.2],
    [19.5, 36.9, 43.8],
    [20.1, 37.5, 44.5],
    [20.4, 37.9, 44.9]
]

# M2 組：50% 再生細粒料複合組（10組試體在 7天、28天、91天的實測數據）
group_M2_data = [
    [21.0, 32.9, 48.6],
    [21.5, 33.4, 49.2],
    [20.6, 32.4, 48.0],
    [21.8, 33.8, 49.8],
    [20.8, 32.7, 48.3],
    [21.2, 33.1, 48.8],
    [21.6, 33.5, 49.3],
    [20.4, 32.2, 47.8],
    [21.1, 33.0, 48.5],
    [21.4, 33.3, 49.0]
]

# M3 組：摻加 5% 再生石粉之複合組（10組試體在 7天、28天、91天的實測數據）[cite: 1]
group_M3_data = [
    [19.6, 38.4, 47.2],
    [20.0, 39.0, 48.0],
    [19.2, 37.8, 46.5],
    [20.3, 39.5, 48.8],
    [19.4, 38.1, 46.8],
    [19.8, 38.6, 47.5],
    [20.1, 39.1, 48.2],
    [19.0, 37.5, 46.0],
    [19.7, 38.3, 47.0],
    [19.9, 38.8, 47.8]
]

datasets = [
    ("M1 組 (天然砂控制組)", group_M1_data),
    ("M2 組 (50%再生細粒料組)", group_M2_data),
    ("M3 組 (5%再生石粉複合組)", group_M3_data)
]

# ==========================================
# 執行程式與輸出結果 (Output)
# ==========================================
print("==================================================")
print("  【混凝土抗壓強度分析與曲線係數反推程式 (M1-M3)】")
print("==================================================\n")

for group_name, strength_matrix in datasets:
    # 呼叫自定義函式並接收 return 的 dict 結果
    result = analyze_concrete_strength(group_name, test_ages, strength_matrix, design_target_fc)
    
    print(f"【組別：{result['group_name']}】")
    
    # 使用 for 迴圈與 dict 讀取並顯示處理後的詳細結果
    for rec in result["records"]:
        print(f"  * 齡期 {rec['age']:2d} 天 | 試體數量: {rec['trial_count']} 組 | "
              f"平均強度: {rec['mean_strength']:5.2f} MPa | 達成率: {rec['ratio_pct']:5.1f}% | 狀態: {rec['status']}")
        
    print(f"  ==> 反推預測曲線方程式: {result['model_formula']}\n")
    print("-" * 50)