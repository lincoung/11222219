import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

def analyze_and_plot_aci(datasets, test_ages, target_fc):
    """
    功能：處理混凝土抗壓強度數據、計算 ACI 209 雙曲線模型參數，並繪製平滑預測曲線圖。
    """
    all_results = []
    
    # 準備繪圖畫布
    plt.figure(figsize=(12, 7), dpi=100)
    
    # 生成連續平滑的時間軸 (從 1 天到 95 天，共 1000 個點)
    time_smooth = np.linspace(1, 95, 1000)
    
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    for idx, (group_name, strength_matrix) in enumerate(datasets):
        avg_strengths = []
        detailed_records = []
        
        # 1. 計算各齡期 10 組試體的平均抗壓強度
        for i in range(len(test_ages)):
            age = test_ages[i]
            trials = [trial[i] for trial in strength_matrix]
            mean_str = sum(trials) / len(trials)
            avg_strengths.append(mean_str)
            
        # 取得 28 天的基準強度 f_28
        idx_28 = test_ages.index(28)
        f_28 = avg_strengths[idx_28]
        
        # 2. 評估各齡期狀態與達成率
        for i in range(len(test_ages)):
            age = test_ages[i]
            mean_str = avg_strengths[i]
            ratio = (mean_str / target_fc) * 100
            
            if ratio >= 120: status = "強度表現極佳，超越設計標準"
            elif ratio >= 100: status = "已達到設計抗壓強度"
            elif ratio >= 80: status = "強度發展良好，接近達標"
            else: status = "早期強度發展中，需留意"
                
            detailed_records.append({
                "age": age,
                "mean_strength": round(mean_str, 2),
                "ratio_pct": round(ratio, 1),
                "status": status
            })
        
        # 3. ACI 209 雙曲線模型反推 (線性化形式: t * f_28 / f(t) = alpha + beta * t)
        n = len(test_ages)
        x_aci = test_ages
        y_aci = [(age * f_28) / str_v for age, str_v in zip(test_ages, avg_strengths)]
        
        sum_x = sum(x_aci)
        sum_y = sum(y_aci)
        sum_xy = sum(x * y for x, y in zip(x_aci, y_aci))
        sum_xx = sum(x ** 2 for x in x_aci)
        
        beta = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
        alpha = (sum_y - beta * sum_x) / n
        ultimate_strength = (f_28 / beta) if beta != 0 else 0
        
        # 4. 計算連續平滑曲線的理論強度值
        # 公式: f(t) = [ t / (alpha + beta * t) ] * f_28
        y_smooth = [(t / (alpha + beta * t)) * f_28 for t in time_smooth]
        
        # 繪製該組的平滑理論曲線
        plt.plot(time_smooth, y_smooth, label=f'{group_name} (ACI Model)', color=colors[idx], linestyle='-', linewidth=2)
        
        # 繪製實測數據平均點 (7, 28, 91天)
        plt.plot(test_ages, avg_strengths, markers[idx], markersize=8, color=colors[idx])
        
        all_results.append({
            "group_name": group_name,
            "f_28": f_28,
            "records": detailed_records,
            "alpha": alpha,
            "beta": beta,
            "ultimate_strength": ultimate_strength
        })

    # 5. 圖表外觀與標註設定
    plt.title('Concrete Strength Development Curves (ACI 209 Hyperbolic Model)', fontsize=14, fontweight='bold')
    plt.xlabel('Age (Days)', fontsize=12)
    plt.ylabel('Compressive Strength (MPa)', fontsize=12)
    plt.xlim(0, 95)
    plt.ylim(10, 60)
    
    # 繪製設計強度目標參考線
    plt.axhline(y=target_fc, color='gray', linestyle='--', label=f"Design Target f'c = {target_fc} MPa")
    
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    return all_results

# ==========================================
# 準備測試資料 (齡期：7, 28, 91 天；各 10 組試體；M1, M2, M3 三種配比)
# ==========================================
test_ages = [7, 28, 91]
design_target_fc = 30.0

# M1 組：天然砂控制組
group_M1_data = [
    [20.2, 37.6, 44.6], [20.5, 38.0, 45.0], [19.8, 37.1, 44.1], [20.8, 38.5, 45.5], [20.0, 37.4, 44.4],
    [20.3, 37.8, 44.8], [20.6, 38.2, 45.2], [19.5, 36.9, 43.8], [20.1, 37.5, 44.5], [20.4, 37.9, 44.9]
]

# M2 組：50% 再生細粒料組
group_M2_data = [
    [21.0, 32.9, 48.6], [21.5, 33.4, 49.2], [20.6, 32.4, 48.0], [21.8, 33.8, 49.8], [20.8, 32.7, 48.3],
    [21.2, 33.1, 48.8], [21.6, 33.5, 49.3], [20.4, 32.2, 47.8], [21.1, 33.0, 48.5], [21.4, 33.3, 49.0]
]

# M3 組：5% 再生石粉複合組[cite: 1]
group_M3_data = [
    [19.6, 38.4, 47.2], [20.0, 39.0, 48.0], [19.2, 37.8, 46.5], [20.3, 39.5, 48.8], [19.4, 38.1, 46.8],
    [19.8, 38.6, 47.5], [20.1, 39.1, 48.2], [19.0, 37.5, 46.0], [19.7, 38.3, 47.0], [19.9, 38.8, 47.8]
]

datasets = [
    ("M1 (天然砂控制組)", group_M1_data),
    ("M2 (50%再生細粒料組)", group_M2_data),
    ("M3 (5%再生石粉複合組)", group_M3_data)
]

# ==========================================
# 執行分析與輸出終端機數據
# ==========================================
results = analyze_and_plot_aci(datasets, test_ages, design_target_fc)

print("==================================================")
print("     【ACI 209 標準模型分析結果與數據總表】")
print("==================================================\n")

for res in results:
    print(f"【組別：{res['group_name']}】")
    print(f"  * 28天基準強度 (f_28): {res['f_28']:.2f} MPa")
    for rec in res["records"]:
        print(f"    - 齡期 {rec['age']:2d} 天 | 平均強度: {rec['mean_strength']:5.2f} MPa | 達成率: {rec['ratio_pct']:5.1f}% | 狀態: {rec['status']}")
    print(f"  * ACI 模型係數: alpha = {res['alpha']:.4f}, beta = {res['beta']:.4f}")
    print(f"  * 推估長期極限強度 (t→∞): {res['ultimate_strength']:.2f} MPa")
    print("-" * 50)

plt.show()