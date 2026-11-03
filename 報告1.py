import numpy as np
import matplotlib.pyplot as plt

def analyze_and_plot_aci(datasets, test_ages, target_fc):
    """
    功能：處理混凝土抗壓強度數據、計算 ACI 209 雙曲線模型參數，並繪製平滑預測曲線圖。
    """
    ages = np.asarray(test_ages, dtype=float)
    if ages.ndim != 1 or ages.size < 2:
        raise ValueError("test_ages 必須至少包含兩個齡期。")
    if not np.all(np.isfinite(ages)) or np.any(ages <= 0):
        raise ValueError("所有齡期都必須是有限的正數。")
    if np.unique(ages).size != ages.size:
        raise ValueError("test_ages 不可包含重複齡期。")
    if not np.any(ages == 28):
        raise ValueError("test_ages 必須包含 28 天，才能計算 f_28。")
    if not np.isfinite(target_fc) or target_fc <= 0:
        raise ValueError("target_fc 必須是有限的正數。")
    if not datasets:
        raise ValueError("datasets 不可為空。")

    all_results = []
    time_smooth = np.linspace(1, 95, 1000)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ("o", "s", "^", "D", "v", "P", "X")
    plot_data = []
    
    for idx, (group_name, strength_matrix) in enumerate(datasets):
        strengths = np.asarray(strength_matrix, dtype=float)
        if strengths.ndim != 2 or strengths.shape[0] == 0 or strengths.shape[1] != ages.size:
            raise ValueError(
                f"{group_name} 的強度資料必須是非空二維矩陣，"
                "且每列欄數需與 test_ages 相同。"
            )
        if not np.all(np.isfinite(strengths)) or np.any(strengths <= 0):
            raise ValueError(f"{group_name} 的試體強度必須都是有限的正數。")

        avg_strengths = strengths.mean(axis=0)
        detailed_records = []
        idx_28 = int(np.flatnonzero(ages == 28)[0])
        f_28 = float(avg_strengths[idx_28])

        for age, mean_str in zip(test_ages, avg_strengths):
            ratio = (mean_str / target_fc) * 100
            
            if ratio >= 120: status = "強度表現極佳，超越設計標準"
            elif ratio >= 100: status = "已達到設計抗壓強度"
            elif ratio >= 80: status = "強度發展良好，接近達標"
            else: status = "早期強度發展中，需留意"
                
            detailed_records.append({
                "age": age,
                "mean_strength": round(float(mean_str), 2),
                "ratio_pct": round(ratio, 1),
                "status": status
            })
        
        # Constrain the linearized fit to pass through (28, 28), preserving f_28.
        y_aci = ages * f_28 / avg_strengths
        centered_ages = ages - 28.0
        denominator = float(np.dot(centered_ages, centered_ages))
        beta = float(np.dot(centered_ages, y_aci - 28.0) / denominator)
        alpha = 28.0 * (1.0 - beta)
        if beta <= 0:
            raise ValueError(
                f"{group_name} 的 ACI 擬合 beta={beta:.6g}，"
                "無法得到正值的長期極限強度。"
            )

        denominator_smooth = alpha + beta * time_smooth
        if np.any(denominator_smooth <= 0):
            raise ValueError(f"{group_name} 的 ACI 模型在繪圖齡期範圍內分母非正。")
        y_smooth = f_28 * time_smooth / denominator_smooth
        plot_data.append((idx, group_name, avg_strengths, y_smooth))

        all_results.append({
            "group_name": group_name,
            "f_28": f_28,
            "records": detailed_records,
            "alpha": alpha,
            "beta": beta,
            "ultimate_strength": f_28 / beta
        })

    plt.figure(figsize=(12, 7), dpi=100)
    for idx, group_name, avg_strengths, y_smooth in plot_data:
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.plot(
            time_smooth,
            y_smooth,
            label=f"{group_name} (ACI Model)",
            color=color,
            linestyle="-",
            linewidth=2,
        )
        plt.plot(
            test_ages,
            avg_strengths,
            marker,
            markersize=8,
            color=color,
        )

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