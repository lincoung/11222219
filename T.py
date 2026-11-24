import math
from pathlib import Path

import matplotlib.pyplot as plt


def fit_s(ages_days, mean_strengths, t0, fc28):
    """直接使用實際齡期作為 te，以最小平方方法反推單一 S。"""
    if len(ages_days) != len(mean_strengths):
        raise ValueError("齡期數量必須與平均強度數量相同。")
    if t0 <= 0 or fc28 <= 0:
        raise ValueError("t0 與 fc28 必須大於 0。")
    if any(age <= 0.5 for age in ages_days):
        raise ValueError("每個齡期都必須大於 0.5 天。")
    if any(strength <= 0 for strength in mean_strengths):
        raise ValueError("平均強度必須全部大於 0。")

    # ln(fc_te / fc28) = S * [1 - sqrt(28 / ((te - 0.5) / t0))]
    x_values = [
        1 - math.sqrt(28 / ((age - 0.5) / t0))
        for age in ages_days
    ]
    y_values = [
        math.log(strength / fc28)
        for strength in mean_strengths
    ]

    denominator = sum(x * x for x in x_values)
    if denominator == 0:
        raise ValueError("齡期資料無法用來估算 S。")

    return sum(x * y for x, y in zip(x_values, y_values)) / denominator


def predict_strength(age, s_value, t0, fc28):
    """依齡期與反推的 S 計算模型預測強度。"""
    factor = 1 - math.sqrt(28 / ((age - 0.5) / t0))
    return math.exp(s_value * factor) * fc28


def fit_average_logarithmic_regression(ages, strengths):
    """使用最小平方法計算對數迴歸 y = a * ln(x) + b。"""
    if len(ages) != len(strengths):
        raise ValueError("齡期數量必須與平均強度數量相同。")
    if any(age <= 0 for age in ages):
        raise ValueError("齡期必須大於 0。")

    log_ages = [math.log(age) for age in ages]
    mean_log_age = sum(log_ages) / len(log_ages)
    mean_strength = sum(strengths) / len(strengths)
    denominator = sum((log_age - mean_log_age) ** 2 for log_age in log_ages)
    if denominator == 0:
        raise ValueError("齡期資料無法用來建立迴歸線。")

    a_value = sum(
        (log_age - mean_log_age) * (strength - mean_strength)
        for log_age, strength in zip(log_ages, strengths)
    ) / denominator
    b_value = mean_strength - a_value * mean_log_age
    return a_value, b_value


def main():
    # 平均強度依序對應 1、3、7、14、28 天
    ages_days = [1, 3, 7, 14, 28]
    mean_strengths = [130.6667, 173.6667, 229.3333, 320.3333, 353]

    t0 = 1
    fc28 = float(input("請輸入 28 天設計強度 fc28 (MPa)："))
    if fc28 <= 0:
        raise ValueError("28 天設計強度必須大於 0。")

    s_value = fit_s(ages_days, mean_strengths, t0, fc28)
    predicted_strengths = [
        predict_strength(age, s_value, t0, fc28)
        for age in ages_days
    ]
    smooth_ages = [
        ages_days[0] + (ages_days[-1] - ages_days[0]) * index / 499
        for index in range(500)
    ]
    smooth_predictions = [
        predict_strength(age, s_value, t0, fc28)
        for age in smooth_ages
    ]
    regression_a, regression_b = fit_average_logarithmic_regression(
        ages_days, mean_strengths
    )
    average_regression_strengths = [
        regression_a * math.log(age) + regression_b
        for age in smooth_ages
    ]

    print(f"\n28 天設計強度 fc28 = {fc28:.4f} MPa")
    print(f"反推共同參數 S = {s_value:.6f}")
    print(
        f"平均值對數迴歸：y = {regression_a:.6f} "
        f"* ln(x) + {regression_b:.6f}"
    )
    print("\n各齡期實測平均值與模型預測值：")
    for age, measured, predicted in zip(
        ages_days, mean_strengths, predicted_strengths
    ):
        print(
            f"{age:>2} 天：平均值 {measured:>9.4f} MPa，"
            f"預測值 {predicted:>9.4f} MPa"
        )

    plt.figure(figsize=(9, 6))
    plt.plot(
        smooth_ages,
        average_regression_strengths,
        linewidth=2,
        linestyle="--",
        label="Measured average logarithmic regression",
    )
    plt.scatter(
        ages_days,
        mean_strengths,
        marker="o",
        s=55,
        label="Measured average data points",
        zorder=3,
    )
    plt.plot(
        smooth_ages,
        smooth_predictions,
        linewidth=2,
        label="Model prediction (formula)",
    )
    plt.scatter(
        ages_days,
        predicted_strengths,
        marker="s",
        s=45,
        label="Predicted values at test ages",
        zorder=3,
    )

    for age, strength in zip(ages_days, mean_strengths):
        plt.annotate(f"{strength:.1f}", (age, strength), xytext=(0, 8),
                     textcoords="offset points", ha="center")
    for age, strength in zip(ages_days, predicted_strengths):
        plt.annotate(f"{strength:.1f}", (age, strength), xytext=(0, -14),
                     textcoords="offset points", ha="center")

    plt.title("Measured Average vs. Model Prediction (T)")
    plt.xlabel("Age (days)")
    plt.ylabel("Compressive Strength (kgf/cm^2)")
    plt.xticks(ages_days)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path = Path(__file__).with_name("平均值與預測值.png")
    plt.savefig(output_path, dpi=150)
    print(f"\n圖表已儲存至：{output_path}")
    plt.show()


if __name__ == "__main__":
    main()