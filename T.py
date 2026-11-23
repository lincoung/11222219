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


def smooth_average_curve(ages, strengths, points_per_interval=100):
    """建立通過平均值資料點且避免過衝的平滑曲線。"""
    intervals = [x_next - x for x, x_next in zip(ages, ages[1:])]
    slopes = [
        (y_next - y) / interval
        for y, y_next, interval in zip(strengths, strengths[1:], intervals)
    ]
    tangents = [slopes[0]]

    for index in range(1, len(ages) - 1):
        previous_slope = slopes[index - 1]
        next_slope = slopes[index]
        if previous_slope * next_slope <= 0:
            tangents.append(0)
            continue

        previous_interval = intervals[index - 1]
        next_interval = intervals[index]
        weight_previous = 2 * next_interval + previous_interval
        weight_next = next_interval + 2 * previous_interval
        tangents.append(
            (weight_previous + weight_next)
            / (weight_previous / previous_slope + weight_next / next_slope)
        )

    tangents.append(slopes[-1])
    smooth_ages = []
    smooth_strengths = []

    for index, (start_age, end_age) in enumerate(zip(ages, ages[1:])):
        interval = intervals[index]
        for point in range(points_per_interval):
            ratio = point / points_per_interval
            ratio_squared = ratio * ratio
            ratio_cubed = ratio_squared * ratio
            h00 = 2 * ratio_cubed - 3 * ratio_squared + 1
            h10 = ratio_cubed - 2 * ratio_squared + ratio
            h01 = -2 * ratio_cubed + 3 * ratio_squared
            h11 = ratio_cubed - ratio_squared

            smooth_ages.append(start_age + ratio * interval)
            smooth_strengths.append(
                h00 * strengths[index]
                + h10 * interval * tangents[index]
                + h01 * strengths[index + 1]
                + h11 * interval * tangents[index + 1]
            )

    smooth_ages.append(ages[-1])
    smooth_strengths.append(strengths[-1])
    return smooth_ages, smooth_strengths


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
    average_curve_ages, average_curve_strengths = smooth_average_curve(
        ages_days, mean_strengths
    )
    smooth_ages = [
        ages_days[0] + (ages_days[-1] - ages_days[0]) * index / 499
        for index in range(500)
    ]
    smooth_predictions = [
        predict_strength(age, s_value, t0, fc28)
        for age in smooth_ages
    ]

    print(f"\n28 天設計強度 fc28 = {fc28:.4f} MPa")
    print(f"反推共同參數 S = {s_value:.6f}")
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
        average_curve_ages,
        average_curve_strengths,
        linewidth=2,
        label="Measured average (smooth interpolation)",
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