import math


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


def main():
    # 平均強度依序對應 1、3、7、14、28 天；直接以實際齡期作為 te
    ages_days = [1, 3, 7, 14, 28]
    mean_strengths = [130.6667, 173.6667, 229.3333, 320.3333, 353]

    t0 = 1
    fc28 = float(input("請輸入 28 天設計強度 fc28 (MPa)："))
    if fc28 <= 0:
        raise ValueError("28 天設計強度必須大於 0。")

    s_value = fit_s(ages_days, mean_strengths, t0, fc28)

    print(f"28 天基準強度 fc28 = {fc28:.4f} MPa")
    print("te 直接使用實際齡期（天）")
    print(f"反推共同參數 S = {s_value:.6f}")
    print("\n各齡期實測與模型預測：")

    for age, measured in zip(ages_days, mean_strengths):
        factor = 1 - math.sqrt(28 / ((age - 0.5) / t0))
        predicted = math.exp(s_value * factor) * fc28
        print(
            f"{age:>2} 天（te = {age:>2}）："
            f"實測平均 {measured:>9.4f} MPa，"
            f"模型預測 {predicted:>9.4f} MPa"
        )


if __name__ == "__main__":
    main()
