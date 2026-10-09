import math
from statistics import multimode

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm, chi2, t

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Statistical Formula Lab",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f7fbff, #eef4ff);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
h1, h2, h3 {
    color: #163a63;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dbe7f3;
    border-radius: 16px;
    padding: 10px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
}
div[data-testid="stDataFrame"] {
    background: white;
    border-radius: 14px;
    padding: 4px;
}
</style>
""", unsafe_allow_html=True)

st.title("📘 Statistical Perfect Formula Lab")
st.caption("Step-by-step solutions, highlighted statistical tables, and graph-based explanation")

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def parse_number_list(text):
    try:
        nums = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
        return nums
    except Exception:
        return None


def safe_div(a, b):
    return a / b if b != 0 else 0


def plot_line(values, title="Line Plot", annotate=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(1, len(values) + 1))
    ax.plot(x, values, marker="o", linewidth=2)
    if annotate:
        for xi, yi in zip(x, values):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_bar(values, title="Bar Plot", annotate=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(1, len(values) + 1))
    bars = ax.bar(x, values)
    if annotate:
        for b, val in zip(bars, values):
            ax.annotate(f"{val:.2f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha="center",
                        fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig)


def plot_hist(values, title="Histogram"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=min(10, max(3, len(values)//2)))
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig)


def plot_scatter(x_vals, y_vals, title="Scatter Plot"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x_vals, y_vals)
    for x, y in zip(x_vals, y_vals):
        ax.annotate(f"({x:.1f},{y:.1f})", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def show_normal_curve(z_value=None, critical_values=None, title="Standard Normal Curve"):
    x = np.linspace(-4, 4, 500)
    y = norm.pdf(x, 0, 1)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Z value")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)

    if z_value is not None:
        x_fill = np.linspace(-4, z_value, 300) if z_value >= 0 else np.linspace(z_value, 0, 300)
        ax.fill_between(x_fill, norm.pdf(x_fill, 0, 1), alpha=0.25)
        ax.axvline(z_value, linestyle="--", linewidth=2, label=f"z = {z_value:.4f}")

    if critical_values is not None:
        for c in critical_values:
            ax.axvline(c, linestyle=":", linewidth=2, label=f"critical = {c:.4f}")
            ax.axvline(-c, linestyle=":", linewidth=2)

    if z_value is not None or critical_values is not None:
        ax.legend(fontsize=8)

    st.pyplot(fig)


def show_t_curve(df_value, t_value=None, critical=None):
    x = np.linspace(-5, 5, 500)
    y = t.pdf(x, df_value)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x, y, linewidth=2)
    ax.set_title(f"t Distribution (df = {df_value})")
    ax.set_xlabel("t value")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)

    if t_value is not None:
        ax.axvline(t_value, linestyle="--", linewidth=2, label=f"t = {t_value:.4f}")
    if critical is not None:
        ax.axvline(critical, linestyle=":", linewidth=2, label=f"critical = {critical:.4f}")
        ax.axvline(-critical, linestyle=":", linewidth=2)

    if t_value is not None or critical is not None:
        ax.legend(fontsize=8)

    st.pyplot(fig)


def show_chi_curve(df_value, chi_value=None, critical=None):
    x = np.linspace(0, max(20, chi2.ppf(0.999, df_value)), 500)
    y = chi2.pdf(x, df_value)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x, y, linewidth=2)
    ax.set_title(f"Chi-Square Distribution (df = {df_value})")
    ax.set_xlabel("Chi-square value")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)

    if chi_value is not None:
        ax.axvline(chi_value, linestyle="--", linewidth=2, label=f"χ² = {chi_value:.4f}")
    if critical is not None:
        ax.axvline(critical, linestyle=":", linewidth=2, label=f"critical = {critical:.4f}")

    if chi_value is not None or critical is not None:
        ax.legend(fontsize=8)

    st.pyplot(fig)


# -------------------------------------------------
# TABLES WITH HIGHLIGHT
# -------------------------------------------------
def create_z_table():
    row_vals = np.round(np.arange(0.0, 3.1, 0.1), 1)
    col_vals = np.round(np.arange(0.00, 0.10, 0.01), 2)

    data = []
    for r in row_vals:
        row = []
        for c in col_vals:
            z = round(r + c, 2)
            row.append(round(norm.cdf(z), 4))
        data.append(row)

    df = pd.DataFrame(data, index=row_vals, columns=col_vals)
    df.index.name = "Z"
    return df


def highlight_z_table(z_input):
    z_abs = abs(round(z_input, 2))
    row_key = round(math.floor(z_abs * 10) / 10, 1)
    col_key = round(z_abs - row_key, 2)

    z_df = create_z_table()

    def style_func(data):
        style = pd.DataFrame("", index=data.index, columns=data.columns)
        if row_key in data.index and col_key in data.columns:
            style.loc[row_key, col_key] = "background-color: yellow; color: black; font-weight: bold;"
        return style

    styled = z_df.style.apply(style_func, axis=None)
    return styled, row_key, col_key, z_df.loc[row_key, col_key] if row_key in z_df.index and col_key in z_df.columns else None


def create_z_critical_table():
    alpha_levels = [0.10, 0.05, 0.025, 0.01, 0.005]
    table = {
        "Alpha": alpha_levels,
        "One-Tailed Critical Z": [round(norm.ppf(1 - a), 4) for a in alpha_levels],
        "Two-Tailed Critical Z": [round(norm.ppf(1 - a/2), 4) for a in alpha_levels],
    }
    return pd.DataFrame(table)


def highlight_z_critical(alpha_value, two_tailed=True):
    df = create_z_critical_table()
    col_name = "Two-Tailed Critical Z" if two_tailed else "One-Tailed Critical Z"

    def style_func(data):
        style = pd.DataFrame("", index=data.index, columns=data.columns)
        match_idx = data.index[data["Alpha"] == alpha_value]
        if len(match_idx) > 0:
            idx = match_idx[0]
            style.loc[idx, :] = "background-color: #fff59d;"
            style.loc[idx, col_name] = "background-color: yellow; color: black; font-weight: bold;"
        return style

    return df.style.apply(style_func, axis=None)


def create_t_table():
    dfs = list(range(1, 31))
    alpha_cols = [0.10, 0.05, 0.025, 0.01]
    data = {"df": dfs}
    for a in alpha_cols:
        data[f"alpha={a}"] = [round(t.ppf(1 - a/2, df), 4) for df in dfs]
    return pd.DataFrame(data)


def highlight_t_table(df_value, alpha_value):
    table = create_t_table()
    col_name = f"alpha={alpha_value}"

    def style_func(data):
        style = pd.DataFrame("", index=data.index, columns=data.columns)
        row_match = data.index[data["df"] == df_value]
        if len(row_match) > 0 and col_name in data.columns:
            idx = row_match[0]
            style.loc[idx, :] = "background-color: #fff59d;"
            style.loc[idx, col_name] = "background-color: yellow; color: black; font-weight: bold;"
        return style

    return table.style.apply(style_func, axis=None)


def create_chi_table():
    dfs = list(range(1, 31))
    alpha_cols = [0.10, 0.05, 0.025, 0.01]
    data = {"df": dfs}
    for a in alpha_cols:
        data[f"alpha={a}"] = [round(chi2.ppf(1 - a, df), 4) for df in dfs]
    return pd.DataFrame(data)


def highlight_chi_table(df_value, alpha_value):
    table = create_chi_table()
    col_name = f"alpha={alpha_value}"

    def style_func(data):
        style = pd.DataFrame("", index=data.index, columns=data.columns)
        row_match = data.index[data["df"] == df_value]
        if len(row_match) > 0 and col_name in data.columns:
            idx = row_match[0]
            style.loc[idx, :] = "background-color: #fff59d;"
            style.loc[idx, col_name] = "background-color: yellow; color: black; font-weight: bold;"
        return style

    return table.style.apply(style_func, axis=None)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("📚 Formula Categories")
category = st.sidebar.selectbox(
    "Choose Category",
    [
        "Descriptive Statistics",
        "Probability and Distributions",
        "Hypothesis Testing",
        "Correlation and Regression",
        "Classification Metrics",
        "Error Metrics",
        "Normalization"
    ]
)

# -------------------------------------------------
# DESCRIPTIVE STATISTICS
# -------------------------------------------------
if category == "Descriptive Statistics":
    formula = st.selectbox(
        "Choose Formula",
        ["Mean", "Median", "Mode", "Range", "Variance", "Standard Deviation", "Quartiles and IQR"]
    )

    data_input = st.text_area("Enter values separated by comma", "10, 20, 30, 40, 50")
    values = parse_number_list(data_input)

    if st.button("Calculate"):
        if not values:
            st.error("Please enter valid numeric values.")
        else:
            left, right = st.columns([1, 1])

            if formula == "Mean":
                total = sum(values)
                n = len(values)
                mean_val = total / n

                with left:
                    st.markdown("### Formula")
                    st.latex(r"\bar{x} = \frac{\sum x}{n}")
                    st.markdown("### Step-by-Step")
                    st.write(f"Sum = {total}")
                    st.write(f"n = {n}")
                    st.success(f"Mean = {mean_val:.4f}")

                with right:
                    plot_bar(values, "Mean - Bar Plot")
                    plot_line(values, "Mean - Line Plot")

            elif formula == "Median":
                sorted_vals = sorted(values)
                n = len(sorted_vals)

                with left:
                    st.markdown("### Step-by-Step")
                    st.write(f"Sorted values = {sorted_vals}")
                    if n % 2 == 1:
                        median_val = sorted_vals[n // 2]
                        st.success(f"Median = {median_val:.4f}")
                    else:
                        median_val = (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
                        st.success(f"Median = {median_val:.4f}")

                with right:
                    plot_bar(sorted_vals, "Median - Sorted Data Plot")

            elif formula == "Mode":
                modes = multimode(values)
                freq_series = pd.Series(values).value_counts().sort_index()

                with left:
                    st.markdown("### Step-by-Step")
                    st.write("Frequency Table")
                    st.dataframe(freq_series.reset_index().rename(columns={"index": "Value", "count": "Frequency"}), use_container_width=True)
                    st.success(f"Mode = {modes}")

                with right:
                    plot_bar(freq_series.tolist(), "Mode - Frequency Plot")

            elif formula == "Range":
                minimum = min(values)
                maximum = max(values)
                range_val = maximum - minimum

                with left:
                    st.latex(r"Range = Maximum - Minimum")
                    st.write(f"Maximum = {maximum}")
                    st.write(f"Minimum = {minimum}")
                    st.success(f"Range = {range_val:.4f}")

                with right:
                    plot_line(values, "Range - Line Plot")

            elif formula == "Variance":
                mean_val = sum(values) / len(values)
                sq_dev = [(x - mean_val) ** 2 for x in values]
                variance_val = sum(sq_dev) / len(values)

                with left:
                    st.latex(r"\sigma^2 = \frac{\sum (x-\bar{x})^2}{n}")
                    step_df = pd.DataFrame({
                        "Value": values,
                        "x - mean": [round(x - mean_val, 4) for x in values],
                        "(x - mean)^2": [round(v, 4) for v in sq_dev]
                    })
                    st.dataframe(step_df, use_container_width=True)
                    st.success(f"Variance = {variance_val:.4f}")

                with right:
                    plot_hist(values, "Variance - Histogram")
                    plot_line(sq_dev, "Squared Deviations Plot")

            elif formula == "Standard Deviation":
                mean_val = sum(values) / len(values)
                sq_dev = [(x - mean_val) ** 2 for x in values]
                variance_val = sum(sq_dev) / len(values)
                sd_val = math.sqrt(variance_val)

                with left:
                    st.latex(r"\sigma = \sqrt{\frac{\sum (x-\bar{x})^2}{n}}")
                    step_df = pd.DataFrame({
                        "Value": values,
                        "x - mean": [round(x - mean_val, 4) for x in values],
                        "(x - mean)^2": [round(v, 4) for v in sq_dev]
                    })
                    st.dataframe(step_df, use_container_width=True)
                    st.write(f"Variance = {variance_val:.4f}")
                    st.success(f"Standard Deviation = {sd_val:.4f}")

                with right:
                    plot_hist(values, "Standard Deviation - Histogram")
                    plot_line(values, "Standard Deviation - Line Plot")

            elif formula == "Quartiles and IQR":
                sorted_vals = sorted(values)
                q1 = np.percentile(sorted_vals, 25)
                q2 = np.percentile(sorted_vals, 50)
                q3 = np.percentile(sorted_vals, 75)
                iqr = q3 - q1

                with left:
                    st.latex(r"IQR = Q_3 - Q_1")
                    st.write(f"Q1 = {q1:.4f}")
                    st.write(f"Q2 = {q2:.4f}")
                    st.write(f"Q3 = {q3:.4f}")
                    st.success(f"IQR = {iqr:.4f}")

                with right:
                    plot_bar(sorted_vals, "Quartiles - Sorted Values Plot")

# -------------------------------------------------
# PROBABILITY AND DISTRIBUTIONS
# -------------------------------------------------
elif category == "Probability and Distributions":
    formula = st.selectbox(
        "Choose Formula",
        ["Basic Probability", "Conditional Probability", "Bayes Theorem", "Z-Score", "Normal Distribution Probability"]
    )

    if formula == "Basic Probability":
        favorable = st.number_input("Favorable outcomes", min_value=0, value=2)
        total = st.number_input("Total outcomes", min_value=1, value=6)

        if st.button("Calculate Probability"):
            p = favorable / total
            left, right = st.columns([1, 1])
            with left:
                st.latex(r"P(A) = \frac{\text{Favorable}}{\text{Total}}")
                st.write(f"P(A) = {favorable}/{total}")
                st.success(f"Probability = {p:.4f}")
            with right:
                plot_bar([favorable, total], "Favorable vs Total")

    elif formula == "Conditional Probability":
        p_ab = st.number_input("P(A ∩ B)", min_value=0.0, value=0.2, step=0.01)
        p_b = st.number_input("P(B)", min_value=0.01, value=0.5, step=0.01)

        if st.button("Calculate Conditional Probability"):
            result = p_ab / p_b
            left, right = st.columns([1, 1])
            with left:
                st.latex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}")
                st.write(f"P(A|B) = {p_ab}/{p_b}")
                st.success(f"P(A|B) = {result:.4f}")
            with right:
                plot_bar([p_ab, p_b], "Conditional Probability Inputs")

    elif formula == "Bayes Theorem":
        p_b_given_a = st.number_input("P(B|A)", min_value=0.0, value=0.8, step=0.01)
        p_a = st.number_input("P(A)", min_value=0.0, value=0.3, step=0.01)
        p_b = st.number_input("P(B)", min_value=0.01, value=0.5, step=0.01)

        if st.button("Calculate Bayes"):
            result = (p_b_given_a * p_a) / p_b
            left, right = st.columns([1, 1])
            with left:
                st.latex(r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}")
                st.write(f"P(A|B) = ({p_b_given_a} × {p_a}) / {p_b}")
                st.success(f"P(A|B) = {result:.4f}")
            with right:
                plot_bar([p_b_given_a, p_a, p_b], "Bayes Inputs")

    elif formula == "Z-Score":
        x = st.number_input("x value", value=75.0)
        mean = st.number_input("Mean (μ)", value=60.0)
        std = st.number_input("Standard deviation (σ)", min_value=0.01, value=10.0)

        if st.button("Calculate Z-Score"):
            z = (x - mean) / std
            prob = norm.cdf(z)
            styled_table, row_key, col_key, table_val = highlight_z_table(z)

            left, right = st.columns([1.15, 1])

            with left:
                c1, c2, c3 = st.columns(3)
                c1.metric("x", f"{x:.4f}")
                c2.metric("Mean", f"{mean:.4f}")
                c3.metric("Std Dev", f"{std:.4f}")

                st.latex(r"z = \frac{x-\mu}{\sigma}")
                st.write(f"z = ({x} - {mean}) / {std}")
                st.success(f"Z-Score = {z:.4f}")
                st.info(f"Cumulative Probability = {prob:.4f}")
                st.info(f"Highlighted table cell: row {row_key}, column {col_key}, value {table_val}")
                show_normal_curve(z_value=z, title="Z-Score on Standard Normal Curve")

            with right:
                st.markdown("### Highlighted Z-Table")
                st.caption("The computed z-value is highlighted in yellow")
                st.dataframe(styled_table, height=520, use_container_width=True)

    elif formula == "Normal Distribution Probability":
        z1 = st.number_input("Lower z value", value=-1.0)
        z2 = st.number_input("Upper z value", value=1.0)

        if st.button("Calculate Normal Probability"):
            p = norm.cdf(z2) - norm.cdf(z1)
            left, right = st.columns([1, 1])
            with left:
                st.latex(r"P(z_1 < Z < z_2) = \Phi(z_2) - \Phi(z_1)")
                st.write(f"Φ({z2}) = {norm.cdf(z2):.4f}")
                st.write(f"Φ({z1}) = {norm.cdf(z1):.4f}")
                st.success(f"Probability = {p:.4f}")
            with right:
                show_normal_curve(z_value=z2, title="Normal Probability Curve")

# -------------------------------------------------
# HYPOTHESIS TESTING
# -------------------------------------------------
elif category == "Hypothesis Testing":
    test_type = st.selectbox(
        "Choose Test",
        ["Z-Test (One Sample)", "t-Test (One Sample)", "Chi-Square Test"]
    )

    if test_type == "Z-Test (One Sample)":
        sample_mean = st.number_input("Sample mean", value=52.0)
        population_mean = st.number_input("Population mean", value=50.0)
        population_std = st.number_input("Population standard deviation", min_value=0.01, value=5.0)
        n = st.number_input("Sample size", min_value=1, value=30)
        alpha = st.selectbox("Significance level", [0.10, 0.05, 0.025, 0.01], index=1)

        if st.button("Calculate Z-Test"):
            se = population_std / math.sqrt(n)
            z_stat = (sample_mean - population_mean) / se
            critical = norm.ppf(1 - alpha/2)
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))
            styled_critical = highlight_z_critical(alpha, two_tailed=True)

            left, right = st.columns([1.15, 1])

            with left:
                m1, m2, m3 = st.columns(3)
                m1.metric("Z Statistic", f"{z_stat:.4f}")
                m2.metric("Critical Z", f"±{critical:.4f}")
                m3.metric("p-value", f"{p_value:.4f}")

                st.latex(r"z = \frac{\bar{x}-\mu}{\sigma/\sqrt{n}}")
                st.write(f"Standard Error = {population_std} / √{n} = {se:.4f}")
                st.write(f"z = ({sample_mean} - {population_mean}) / {se:.4f} = {z_stat:.4f}")

                if abs(z_stat) > critical:
                    st.error("Decision: Reject H₀")
                else:
                    st.success("Decision: Fail to Reject H₀")

                show_normal_curve(z_value=z_stat, critical_values=[critical], title="Z-Test Curve")

            with right:
                st.markdown("### Z Critical Table")
                st.caption("Selected alpha row is highlighted")
                st.dataframe(styled_critical, use_container_width=True, height=260)

                st.markdown("### Z-Score Table")
                styled_table, row_key, col_key, table_val = highlight_z_table(z_stat)
                st.caption(f"Computed z-statistic highlighted: row {row_key}, column {col_key}")
                st.dataframe(styled_table, use_container_width=True, height=360)

    elif test_type == "t-Test (One Sample)":
        sample_mean = st.number_input("Sample mean", value=52.0, key="t_sm")
        population_mean = st.number_input("Population mean", value=50.0, key="t_pm")
        sample_std = st.number_input("Sample standard deviation", min_value=0.01, value=6.0)
        n = st.number_input("Sample size", min_value=2, value=20, key="t_n")
        alpha = st.selectbox("Significance level ", [0.10, 0.05, 0.025, 0.01], index=1)

        if st.button("Calculate t-Test"):
            se = sample_std / math.sqrt(n)
            t_stat = (sample_mean - population_mean) / se
            df_value = n - 1
            critical = t.ppf(1 - alpha/2, df_value)
            p_value = 2 * (1 - t.cdf(abs(t_stat), df_value))
            styled_t = highlight_t_table(df_value, alpha)

            left, right = st.columns([1.15, 1])

            with left:
                m1, m2, m3 = st.columns(3)
                m1.metric("t Statistic", f"{t_stat:.4f}")
                m2.metric("Critical t", f"±{critical:.4f}")
                m3.metric("p-value", f"{p_value:.4f}")

                st.latex(r"t = \frac{\bar{x}-\mu}{s/\sqrt{n}}")
                st.write(f"df = {df_value}")
                st.write(f"Standard Error = {sample_std} / √{n} = {se:.4f}")
                st.write(f"t = ({sample_mean} - {population_mean}) / {se:.4f} = {t_stat:.4f}")

                if abs(t_stat) > critical:
                    st.error("Decision: Reject H₀")
                else:
                    st.success("Decision: Fail to Reject H₀")

                show_t_curve(df_value, t_stat, critical)

            with right:
                st.markdown("### t Critical Table")
                st.caption("Selected df row and alpha column are highlighted")
                st.dataframe(styled_t, use_container_width=True, height=520)

    elif test_type == "Chi-Square Test":
        observed_input = st.text_area("Observed values", "20, 30, 25, 25")
        expected_input = st.text_area("Expected values", "25, 25, 25, 25")
        alpha = st.selectbox("Significance level  ", [0.10, 0.05, 0.025, 0.01], index=1)

        if st.button("Calculate Chi-Square"):
            observed = parse_number_list(observed_input)
            expected = parse_number_list(expected_input)

            if not observed or not expected or len(observed) != len(expected):
                st.error("Observed and Expected values must be valid and equal in length.")
            else:
                chi_terms = [((o - e) ** 2) / e for o, e in zip(observed, expected)]
                chi_stat = sum(chi_terms)
                df_value = len(observed) - 1
                critical = chi2.ppf(1 - alpha, df_value)
                p_value = 1 - chi2.cdf(chi_stat, df_value)
                styled_chi = highlight_chi_table(df_value, alpha)

                left, right = st.columns([1.15, 1])

                with left:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Chi-Square", f"{chi_stat:.4f}")
                    m2.metric("Critical Value", f"{critical:.4f}")
                    m3.metric("p-value", f"{p_value:.4f}")

                    st.latex(r"\chi^2 = \sum \frac{(O-E)^2}{E}")
                    step_df = pd.DataFrame({
                        "Observed (O)": observed,
                        "Expected (E)": expected,
                        "O - E": [round(o - e, 4) for o, e in zip(observed, expected)],
                        "(O-E)^2 / E": [round(v, 4) for v in chi_terms]
                    })
                    st.dataframe(step_df, use_container_width=True)

                    if chi_stat > critical:
                        st.error("Decision: Reject H₀")
                    else:
                        st.success("Decision: Fail to Reject H₀")

                    show_chi_curve(df_value, chi_stat, critical)
                    plot_bar(observed, "Observed Values Plot")

                with right:
                    st.markdown("### Chi-Square Critical Table")
                    st.caption("Selected df row and alpha column are highlighted")
                    st.dataframe(styled_chi, use_container_width=True, height=520)

# -------------------------------------------------
# CORRELATION AND REGRESSION
# -------------------------------------------------
elif category == "Correlation and Regression":
    formula = st.selectbox(
        "Choose Formula",
        ["Pearson Correlation", "Simple Linear Regression"]
    )

    x_input = st.text_area("Enter X values", "1, 2, 3, 4, 5")
    y_input = st.text_area("Enter Y values", "2, 4, 5, 4, 5")
    x_vals = parse_number_list(x_input)
    y_vals = parse_number_list(y_input)

    if st.button("Calculate"):
        if not x_vals or not y_vals or len(x_vals) != len(y_vals):
            st.error("Enter valid X and Y values with equal length.")
        else:
            left, right = st.columns([1, 1])

            if formula == "Pearson Correlation":
                corr = np.corrcoef(x_vals, y_vals)[0, 1]
                with left:
                    st.latex(r"r = \frac{\sum (x-\bar{x})(y-\bar{y})}{\sqrt{\sum (x-\bar{x})^2 \sum (y-\bar{y})^2}}")
                    st.success(f"Pearson Correlation = {corr:.4f}")
                    if corr > 0:
                        st.info("Interpretation: Positive relationship")
                    elif corr < 0:
                        st.info("Interpretation: Negative relationship")
                    else:
                        st.info("Interpretation: No linear relationship")
                with right:
                    plot_scatter(x_vals, y_vals, "Pearson Correlation - Scatter Plot")

            elif formula == "Simple Linear Regression":
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                y_pred = [slope * x + intercept for x in x_vals]

                with left:
                    st.latex(r"y = mx + c")
                    st.write(f"Slope (m) = {slope:.4f}")
                    st.write(f"Intercept (c) = {intercept:.4f}")
                    st.success(f"Regression Equation: y = {slope:.4f}x + {intercept:.4f}")

                with right:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.scatter(x_vals, y_vals, label="Actual")
                    ax.plot(x_vals, y_pred, linewidth=2, label="Regression Line")
                    for x, y in zip(x_vals, y_vals):
                        ax.annotate(f"({x:.1f},{y:.1f})", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
                    ax.set_title("Simple Linear Regression")
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)

# -------------------------------------------------
# CLASSIFICATION METRICS
# -------------------------------------------------
elif category == "Classification Metrics":
    st.subheader("📌 Classification Metrics")
    tp = st.number_input("True Positive", min_value=0, value=50)
    tn = st.number_input("True Negative", min_value=0, value=40)
    fp = st.number_input("False Positive", min_value=0, value=5)
    fn = st.number_input("False Negative", min_value=0, value=5)

    if st.button("Calculate Metrics"):
        accuracy = safe_div(tp + tn, tp + tn + fp + fn)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        left, right = st.columns([1, 1])
        with left:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{accuracy:.4f}")
            c2.metric("Precision", f"{precision:.4f}")
            c3.metric("Recall", f"{recall:.4f}")
            c4.metric("F1 Score", f"{f1:.4f}")

        with right:
            plot_bar([accuracy, precision, recall, f1], "Classification Metrics Plot")

# -------------------------------------------------
# ERROR METRICS
# -------------------------------------------------
elif category == "Error Metrics":
    st.subheader("📌 Error Metrics")
    actual_input = st.text_area("Actual values", "100, 120, 130, 150")
    predicted_input = st.text_area("Predicted values", "90, 125, 128, 145")

    actual = parse_number_list(actual_input)
    predicted = parse_number_list(predicted_input)

    if st.button("Calculate Error Metrics"):
        if not actual or not predicted or len(actual) != len(predicted):
            st.error("Please enter valid Actual and Predicted values of equal length.")
        else:
            actual = np.array(actual)
            predicted = np.array(predicted)

            mae = np.mean(np.abs(actual - predicted))
            mse = np.mean((actual - predicted) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100

            left, right = st.columns([1, 1])

            with left:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MAE", f"{mae:.4f}")
                c2.metric("MSE", f"{mse:.4f}")
                c3.metric("RMSE", f"{rmse:.4f}")
                c4.metric("MAPE", f"{mape:.4f}%")

            with right:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(actual, marker="o", label="Actual")
                ax.plot(predicted, marker="o", label="Predicted")
                for i, (a, p) in enumerate(zip(actual, predicted), start=1):
                    ax.annotate(f"A:{a:.1f}", (i - 1, a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
                    ax.annotate(f"P:{p:.1f}", (i - 1, p), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)
                ax.set_title("Actual vs Predicted")
                ax.set_xlabel("Index")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)

# -------------------------------------------------
# NORMALIZATION
# -------------------------------------------------
elif category == "Normalization":
    formula = st.selectbox("Choose Formula", ["Min-Max Normalization", "Z-Score Normalization"])
    data_input = st.text_area("Enter values separated by comma", "10, 20, 30, 40, 50")
    values = parse_number_list(data_input)

    if st.button("Normalize"):
        if not values:
            st.error("Please enter valid numeric values.")
        else:
            arr = np.array(values)
            left, right = st.columns([1, 1])

            if formula == "Min-Max Normalization":
                min_val = np.min(arr)
                max_val = np.max(arr)
                norm_vals = (arr - min_val) / (max_val - min_val)

                with left:
                    st.latex(r"x' = \frac{x - min}{max - min}")
                    df = pd.DataFrame({"Original": arr, "Normalized": np.round(norm_vals, 4)})
                    st.dataframe(df, use_container_width=True)

                with right:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(arr, marker="o", label="Original")
                    ax.plot(norm_vals, marker="o", label="Normalized")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    ax.set_title("Min-Max Normalization Plot")
                    st.pyplot(fig)

            elif formula == "Z-Score Normalization":
                mean_val = np.mean(arr)
                std_val = np.std(arr)
                norm_vals = (arr - mean_val) / std_val

                with left:
                    st.latex(r"x' = \frac{x - \mu}{\sigma}")
                    df = pd.DataFrame({"Original": arr, "Z-Score Normalized": np.round(norm_vals, 4)})
                    st.dataframe(df, use_container_width=True)

                with right:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(arr, marker="o", label="Original")
                    ax.plot(norm_vals, marker="o", label="Normalized")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    ax.set_title("Z-Score Normalization Plot")
                    st.pyplot(fig)