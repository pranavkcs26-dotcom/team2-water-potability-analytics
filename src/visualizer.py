import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

FEATURE_COLS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]


def plot_who_envelope_scatter(df_clean, output_path):
    """
    q10: plots ph vs solids with matplotlib, shades the who safe ph zone (6.5-8.5)
    using axvspan, overlays regression envelope, and distinguishes potability classes.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # shade who safe drinking ph band
    ax.axvspan(6.5, 8.5, color="#2ecc71", alpha=0.14, label="WHO Safe pH Zone (6.5 - 8.5)")
    ax.axvline(6.5, color="#27ae60", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(8.5, color="#27ae60", linestyle="--", linewidth=1.0, alpha=0.7)

    non_pot = df_clean[df_clean["Potability"] == 0]
    pot = df_clean[df_clean["Potability"] == 1]

    ax.scatter(
        non_pot["ph"], non_pot["Solids"],
        c="#e74c3c", alpha=0.45, s=22, edgecolors="none",
        label=f"Non-Potable (n={len(non_pot)})"
    )
    ax.scatter(
        pot["ph"], pot["Solids"],
        c="#2980b9", alpha=0.55, s=24, edgecolors="none",
        label=f"Potable (n={len(pot)})"
    )

    # linear trendline and +/- 1 std confidence envelope across ph
    x_vals = df_clean["ph"].to_numpy()
    y_vals = df_clean["Solids"].to_numpy()
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    x_grid = np.linspace(x_vals.min(), x_vals.max(), 200)
    y_trend = slope * x_grid + intercept
    resid_std = np.std(y_vals - (slope * x_vals + intercept))

    ax.plot(x_grid, y_trend, color="#2c3e50", linewidth=1.8, label="OLS Linear Trend")
    ax.fill_between(
        x_grid, y_trend - resid_std, y_trend + resid_std,
        color="#34495e", alpha=0.10, label="±1 SD Envelope"
    )

    ax.set_title("pH vs. Total Dissolved Solids (TDS) with WHO Safe Envelope", fontsize=13, fontweight="bold")
    ax.set_xlabel("pH Level (0 - 14 scale)", fontsize=11)
    ax.set_ylabel("Total Dissolved Solids / Solids (ppm)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_correlation_heatmap(df_clean, output_path):
    """
    q11a: generates a full 9-parameter annotated correlation heatmap with seaborn.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    corr_matrix = df_clean[FEATURE_COLS].corr()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.3,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        ax=ax
    )
    ax.set_title("Hydro-Chemical Multi-Pollutant Correlation Matrix (9 Parameters)", fontsize=13, fontweight="bold", pad=14)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_distribution_regimes(df_clean, output_path):
    """
    q11b: creates side-by-side seaborn violin plots comparing sulfate and hardness
    distributions across potability classes.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_df = df_clean.copy()
    plot_df["Status"] = plot_df["Potability"].map({0: "Non-Potable (0)", 1: "Potable (1)"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)
    palette = {"Non-Potable (0)": "#e67e22", "Potable (1)": "#1abc9c"}

    sns.violinplot(
        data=plot_df, x="Status", y="Sulfate",
        hue="Status", palette=palette, inner="quartile",
        legend=False, ax=axes[0]
    )
    axes[0].set_title("Sulfate Concentration by Potability Class", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Sulfate (mg/L)", fontsize=11)

    sns.violinplot(
        data=plot_df, x="Status", y="Hardness",
        hue="Status", palette=palette, inner="quartile",
        legend=False, ax=axes[1]
    )
    axes[1].set_title("Water Hardness Distribution by Potability Class", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Hardness (mg/L)", fontsize=11)

    sns.despine(fig=fig)
    fig.suptitle("Physicochemical Distribution Regimes Across Potability Classes", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_interactive_dashboard(df_clean, output_html_path):
    """
    q12: builds an interactive plotly html dashboard with hover tooltips showing
    all 9 parameters, wqi score, and potability status, plus a wqi filter slider.
    """
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    dash_df = df_clean.copy()
    dash_df["Potability_Label"] = dash_df["Potability"].map({0: "Non-Potable", 1: "Potable"})
    dash_df["WQI_Band"] = pd.cut(
        dash_df["wqi_score"],
        bins=[0, 40, 45, 50, 55, 100],
        labels=["<40 (Low)", "40-45", "45-50", "50-55", ">55 (High)"]
    ).astype(str)

    hover_cols = {col: ":.2f" for col in FEATURE_COLS}
    hover_cols["wqi_score"] = ":.2f"
    hover_cols["Potability_Label"] = True

    fig = px.scatter_3d(
        dash_df,
        x="ph",
        y="Solids",
        z="Turbidity",
        color="Potability_Label",
        color_discrete_map={"Non-Potable": "#e74c3c", "Potable": "#2980b9"},
        hover_data=hover_cols,
        opacity=0.72,
        title="Interactive Hydro-Chemical Triage Dashboard (pH × Solids × Turbidity)"
    )

    fig.update_traces(marker=dict(size=4))

    # add interactive slider steps to filter samples by minimum wqi threshold
    thresholds = [0, 35, 40, 45, 50, 55]
    steps = []
    for thresh in thresholds:
        mask_0 = (dash_df["Potability"] == 0) & (dash_df["wqi_score"] >= thresh)
        mask_1 = (dash_df["Potability"] == 1) & (dash_df["wqi_score"] >= thresh)

        sub_0 = dash_df[mask_0]
        sub_1 = dash_df[mask_1]

        step = dict(
            method="restyle",
            args=[
                {
                    "x": [sub_0["ph"], sub_1["ph"]],
                    "y": [sub_0["Solids"], sub_1["Solids"]],
                    "z": [sub_0["Turbidity"], sub_1["Turbidity"]],
                }
            ],
            label=f"WQI ≥ {thresh}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Filter Threshold: "},
        pad={"t": 30},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        scene=dict(
            xaxis_title="pH (0-14)",
            yaxis_title="Solids / TDS (ppm)",
            zaxis_title="Turbidity (NTU)"
        ),
        margin=dict(l=0, r=0, b=40, t=50),
        legend_title_text="Water Status"
    )

    fig.write_html(output_html_path, include_plotlyjs="cdn")


def generate_all_visuals(df_clean, report_dir):
    """generates all static figures and the interactive html dashboard."""
    figures_dir = os.path.join(report_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    envelope_path = os.path.join(figures_dir, "primary_envelope.png")
    heatmap_path = os.path.join(figures_dir, "correlation_heatmap.png")
    violin_path = os.path.join(figures_dir, "distribution_regimes.png")
    html_path = os.path.join(report_dir, "interactive_dashboard.html")

    plot_who_envelope_scatter(df_clean, envelope_path)
    plot_correlation_heatmap(df_clean, heatmap_path)
    plot_distribution_regimes(df_clean, violin_path)
    build_interactive_dashboard(df_clean, html_path)

    return {
        "primary_envelope": envelope_path,
        "correlation_heatmap": heatmap_path,
        "distribution_regimes": violin_path,
        "interactive_dashboard": html_path
    }


# standalone test block so member 3 can run and verify visuals independently
if __name__ == "__main__":
    clean_path = "water-potability-analytics/data/processed/clean_water_potability.csv"
    raw_path = "water-potability-analytics/data/raw/water_potability_raw.csv"
    target_csv = clean_path if os.path.exists(clean_path) else raw_path

    df = pd.read_csv(target_csv)
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(df.groupby("Potability")[col].transform("median"))

    mat = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    c_min, c_max = np.min(mat, axis=0), np.max(mat, axis=0)
    df["wqi_score"] = (((mat - c_min) / (c_max - c_min)) @ (np.ones(9) / 9.0)) * 100.0

    outputs = generate_all_visuals(df, "water-potability-analytics/reports")
    print("generated visualization artifacts:", outputs)
