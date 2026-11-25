import os
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]


def load_and_prepare_data(csv_path):
    """loads dataset and adds class-conditional median imputed columns."""
    df = pd.read_csv(csv_path)
    df_clean = df.copy()
    for col in FEATURE_COLS:
        df_clean[col] = df_clean[col].fillna(
            df_clean.groupby("Potability")[col].transform("median")
        )
    return df, df_clean


def compute_vectorized_wqi(df_clean):
    """
    q4: computes min-max normalized scores and empirical wqi using raw numpy arrays,
    verified against a manual loop/reference array using np.allclose().
    """
    raw_matrix = df_clean[FEATURE_COLS].to_numpy(dtype=np.float64)

    col_min = np.min(raw_matrix, axis=0)
    col_max = np.max(raw_matrix, axis=0)
    col_range = np.where((col_max - col_min) == 0, 1.0, col_max - col_min)

    # vectorized min-max scaling via broadcasting
    norm_matrix = (raw_matrix - col_min) / col_range
    weights = np.ones(len(FEATURE_COLS), dtype=np.float64) / len(FEATURE_COLS)
    wqi_vec = (norm_matrix @ weights) * 100.0

    # independent column-by-column verification
    wqi_ref = np.zeros(raw_matrix.shape[0], dtype=np.float64)
    for j in range(len(FEATURE_COLS)):
        scaled_col = (raw_matrix[:, j] - col_min[j]) / col_range[j]
        wqi_ref += scaled_col * weights[j] * 100.0

    is_verified = bool(np.allclose(wqi_vec, wqi_ref, rtol=1e-9, atol=1e-9))

    return {
        "norm_matrix": norm_matrix,
        "wqi_scores": wqi_vec,
        "allclose_verified": is_verified,
        "mean_wqi": round(float(np.mean(wqi_vec)), 4),
        "std_wqi": round(float(np.std(wqi_vec)), 4)
    }


def detect_who_and_zscore_anomalies(df_clean):
    """
    q5: compares samples violating who safe thresholds (ph outside [6.5, 8.5]
    or turbidity > 5.0) against multivariate z-score outliers (|z| > 3.0).
    """
    ph_arr = df_clean["ph"].to_numpy(dtype=np.float64)
    turb_arr = df_clean["Turbidity"].to_numpy(dtype=np.float64)

    who_mask = (ph_arr < 6.5) | (ph_arr > 8.5) | (turb_arr > 5.0)
    who_indices = set(np.where(who_mask)[0].tolist())

    # multivariate z-score across all 9 chemical features
    feat_matrix = df_clean[FEATURE_COLS].to_numpy(dtype=np.float64)
    means = np.mean(feat_matrix, axis=0)
    stds = np.std(feat_matrix, axis=0)
    z_scores = np.abs((feat_matrix - means) / stds)

    z_mask = np.any(z_scores > 3.0, axis=1)
    zscore_indices = set(np.where(z_mask)[0].tolist())

    overlap_indices = who_indices.intersection(zscore_indices)

    return {
        "who_violation_count": len(who_indices),
        "zscore_outlier_count": len(zscore_indices),
        "intersection_count": len(overlap_indices),
        "intersection_sample_ids": sorted(list(overlap_indices))
    }


def fit_solids_ols_plane(df_clean):
    """
    q6: fits least-squares plane Solids = b0 + b1*Conductivity + b2*Hardness
    via np.linalg.lstsq and computes residual variance, r2, and pearson correlation.
    """
    y = df_clean["Solids"].to_numpy(dtype=np.float64)
    cond = df_clean["Conductivity"].to_numpy(dtype=np.float64)
    hard = df_clean["Hardness"].to_numpy(dtype=np.float64)

    design_matrix = np.column_stack([np.ones(len(y)), cond, hard])
    coeffs, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)

    y_pred = design_matrix @ coeffs
    residuals = y - y_pred

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_score = 1.0 - (ss_res / ss_tot)
    residual_variance = float(np.var(residuals, ddof=3))

    # pearson correlation between actual and predicted solids
    pearson_r = float(np.corrcoef(y, y_pred)[0, 1])

    return {
        "b0_intercept": round(float(coeffs[0]), 4),
        "b1_conductivity": round(float(coeffs[1]), 4),
        "b2_hardness": round(float(coeffs[2]), 4),
        "residual_variance": round(residual_variance, 2),
        "r2": round(r2_score, 6),
        "pearson_r": round(pearson_r, 6)
    }


def audit_missing_and_imputation(df_raw):
    """
    q7: quantifies missing percentages across all 9 parameters and compares
    potability-class-conditional median imputation vs global median imputation.
    """
    total_rows = len(df_raw)
    missing_counts = df_raw[FEATURE_COLS].isna().sum()
    missing_pct = ((missing_counts / total_rows) * 100.0).round(2)

    df_global = df_raw.copy()
    df_class = df_raw.copy()

    comparison_rows = []
    for col in FEATURE_COLS:
        global_med = df_raw[col].median()
        df_global[col] = df_global[col].fillna(global_med)

        class_meds = df_raw.groupby("Potability")[col].transform("median")
        df_class[col] = df_class[col].fillna(class_meds)

        comparison_rows.append({
            "parameter": col,
            "missing_count": int(missing_counts[col]),
            "missing_pct": float(missing_pct[col]),
            "global_imputed_std": round(float(df_global[col].std()), 4),
            "class_imputed_std": round(float(df_class[col].std()), 4),
            "mean_abs_diff": round(float(np.abs(df_global[col] - df_class[col]).mean()), 4)
        })

    return pd.DataFrame(comparison_rows)


def build_risk_pivot_table(df_clean):
    """
    q8: discretizes ph and turbidity into risk tiers and builds a 3x3
    pivot_table(margins=True) of mean potability rate.
    """
    temp = df_clean.copy()
    temp["pH_Regime"] = pd.cut(
        temp["ph"],
        bins=[-np.inf, 6.5, 8.5, np.inf],
        labels=["Acidic (<6.5)", "Neutral (6.5-8.5)", "Alkaline (>8.5)"]
    )
    temp["Turbidity_Tier"] = pd.cut(
        temp["Turbidity"],
        bins=[-np.inf, 3.0, 5.0, np.inf],
        labels=["Clear (<3)", "Moderate (3-5)", "Turbid (>5)"]
    )

    pivot = temp.pivot_table(
        values="Potability",
        index="pH_Regime",
        columns="Turbidity_Tier",
        aggfunc="mean",
        margins=True,
        margins_name="Overall_Mean",
        observed=False
    ).round(4)

    return pivot


def extract_top_concentrations_long(df_clean, top_n=5):
    """
    q9: reshapes the 9 chemical attributes into tidy long format via pd.melt()
    and extracts the top 5 highest concentration samples per parameter.
    """
    temp = df_clean[FEATURE_COLS].copy()
    temp["Sample_ID"] = np.arange(len(temp))

    long_df = pd.melt(
        temp,
        id_vars=["Sample_ID"],
        value_vars=FEATURE_COLS,
        var_name="Chemical_Parameter",
        value_name="Value"
    )

    top_samples = (
        long_df.sort_values(["Chemical_Parameter", "Value"], ascending=[True, False])
        .groupby("Chemical_Parameter", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return long_df, top_samples


def run_full_analysis(csv_path):
    """runs q4-q9 end-to-end and returns a dictionary of results."""
    df_raw, df_clean = load_and_prepare_data(csv_path)
    wqi_res = compute_vectorized_wqi(df_clean)
    df_clean["wqi_score"] = wqi_res["wqi_scores"]

    anomaly_res = detect_who_and_zscore_anomalies(df_clean)
    ols_res = fit_solids_ols_plane(df_clean)
    imputation_df = audit_missing_and_imputation(df_raw)
    risk_pivot = build_risk_pivot_table(df_clean)
    _, top_5_df = extract_top_concentrations_long(df_clean, top_n=5)

    return {
        "df_clean": df_clean,
        "wqi": wqi_res,
        "anomalies": anomaly_res,
        "ols": ols_res,
        "imputation_audit": imputation_df,
        "risk_pivot": risk_pivot,
        "top_5_long": top_5_df
    }


if __name__ == "__main__":
    clean_path = "water-potability-analytics/data/processed/clean_water_potability.csv"
    raw_path = "water-potability-analytics/data/raw/water_potability_raw.csv"
    target_csv = clean_path if os.path.exists(clean_path) else raw_path

    results = run_full_analysis(target_csv)
    print("q4 wqi verified via allclose:", results["wqi"]["allclose_verified"], "| mean:", results["wqi"]["mean_wqi"])
    print("q5 anomalies:", results["anomalies"])
    print("q6 ols fit:", results["ols"])
    print("\nq7 imputation audit:\n", results["imputation_audit"].to_string(index=False))
    print("\nq8 ph-turbidity potability pivot:\n", results["risk_pivot"])
    print("\nq9 top 5 sample preview (first 10 rows):\n", results["top_5_long"].head(10).to_string(index=False))
