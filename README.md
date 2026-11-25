# Project 02: Water Quality Potability & Hydro-Chemical Contamination Analyzer

[![Course](https://img.shields.io/badge/Course-Programming%20in%20Python-blue.svg)](https://duk.ac.in)
[![Python](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 1. Project Overview & Domain Context
* **Institution:** Digital University Kerala (MTech AI / MSc AI)
* **Course:** M4010104 / M4020104 Programming in Python | **Instructor:** Dr. Sunil TT
* **Dataset:** [Water Quality and Potability Dataset on Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

This analytical pipeline audits raw municipal and watershed water telemetry across nine physicochemical attributes (`ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, and `Turbidity`), filters out physical/chemical boundary violations, imputes missing sensor intervals via potability-class-conditional medians, derives an empirical Water Quality Index (WQI), models multi-pollutant relationships, and exports static and interactive visualizations.

## 2. System Architecture & Pipeline Design

    +-----------------------------------+
    |  data/raw/water_potability_raw.csv|
    +-----------------+-----------------+
                      |
                      v  (src/ingestion.py: Q1-Q3)
    +-----------------------------------+
    | Line-by-Line Physical Bounds Audit|---> data/processed/water_ingestion_audit.log
    | & WaterPotabilityAuditor Class    |---> data/processed/clean_water_potability.csv
    +-----------------+-----------------+
                      |
                      v  (src/analyzer.py: Q4-Q9)
    +-----------------------------------+
    | NumPy Vectorized WQI (allclose),  |
    | WHO vs Z-Score Set Intersection,  |
    | OLS Plane Fit, Pandas Pivots/Melt |
    +-----------------+-----------------+
                      |
                      v  (src/visualizer.py: Q10-Q12)
    +-----------------------------------+
    | Matplotlib WHO Envelope Scatter   |---> reports/figures/primary_envelope.png
    | Seaborn Heatmap & Violin Regimes  |---> reports/figures/correlation_heatmap.png
    |                                   |---> reports/figures/distribution_regimes.png
    | Plotly 3D Interactive Dashboard   |---> reports/interactive_dashboard.html
    +-----------------------------------+

## 3. Installation & Environment Setup

    git clone [https://github.com/pranavkcs26-dotcom/team2-water-potability-analytics.git](https://github.com/pranavkcs26-dotcom/team2-water-potability-analytics.git)
    cd team2-water-potability-analytics
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt

## 4. Execution Instructions

### A. Running the Standalone Application

    python run_pipeline.py --input data/raw/water_potability_raw.csv --output-dir data/processed --report-dir reports

### B. Running the Development Notebook

    jupyter notebook notebooks/development_cycle.ipynb

## 5. Key Analytical & Statistical Findings (Q1–Q12 Summary)

| Question / Investigation | Key Metric / Analytical Finding |
| :--- | :--- |
| **Q1–Q2: Ingestion & Bounds Audit** | `3,276` raw records parsed; `3,276` valid records passed physical bounds (`pH` in `[0, 14]`, non-negative concentrations); `0` corrupted/out-of-range rows rejected. |
| **Q3: WaterPotabilityAuditor OOP** | Encapsulated CSV loader & weighted WQI engine (`mean WQI = 49.03`); enforces `ValueError` on negative or mismatched feature weights. |
| **Q4: Vectorized NumPy WQI** | Vectorized min-max normalization verified against iterative column computation via `np.allclose() == True` (`mean = 49.0252`, `std = 7.46`). |
| **Q5: WHO vs. Z-Score Anomalies** | `1,610` samples violate WHO limits (`pH` outside `[6.5, 8.5]` or `Turbidity > 5.0`); `148` multivariate outliers (`|z| > 3.0`); exact set intersection = `106` samples. |
| **Q6: Solids OLS Linear Plane** | $\text{Solids} = 23866.24 + 1.3796(\text{Conductivity}) - 12.4263(\text{Hardness})$; $R^2 = 0.002361$, Pearson $r = 0.048592$, residual variance = `76,753,161.30`. |
| **Q7: Missing Value & Imputation** | Missingness found in `Sulfate` (`23.84%`, 781 rows), `ph` (`14.99%`, 491 rows), and `Trihalomethanes` (`4.95%`, 162 rows); class-conditional vs. global median imputation compared. |
| **Q8: pH–Turbidity Risk Pivot** | Neutral pH (`6.5–8.5`) yields the highest mean potability rate (`0.4178`), whereas Alkaline + Turbid (`>8.5`, `>5 NTU`) drops to `0.2308` (overall mean `0.3901`). |
| **Q9: Wide-to-Long Ranking** | Reshaped 29,484 parameter readings via `pd.melt()` and extracted top-5 extreme samples per pollutant (e.g., max `Chloramines = 13.13 ppm`, `Conductivity = 753.34 µS/cm`). |
| **Q10–Q12: Visual Artifacts** | Generated WHO safe-zone scatter envelope, 9×9 annotated correlation heatmap, Sulfate/Hardness potability violin regimes, and interactive 3D Plotly HTML dashboard with WQI slider. |

## 6. Standalone Interactive Visualizations
The standalone interactive Plotly dashboard is generated locally as an offline HTML artifact:
* **File Location:** `reports/interactive_dashboard.html` (viewable directly in any standard web browser without requiring a running Python server).

## 7. Contributor Git Commit & Pull Request Log Table

| Contributor Name | GitHub Handle | Assigned Module / Tasks | Verified Commit Hash Range | Merged Pull Request |
| :--- | :--- | :--- | :--- | :--- |
| Aswathy MS | `aswathycs26-gif` | Package Setup, `src/ingestion.py` (Q1–Q3), `run_pipeline.py` & `README.md` | `initial`..`HEAD` | `#1 (feature/ingestion)` |
| Pranav Krishna S | `@@pranavkcs26-dotcom ` | Statistical & NumPy/Pandas Engine `src/analyzer.py` (Q4–Q9) | `initial`..`HEAD` | `#2 (feature/analysis)` |
| Roshni A N | `@Roshni-A-N` | Visualization Suite `src/visualizer.py` (Q10–Q12) & `notebooks/development_cycle.ipynb` | `initial`..`HEAD` | `#3 (feature/visualization)` |
