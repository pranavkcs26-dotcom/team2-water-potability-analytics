Project 02: Water Quality Potability & Hydro-Chemical Contamination Analyzer
Course Python License

1. Project Overview & Domain Context
Institution: Digital University Kerala (MTech AI / MSc AI)
Course: M4010104 / M4020104 Programming in Python | Instructor: Dr. Sunil TT
Dataset: Water Quality and Potability Dataset on Kaggle
This analytical pipeline audits raw municipal and watershed water telemetry across nine physicochemical attributes (ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, and Turbidity), filters out physical/chemical boundary violations, imputes missing sensor intervals via potability-class-conditional medians, derives an empirical Water Quality Index (WQI), models multi-pollutant relationships, and exports static and interactive visualizations.

2. System Architecture & Pipeline Design
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
