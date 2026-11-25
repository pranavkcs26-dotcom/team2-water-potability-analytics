import argparse
import os
import sys

from src.ingestion import audit_and_clean_csv, WaterPotabilityAuditor
from src.analyzer import run_full_analysis
from src.visualizer import generate_all_visuals


def parse_args():
    parser = argparse.ArgumentParser(
        description="Water Quality Potability & Hydro-Chemical Contamination Analyzer"
    )
    parser.add_argument(
        "--input",
        default="data/raw/water_potability_raw.csv",
        help="path to raw input csv dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="directory for clean csv and audit log"
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="directory for static figures and interactive html dashboard"
    )
    parser.add_argument(
        "--strict-who",
        action="store_true",
        help="enforce strict who ph [6.5, 8.5] and turbidity <= 5.0 filtering during ingestion"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    clean_csv_path = os.path.join(args.output_dir, "clean_water_potability.csv")
    audit_log_path = os.path.join(args.output_dir, "water_ingestion_audit.log")

    print("[1/4] starting raw data ingestion & physical bounds audit...")
    try:
        ingest_stats = audit_and_clean_csv(
            input_path=args.input,
            clean_csv_path=clean_csv_path,
            log_path=audit_log_path,
            check_who=args.strict_who
        )
    except Exception as exc:
        print(f"error during ingestion: {exc}", file=sys.stderr)
        return 1

    print(
        f"      total={ingest_stats['total_records']} | "
        f"valid={ingest_stats['valid_records']} | "
        f"rejected={ingest_stats['rejected_records']}"
    )

    print("[2/4] initializing WaterPotabilityAuditor & computing baseline wqi...")
    auditor = WaterPotabilityAuditor()
    auditor.load_from_csv(clean_csv_path)
    base_wqi = auditor.compute_wqi()
    summary = auditor.summary_report()
    print(
        f"      potable={summary['potable_samples']} ({summary['potable_ratio']*100:.2f}%) | "
        f"mean wqi={base_wqi.mean():.2f}"
    )

    print("[3/4] running numpy vectorization, anomaly masking, ols regression & pandas pivots...")
    analysis = run_full_analysis(clean_csv_path)
    print(
        f"      allclose verified={analysis['wqi']['allclose_verified']} | "
        f"who violations={analysis['anomalies']['who_violation_count']} | "
        f"z-outliers={analysis['anomalies']['zscore_outlier_count']} | "
        f"overlap={analysis['anomalies']['intersection_count']}"
    )
    print(
        f"      ols solids r2={analysis['ols']['r2']} | "
        f"pearson r={analysis['ols']['pearson_r']}"
    )

    print("[4/4] exporting matplotlib, seaborn & interactive plotly artifacts...")
    artifacts = generate_all_visuals(analysis["df_clean"], args.report_dir)
    for name, path in artifacts.items():
        print(f"      saved {name} -> {path}")

    print("pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
