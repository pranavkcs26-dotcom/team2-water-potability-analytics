import csv
import os
from datetime import datetime
import numpy as np

# physical feasibility bounds (q1)
PH_MIN, PH_MAX = 0.0, 14.0
NON_NEGATIVE_COLS = [
    "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

# who safe drinking limits (useful for flagging & live viva checks)
WHO_PH_MIN, WHO_PH_MAX = 6.5, 8.5
WHO_TURBIDITY_MAX = 5.0

EXPECTED_HEADERS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes",
    "Turbidity", "Potability"
]


def audit_and_clean_csv(input_path, clean_csv_path, log_path, check_who=False):
    """
    reads raw csv line-by-line, checks physical bounds (and optional who limits),
    writes valid rows to clean_csv_path, and logs rejected lines to log_path.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"raw dataset not found at: {input_path}")

    os.makedirs(os.path.dirname(clean_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    total_rows = 0
    valid_rows = []
    rejected_logs = []
    violation_counts = {
        "corrupted_or_non_numeric": 0,
        "ph_out_of_bounds": 0,
        "negative_chemical": 0,
        "invalid_potability": 0,
        "who_limit_exceeded": 0
    }

    with open(input_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("input csv file is empty")

        header = [h.strip() for h in header]

        for line_num, row in enumerate(reader, start=2):
            total_rows += 1

            # check column count mismatch
            if len(row) != len(header):
                violation_counts["corrupted_or_non_numeric"] += 1
                rejected_logs.append(f"Line {line_num}: expected {len(header)} cols, got {len(row)}")
                continue

            record = {}
            parse_failed = False

            # parse each cell defensively; empty strings stay None for later imputation
            for col_name, raw_val in zip(header, row):
                val_str = raw_val.strip()
                if val_str == "" or val_str.lower() in ("na", "nan", "null"):
                    record[col_name] = None
                else:
                    try:
                        record[col_name] = float(val_str)
                    except ValueError:
                        parse_failed = True
                        violation_counts["corrupted_or_non_numeric"] += 1
                        rejected_logs.append(
                            f"Line {line_num}: non-numeric value '{val_str}' in column '{col_name}'"
                        )
                        break

            if parse_failed:
                continue

            # check physical bounds on ph [0, 14]
            ph_val = record.get("ph")
            if ph_val is not None and not (PH_MIN <= ph_val <= PH_MAX):
                violation_counts["ph_out_of_bounds"] += 1
                rejected_logs.append(
                    f"Line {line_num}: ph={ph_val} outside physical range [{PH_MIN}, {PH_MAX}]"
                )
                continue

            # check non-negative chemical concentrations
            neg_col = None
            for col in NON_NEGATIVE_COLS:
                val = record.get(col)
                if val is not None and val < 0:
                    neg_col = (col, val)
                    break

            if neg_col is not None:
                violation_counts["negative_chemical"] += 1
                rejected_logs.append(
                    f"Line {line_num}: negative concentration in {neg_col[0]} ({neg_col[1]})"
                )
                continue

            # check target label (must be 0 or 1)
            pot = record.get("Potability")
            if pot not in (0.0, 1.0):
                violation_counts["invalid_potability"] += 1
                rejected_logs.append(f"Line {line_num}: invalid Potability label '{pot}'")
                continue

            # optional strict who threshold check (handy for live viva filter toggle)
            if check_who:
                turb = record.get("Turbidity")
                ph_bad = (ph_val is not None) and not (WHO_PH_MIN <= ph_val <= WHO_PH_MAX)
                turb_bad = (turb is not None) and (turb > WHO_TURBIDITY_MAX)
                if ph_bad or turb_bad:
                    violation_counts["who_limit_exceeded"] += 1
                    rejected_logs.append(
                        f"Line {line_num}: failed WHO safe limits (ph={ph_val}, Turbidity={turb})"
                    )
                    continue

            record["Potability"] = int(pot)
            valid_rows.append(record)

    # write curated records to clean csv
    with open(clean_csv_path, mode="w", newline="", encoding="utf-8") as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=header)
        writer.writeheader()
        for rec in valid_rows:
            clean_rec = {k: ("" if v is None else v) for k, v in rec.items()}
            writer.writerow(clean_rec)

    # write timestamped ingestion audit report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, mode="w", encoding="utf-8") as log_f:
        log_f.write("=== WATER QUALITY INGESTION AUDIT LOG ===\n")
        log_f.write(f"Timestamp        : {timestamp}\n")
        log_f.write(f"Source File      : {input_path}\n")
        log_f.write(f"WHO Filter Active: {check_who} (pH: [{WHO_PH_MIN}, {WHO_PH_MAX}], Turbidity <= {WHO_TURBIDITY_MAX})\n")
        log_f.write("-" * 45 + "\n")
        log_f.write(f"Total Raw Records : {total_rows}\n")
        log_f.write(f"Valid Clean Rows  : {len(valid_rows)}\n")
        log_f.write(f"Rejected Rows     : {len(rejected_logs)}\n")
        log_f.write("-" * 45 + "\n")
        log_f.write("Violation Breakdown:\n")
        for reason, count in violation_counts.items():
            log_f.write(f"  - {reason}: {count}\n")
        log_f.write("-" * 45 + "\n")
        log_f.write("Itemized Rejected Records:\n")
        if not rejected_logs:
            log_f.write("  None (all records passed physical boundary checks).\n")
        else:
            for entry in rejected_logs:
                log_f.write(f"  {entry}\n")

    return {
        "total_records": total_rows,
        "valid_records": len(valid_rows),
        "rejected_records": len(rejected_logs),
        "violations": violation_counts
    }


class WaterPotabilityAuditor:
    """
    encapsulated auditor class for loading curated water data,
    computing weighted wqi scores, and summarizing dataset health.
    """

    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or [
            "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
        ]
        self._records = []
        self._matrix = None
        self._potability = None

    def load_from_csv(self, csv_path):
        """loads clean csv records into internal storage and numpy arrays."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"clean csv not found: {csv_path}")

        loaded = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_row = {}
                for col in self.feature_cols:
                    val = row.get(col, "").strip()
                    parsed_row[col] = np.nan if val == "" else float(val)
                parsed_row["Potability"] = int(float(row["Potability"]))
                loaded.append(parsed_row)

        self._records = loaded
        self._matrix = np.array(
            [[r[c] for c in self.feature_cols] for r in loaded],
            dtype=np.float64
        )
        self._potability = np.array([r["Potability"] for r in loaded], dtype=np.int64)
        return len(self._records)

    def compute_wqi(self, weights=None):
        """
        computes min-max normalized water quality index (0 to 100 scale).
        raises ValueError if any weight is negative or length mismatches.
        """
        if self._matrix is None or len(self._records) == 0:
            raise RuntimeError("no data loaded; call load_from_csv() first")

        n_features = len(self.feature_cols)

        if weights is None:
            w = np.ones(n_features, dtype=np.float64) / n_features
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.ndim != 1 or len(w) != n_features:
                raise ValueError(f"weights must be a 1d sequence of length {n_features}")
            if np.any(w < 0):
                raise ValueError("weights cannot contain negative values")
            if np.sum(w) == 0:
                raise ValueError("sum of weights must be greater than zero")
            w = w / np.sum(w)

        # impute nan with column median for clean wqi calculation
        col_medians = np.nanmedian(self._matrix, axis=0)
        filled = np.where(np.isnan(self._matrix), col_medians, self._matrix)

        c_min = np.min(filled, axis=0)
        c_max = np.max(filled, axis=0)
        denom = np.where((c_max - c_min) == 0, 1.0, c_max - c_min)

        norm_matrix = (filled - c_min) / denom
        wqi_scores = (norm_matrix @ w) * 100.0
        return wqi_scores

    def summary_report(self):
        """returns a dictionary summarizing loaded rows, missing counts, and potability split."""
        if self._matrix is None:
            raise RuntimeError("no data loaded; call load_from_csv() first")

        missing_per_col = {
            col: int(np.isnan(self._matrix[:, idx]).sum())
            for idx, col in enumerate(self.feature_cols)
        }
        potable_count = int(np.sum(self._potability == 1))
        non_potable_count = int(np.sum(self._potability == 0))

        return {
            "total_samples": len(self._records),
            "potable_samples": potable_count,
            "non_potable_samples": non_potable_count,
            "potable_ratio": round(potable_count / max(len(self._records), 1), 4),
            "missing_values": missing_per_col
        }


# standalone test block so member 1 can run and verify ingestion independently
if __name__ == "__main__":
    raw_file = "water-potability-analytics/data/raw/water_potability_raw.csv"
    clean_file = "water-potability-analytics/data/processed/clean_water_potability.csv"
    log_file = "water-potability-analytics/data/processed/water_ingestion_audit.log"

    stats = audit_and_clean_csv(raw_file, clean_file, log_file, check_who=False)
    print("ingestion audit stats:", stats)

    auditor = WaterPotabilityAuditor()
    n_loaded = auditor.load_from_csv(clean_file)
    wqi = auditor.compute_wqi()
    print(f"loaded {n_loaded} rows | mean wqi: {wqi.mean():.2f}")
    print("summary report:", auditor.summary_report())
