# -*- coding: utf-8 -*-
"""
Rebuild the global metrics Excel from per-experiment metric files.

Use case
--------
You ran cross-validation in separate sessions — e.g. one model on Monday,
another model on Tuesday, the same model with a new temperature on Wednesday.
Each session produced its own ``outputs/metrics/<experiment_tag>/`` directory
containing ``metrics_<sut>.csv`` (per-fold) and ``metrics_summary.csv``
(per-SUT).  This tool walks all of those directories and rebuilds:

  * ``outputs/metrics/metrics_global.xlsx`` — Summary + All Results + one
    sheet per model.
  * ``outputs/metrics/metrics_global_summary.csv`` — one row per SUT × experiment.

The script does not re-run any LLM queries; it only re-aggregates files
already on disk, so it is safe to run as often as needed.

Usage
-----
    poetry run python ril2m/aggregate_metrics.py
    poetry run python ril2m/aggregate_metrics.py --metrics-dir /path/to/outputs/metrics
"""

import argparse
import csv
import logging
import os

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METRICS_DIR = os.path.join(BASE_DIR, "..", "outputs", "metrics")


# ── readers ───────────────────────────────────────────────────────────────────

def _read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _discover_experiments(metrics_dir: str) -> list[str]:
    """Return sorted experiment tags (subdirectories of *metrics_dir*)."""
    if not os.path.isdir(metrics_dir):
        return []
    return [
        d for d in sorted(os.listdir(metrics_dir))
        if os.path.isdir(os.path.join(metrics_dir, d))
    ]


def _fold_csv_files(exp_dir: str) -> list[str]:
    """Return per-SUT fold CSVs in *exp_dir* (excludes _runs and summary CSVs)."""
    out = []
    for fname in sorted(os.listdir(exp_dir)):
        if (fname.startswith("metrics_") and fname.endswith(".csv")
                and not fname.endswith("_runs.csv") and fname != "metrics_summary.csv"):
            out.append(os.path.join(exp_dir, fname))
    return out


def _summary_row_to_result(row: dict) -> dict:
    """Convert a `metrics_summary.csv` row back into a result-shaped dict."""
    n = int(row.get("n_runs", 1) or 1)
    f = lambda k, default=0.0: float(row.get(k, default) or default)  # noqa: E731
    return {
        "sut": row.get("sut", ""),
        "model": row.get("model", ""),
        "temperature": float(row.get("temperature", 0.0) or 0.0),
        "n_folds": int(row.get("n_folds", 0) or 0),
        "n_runs": n,
        f"M1_correct_at_{n}": f(f"M1_correct_at_{n}"),
        f"M2_pass_at_{n}": f(f"M2_pass_at_{n}"),
        f"M3_acc_at_{n}": f(f"M3_acc_at_{n}"),
        "M4_avg_tp": f("M4_avg_tp"),
        "M5_avg_tn": f("M5_avg_tn"),
        "M6_avg_fp": f("M6_avg_fp"),
        "M6_1_avg_fp_real": f("M6_1_avg_fp_real"),
        "M6_2_avg_fp_hall": f("M6_2_avg_fp_hall"),
        "M7_avg_fn": f("M7_avg_fn"),
        "M8_f1": f("M8_f1"),
        "M8_f1_hall": f("M8_f1_hall"),
    }


# ── public API ────────────────────────────────────────────────────────────────

def rebuild_global_excel(metrics_dir: str = DEFAULT_METRICS_DIR) -> str:
    """Rebuild ``metrics_global.xlsx`` and ``metrics_global_summary.csv`` in *metrics_dir*.

    Returns the path to the rebuilt Excel file.
    """
    from ril2m.metrics import (  # noqa: PLC0415
        _FOLD_CSV_FIELDS,
        _GLOBAL_EXCEL_NAME,
        _GLOBAL_SUMMARY_CSV_NAME,
        _GLOBAL_SUMMARY_FIELDS,
        _to_global_summary_row,
    )
    from ril2m.helpers.excel_utils import (  # noqa: PLC0415
        append_fold_to_global_excel,
        init_global_metrics_excel,
        update_global_summary_in_excel,
    )

    experiments = _discover_experiments(metrics_dir)
    if not experiments:
        logger.warning("No experiment directories found in %s", metrics_dir)
        return ""

    global_excel = os.path.join(metrics_dir, _GLOBAL_EXCEL_NAME)
    if os.path.exists(global_excel):
        os.remove(global_excel)   # rebuild from scratch
    init_global_metrics_excel(global_excel, _GLOBAL_SUMMARY_FIELDS, _FOLD_CSV_FIELDS)

    summary_rows: list[dict] = []
    n_fold_rows = 0

    for exp_tag in experiments:
        exp_dir = os.path.join(metrics_dir, exp_tag)
        logger.info("Aggregating experiment: %s", exp_tag)

        # 1. Per-SUT fold rows → All Results sheet + per-model sheet
        for fold_csv in _fold_csv_files(exp_dir):
            for row in _read_csv(fold_csv):
                model_name = row.get("model", "").replace(":", "-")
                append_fold_to_global_excel(global_excel, model_name, row, _FOLD_CSV_FIELDS)
                n_fold_rows += 1

        # 2. Per-experiment summary → Summary sheet
        summary_csv = os.path.join(exp_dir, "metrics_summary.csv")
        if os.path.exists(summary_csv):
            for row in _read_csv(summary_csv):
                if int(row.get("n_folds", 0) or 0) == 0:
                    continue
                summary_rows.append(_to_global_summary_row(_summary_row_to_result(row)))
        else:
            logger.warning("  (no metrics_summary.csv found in %s)", exp_dir)

    # 3. Write the global summary CSV and update the Summary sheet
    global_csv = os.path.join(metrics_dir, _GLOBAL_SUMMARY_CSV_NAME)
    with open(global_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_GLOBAL_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    update_global_summary_in_excel(global_excel, summary_rows, _GLOBAL_SUMMARY_FIELDS)

    logger.info(
        "Rebuilt %s — %d experiments, %d fold rows, %d summary rows",
        global_excel, len(experiments), n_fold_rows, len(summary_rows),
    )
    logger.info("Rebuilt %s", global_csv)
    return global_excel


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild metrics_global.xlsx from per-experiment metrics directories.",
    )
    parser.add_argument(
        "--metrics-dir",
        default=DEFAULT_METRICS_DIR,
        help="Metrics root directory (default: outputs/metrics).",
    )
    args = parser.parse_args()

    import sys  # noqa: PLC0415
    sys.path.insert(0, os.path.join(BASE_DIR, ".."))
    from ril2m.helpers.logging_config import setup_logging  # noqa: PLC0415
    setup_logging()
    rebuild_global_excel(args.metrics_dir)


if __name__ == "__main__":
    main()
