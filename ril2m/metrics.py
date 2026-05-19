# -*- coding: utf-8 -*-
"""
Metrics for the RETORCH annotation cross-validation evaluation.

Aggregation levels
------------------
  run      – one LLM query for a specific fold.
  fold     – average of all runs for one held-out test case.
  SUT      – average of all fold averages for one repository.

Metrics (M1-M8) computed per fold (averaged across runs) and per SUT:

  M1  Correct@N  – proportion of runs where every predicted @AccessMode is
                   syntactically valid (required fields present, valid accessMode value).
  M2  Pass@N     – proportion of runs where every predicted resID exists in
                   SystemResources (no compilation error).
  M3  Acc@N      – proportion of runs where the predicted resource set exactly
                   matches the ground-truth resource set.
  M4  Avg TP     – average resources correctly tagged (across runs and folds).
  M5  Avg TN     – average resources correctly NOT tagged.
  M6  Avg FP     – average resources incorrectly tagged:
        M6.1 Avg FP_real  – exist in SUT but not in GT.
        M6.2 Avg FP_hall  – do not exist anywhere (hallucinations).
  M7  Avg FN     – average resources present in GT but not tagged.
  M8  F1 / F1_Hall – macro-averaged F1; F1_Hall uses only hallucinations as FP.

Output files (in ``outputs/metrics/``):
  metrics_<sut>.csv     – one row per fold (averaged across runs).
  metrics_<sut>_runs.csv – one row per fold × run (individual predictions).
  metrics_<sut>.json    – full per-SUT result dict including fold + run details.
  metrics_summary.csv   – one aggregated row per SUT.
  metrics.xlsx          – Summary + Fold Details + Run Details sheets.

Usage:
    poetry run python ril2m/metrics.py
"""

import csv
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_METRICS_DIR = os.path.join(BASE_DIR, "..", "outputs", "metrics")
CROSSVAL_OUTPUT_BASE = os.path.join(BASE_DIR, "..", "outputs", "crossval")

# Global summary uses normalized names (no @N suffix) so rows from different
# n_runs experiments can coexist in the same sheet/CSV.
_GLOBAL_SUMMARY_FIELDS = [
    "sut", "model", "temperature", "n_folds", "n_runs",
    "M1_correct", "M2_pass", "M3_acc",
    "M4_avg_tp", "M5_avg_tn",
    "M6_avg_fp", "M6_1_avg_fp_real", "M6_2_avg_fp_hall",
    "M7_avg_fn", "M8_f1", "M8_f1_hall",
]

_VALID_ACCESS_MODES = {"READONLY", "READWRITE", "NOACCESS", "DYNAMIC"}
_ACCESS_MODE_RE = re.compile(r"@AccessMode\s*\(([^)]*)\)", re.DOTALL)
_PARAM_RE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|(\d+)|(true|false))')

# Fixed column sets (n_runs-independent)
_FOLD_CSV_FIELDS = [
    "sut", "model", "temperature", "fold_id", "testname", "n_runs",
    "correct", "valid", "accurate",
    "tp", "tn", "fp", "fp_real", "fp_hall", "fn",
    "precision", "recall", "f1", "precision_hall", "f1_hall",
]
_RUN_CSV_FIELDS = [
    "sut", "model", "temperature", "fold_id", "testname", "run_id", "seed",
    "correct", "valid", "accurate",
    "tp", "tn", "fp", "fp_real", "fp_hall", "fn",
    "precision", "recall", "f1", "precision_hall", "f1_hall",
]


# ── parsing ───────────────────────────────────────────────────────────────────

def _to_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_access_modes(text: str) -> list[dict]:
    """Extract all @AccessMode annotations from *text*.

    Returns a list of dicts: resID, accessMode, concurrency, sharing, raw.
    """
    if not text:
        return []
    results = []
    for m in _ACCESS_MODE_RE.finditer(text):
        body = m.group(1)
        params = {
            k: (v_str or v_int or v_bool)
            for k, v_str, v_int, v_bool in _PARAM_RE.findall(body)
        }
        concurrency_raw = params.get("concurrency") if "concurrency" in params else None
        concurrency = _to_int(concurrency_raw) if concurrency_raw is not None else None
        if concurrency_raw is not None and concurrency is None:
            logger.warning(
                "parse_access_modes: non-integer concurrency value %r — kept in concurrency_raw",
                concurrency_raw,
            )
        results.append(
            {
                "resID": params.get("resID"),
                "accessMode": params.get("accessMode"),
                "concurrency": concurrency,
                "concurrency_raw": concurrency_raw,
                "sharing": params.get("sharing") == "true" if "sharing" in params else None,
                "raw": m.group(0),
            }
        )
    return results


# ── per-run metrics ───────────────────────────────────────────────────────────

def _is_syntactically_correct(ann: dict) -> bool:
    return (
        bool(ann.get("resID"))
        and ann.get("accessMode") in _VALID_ACCESS_MODES
        and ann.get("concurrency") is not None
    )


def compute_fold_metrics(
    gt_tc: dict,
    prediction_text: str,
    system_resource_ids: set[str],
) -> dict:
    """Compute M1-M8 metrics for a single run of a single fold.

    Parameters
    ----------
    gt_tc:
        Ground-truth test-case dict (id, testname, annotations).
    prediction_text:
        Raw LLM output for this run.
    system_resource_ids:
        Resource IDs from the SUT's SystemResources JSON.

    Returns
    -------
    Dict with: correct, valid, accurate, tp, tn, fp, fp_real, fp_hall, fn,
    precision, recall, f1, precision_hall, f1_hall.
    """
    gt_anns = parse_access_modes(gt_tc.get("annotations", ""))
    pred_anns = parse_access_modes(prediction_text)

    gt_res: set[str] = {a["resID"] for a in gt_anns if a.get("resID")}
    pred_res: set[str] = {a["resID"] for a in pred_anns if a.get("resID")}

    correct = bool(pred_anns) and all(_is_syntactically_correct(a) for a in pred_anns)
    valid = bool(pred_anns) and all(
        a["resID"] in system_resource_ids for a in pred_anns if a.get("resID")
    )
    accurate = pred_res == gt_res

    tp = gt_res & pred_res
    fn = gt_res - pred_res
    fp = pred_res - gt_res
    tn = (system_resource_ids - gt_res) - pred_res
    fp_real = fp & system_resource_ids
    fp_hall = fp - system_resource_ids

    tp_n, fp_n, fn_n, fp_hall_n = len(tp), len(fp), len(fn), len(fp_hall)

    precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
    recall = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    precision_hall = tp_n / (tp_n + fp_hall_n) if (tp_n + fp_hall_n) > 0 else 0.0
    f1_hall = (
        2 * precision_hall * recall / (precision_hall + recall)
        if (precision_hall + recall) > 0
        else 0.0
    )

    return {
        "correct": correct,
        "valid": valid,
        "accurate": accurate,
        "tp": tp_n,
        "tn": len(tn),
        "fp": fp_n,
        "fp_real": len(fp_real),
        "fp_hall": fp_hall_n,
        "fn": fn_n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_hall": precision_hall,
        "f1_hall": f1_hall,
    }


# ── per-SUT aggregation ───────────────────────────────────────────────────────

def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ── incremental writers ───────────────────────────────────────────────────────

_NUMERIC_KEYS = [
    "correct", "valid", "accurate",
    "tp", "tn", "fp", "fp_real", "fp_hall", "fn",
    "precision", "recall", "f1", "precision_hall", "f1_hall",
]


def load_system_resources(sut: str, context_dir: str) -> set[str]:
    """Load and return resource IDs from ``systemresources_<sut>.json``."""
    sr_path = os.path.join(context_dir, f"systemresources_{sut}.json")
    if not os.path.exists(sr_path):
        logger.warning("SystemResources not found for %s — hallucination detection disabled", sut)
        return set()
    with open(sr_path, encoding="utf-8") as f:
        return set(json.load(f).keys())


def _csv_append_row(path: str, fields: list[str], row: dict) -> None:
    """Append one row to a CSV file, writing the header if the file is new."""
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def append_run_metrics(
    sut: str,
    fold_tc: dict,
    run_id: int,
    seed: int | None,
    prediction_text: str,
    system_resource_ids: set[str],
    output_dir: str,
    model: str = "",
    temperature: float = 0.0,
) -> dict:
    """Compute metrics for one run and immediately append a row to the run CSV.

    Returns the metrics dict (without sut/fold_id/testname prefix keys).
    """
    metrics = compute_fold_metrics(fold_tc, prediction_text, system_resource_ids)
    row = {
        "sut": sut,
        "model": model,
        "temperature": temperature,
        "fold_id": fold_tc["id"],
        "testname": fold_tc["testname"],
        "run_id": run_id,
        "seed": seed,
        **metrics,
    }
    _csv_append_row(os.path.join(output_dir, f"metrics_{sut}_runs.csv"), _RUN_CSV_FIELDS, row)
    return metrics


def append_fold_metrics(
    sut: str,
    fold_tc: dict,
    run_metrics_list: list[dict],
    output_dir: str,
    model: str = "",
    temperature: float = 0.0,
) -> dict:
    """Average run metrics into a fold entry, append to fold CSV, and update Excel.

    *run_metrics_list* — list of dicts, each the return value of ``append_run_metrics``
    plus ``run_id`` and ``seed``.

    Returns the fold-level averaged dict (with ``runs`` key for individual runs).
    """
    fold_avg = {k: _avg([float(r[k]) for r in run_metrics_list]) for k in _NUMERIC_KEYS}
    fold_entry = {
        "fold_id": fold_tc["id"],
        "testname": fold_tc["testname"],
        "n_runs": len(run_metrics_list),
        "model": model,
        "temperature": temperature,
        **fold_avg,
        "runs": run_metrics_list,
    }

    _csv_append_row(
        os.path.join(output_dir, f"metrics_{sut}.csv"),
        _FOLD_CSV_FIELDS,
        {"sut": sut, **fold_entry},
    )

    # Update Excel incrementally (Fold Details + Run Details sheets)
    from ril2m.helpers.excel_utils import append_fold_to_excel, init_metrics_excel  # noqa: PLC0415

    excel_path = os.path.join(output_dir, "metrics.xlsx")
    if not os.path.exists(excel_path):
        init_metrics_excel(
            excel_path,
            _summary_fields(len(run_metrics_list)),
            _FOLD_CSV_FIELDS,
            _RUN_CSV_FIELDS,
        )
    append_fold_to_excel(excel_path, sut, fold_entry, _FOLD_CSV_FIELDS, _RUN_CSV_FIELDS)

    return fold_entry


def aggregate_sut_metrics(
    sut: str,
    fold_details: list[dict],
    model: str = "",
    temperature: float = 0.0,
) -> dict:
    """Build the SUT-level summary dict from an accumulated list of fold entries."""
    n_folds = len(fold_details)
    n_runs = fold_details[0]["n_runs"] if fold_details else 0
    return {
        "sut": sut,
        "model": model,
        "temperature": temperature,
        "n_folds": n_folds,
        "n_runs": n_runs,
        f"M1_correct_at_{n_runs}": _avg([d["correct"] for d in fold_details]),
        f"M2_pass_at_{n_runs}": _avg([d["valid"] for d in fold_details]),
        f"M3_acc_at_{n_runs}": _avg([d["accurate"] for d in fold_details]),
        "M4_avg_tp": _avg([d["tp"] for d in fold_details]),
        "M5_avg_tn": _avg([d["tn"] for d in fold_details]),
        "M6_avg_fp": _avg([d["fp"] for d in fold_details]),
        "M6_1_avg_fp_real": _avg([d["fp_real"] for d in fold_details]),
        "M6_2_avg_fp_hall": _avg([d["fp_hall"] for d in fold_details]),
        "M7_avg_fn": _avg([d["fn"] for d in fold_details]),
        "M8_f1": _avg([d["f1"] for d in fold_details]),
        "M8_f1_hall": _avg([d["f1_hall"] for d in fold_details]),
        "fold_details": fold_details,
    }


def flush_sut_summary(sut_results: list[dict], output_dir: str) -> None:
    """Rewrite summary CSV and update the Summary sheet in Excel.

    Called after each SUT completes so the summary always reflects finished SUTs.
    """
    valid = [r for r in sut_results if r.get("n_folds", 0) > 0]
    if not valid:
        return

    n_runs = valid[0]["n_runs"]
    fields = _summary_fields(n_runs)

    # Rewrite summary CSV
    path = os.path.join(output_dir, "metrics_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(valid)

    # Rebuild Summary sheet in per-experiment Excel
    from ril2m.helpers.excel_utils import update_summary_in_excel  # noqa: PLC0415

    excel_path = os.path.join(output_dir, "metrics.xlsx")
    if os.path.exists(excel_path):
        update_summary_in_excel(excel_path, valid, fields)


# ── global (cross-experiment) incremental writers ─────────────────────────────

_GLOBAL_EXCEL_NAME = "metrics_global.xlsx"
_GLOBAL_SUMMARY_CSV_NAME = "metrics_global_summary.csv"


def _to_global_summary_row(result: dict) -> dict:
    """Normalize a per-SUT result dict to global summary fields (no @N suffixes)."""
    n = result.get("n_runs", 1)
    return {
        "sut": result.get("sut", ""),
        "model": result.get("model", ""),
        "temperature": result.get("temperature", 0.0),
        "n_folds": result.get("n_folds", 0),
        "n_runs": n,
        "M1_correct": result.get(f"M1_correct_at_{n}", 0.0),
        "M2_pass": result.get(f"M2_pass_at_{n}", 0.0),
        "M3_acc": result.get(f"M3_acc_at_{n}", 0.0),
        "M4_avg_tp": result.get("M4_avg_tp", 0.0),
        "M5_avg_tn": result.get("M5_avg_tn", 0.0),
        "M6_avg_fp": result.get("M6_avg_fp", 0.0),
        "M6_1_avg_fp_real": result.get("M6_1_avg_fp_real", 0.0),
        "M6_2_avg_fp_hall": result.get("M6_2_avg_fp_hall", 0.0),
        "M7_avg_fn": result.get("M7_avg_fn", 0.0),
        "M8_f1": result.get("M8_f1", 0.0),
        "M8_f1_hall": result.get("M8_f1_hall", 0.0),
    }


def append_fold_to_global(
    sut: str,
    model: str,
    fold_entry: dict,
    global_output_dir: str,
) -> None:
    """Append one fold row to the global Excel (All-Results + model sheet).

    Creates the global Excel with headers if it does not yet exist.
    """
    from ril2m.helpers.excel_utils import (  # noqa: PLC0415
        append_fold_to_global_excel,
        init_global_metrics_excel,
    )

    excel_path = os.path.join(global_output_dir, _GLOBAL_EXCEL_NAME)
    if not os.path.exists(excel_path):
        init_global_metrics_excel(excel_path, _GLOBAL_SUMMARY_FIELDS, _FOLD_CSV_FIELDS)

    model_name = model.replace(":", "-")
    fold_row = {"sut": sut, **fold_entry}
    append_fold_to_global_excel(excel_path, model_name, fold_row, _FOLD_CSV_FIELDS)


def flush_global_summary(new_sut_results: list[dict], global_output_dir: str) -> None:
    """Merge new SUT results into the global summary CSV and Excel Summary sheet.

    Reads any existing global summary rows, updates entries for the provided SUTs
    (keyed by sut + model + temperature), and rewrites.  Safe to call after every
    fold — cancelled runs keep results up to the last completed fold.
    """
    from ril2m.helpers.excel_utils import update_global_summary_in_excel  # noqa: PLC0415

    csv_path = os.path.join(global_output_dir, _GLOBAL_SUMMARY_CSV_NAME)

    # Load existing rows keyed by (sut, model, temperature)
    existing: dict[tuple, dict] = {}
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("sut"), row.get("model"), row.get("temperature"))
                existing[key] = row

    # Merge / overwrite with new results
    for r in new_sut_results:
        if r.get("n_folds", 0) == 0:
            continue
        summary_row = _to_global_summary_row(r)
        key = (summary_row["sut"], summary_row["model"], str(summary_row["temperature"]))
        existing[key] = summary_row

    all_rows = list(existing.values())
    if not all_rows:
        return

    # Rewrite global summary CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_GLOBAL_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # Rebuild Summary sheet in global Excel
    excel_path = os.path.join(global_output_dir, _GLOBAL_EXCEL_NAME)
    if os.path.exists(excel_path):
        update_global_summary_in_excel(excel_path, all_rows, _GLOBAL_SUMMARY_FIELDS)


def _summary_fields(n_runs: int) -> list[str]:
    """Return summary CSV field names with @N suffix on M1-M3."""
    return [
        "sut", "model", "temperature", "n_folds", "n_runs",
        f"M1_correct_at_{n_runs}",
        f"M2_pass_at_{n_runs}",
        f"M3_acc_at_{n_runs}",
        "M4_avg_tp", "M5_avg_tn",
        "M6_avg_fp", "M6_1_avg_fp_real", "M6_2_avg_fp_hall",
        "M7_avg_fn", "M8_f1", "M8_f1_hall",
    ]


def _discover_run_files(pred_dir: str, fold_id: str) -> list[tuple[int, str]]:
    """Return sorted (run_id, full_path) pairs for all run files of *fold_id*.

    Falls back to a single ``TC-XXX.txt`` file for backwards compatibility.
    """
    pattern = re.compile(rf"^{re.escape(fold_id)}_run_(\d+)\.txt$")
    runs = []
    for fname in os.listdir(pred_dir):
        m = pattern.match(fname)
        if m:
            runs.append((int(m.group(1)), os.path.join(pred_dir, fname)))
    if runs:
        return sorted(runs)
    # Fallback: single prediction file from before multi-run was introduced
    single = os.path.join(pred_dir, f"{fold_id}.txt")
    if os.path.exists(single):
        return [(1, single)]
    return []


def compute_sut_metrics(
    sut: str,
    context_dir: str,
    crossval_output_dir: str,
    model: str = "",
    temperature: float = 0.0,
) -> dict:
    """Compute and return aggregated metrics for all folds of *sut*.

    For each fold all run files are discovered automatically.  Metrics are
    averaged first across runs (→ fold average) then across folds (→ SUT average).

    Returns
    -------
    Dict with ``sut``, ``model``, ``temperature``, ``n_folds``, ``n_runs``,
    aggregated M1-M8 values with @N in key names, and ``fold_details`` list.
    Each fold entry has averaged metrics plus a ``runs`` list with per-run dicts.
    """
    gt_path = os.path.join(context_dir, f"ragtestcases_{sut}.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")
    with open(gt_path, encoding="utf-8") as f:
        cases: list[dict] = json.load(f)

    sr_path = os.path.join(context_dir, f"systemresources_{sut}.json")
    if not os.path.exists(sr_path):
        logger.warning("SystemResources not found for %s — hallucination detection disabled", sut)
        system_resource_ids: set[str] = set()
    else:
        with open(sr_path, encoding="utf-8") as f:
            system_resource_ids = set(json.load(f).keys())

    pred_dir = os.path.join(crossval_output_dir, sut)
    fold_details: list[dict] = []
    numeric_keys = [
        "correct", "valid", "accurate",
        "tp", "tn", "fp", "fp_real", "fp_hall", "fn",
        "precision", "recall", "f1", "precision_hall", "f1_hall",
    ]

    for tc in cases:
        fold_id = tc["id"]
        run_files = _discover_run_files(pred_dir, fold_id)
        if not run_files:
            logger.warning("  [%s] No prediction files for fold %s — skipping", sut, fold_id)
            continue

        # Load seed mapping from fold metadata if available
        meta_path = os.path.join(pred_dir, f"{fold_id}.json")
        seed_by_run: dict[int, int] = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                fold_meta = json.load(f)
            seed_by_run = {r["run_id"]: r.get("seed") for r in fold_meta.get("runs", [])}

        run_results: list[dict] = []
        for run_id, run_path in run_files:
            with open(run_path, encoding="utf-8") as f:
                prediction_text = f.read()
            metrics = compute_fold_metrics(tc, prediction_text, system_resource_ids)
            run_entry = {"run_id": run_id, **metrics}
            if seed_by_run.get(run_id) is not None:
                run_entry["seed"] = seed_by_run[run_id]
            run_results.append(run_entry)

        # Average across runs → fold-level entry
        fold_avg = {k: _avg([float(r[k]) for r in run_results]) for k in numeric_keys}
        fold_details.append(
            {
                "fold_id": fold_id,
                "testname": tc["testname"],
                "n_runs": len(run_results),
                **fold_avg,
                "runs": run_results,
            }
        )
        logger.info(
            "  [%s] %s — %d run(s)  Correct@%d=%.2f  Acc@%d=%.2f  F1=%.4f",
            sut, fold_id, len(run_results),
            len(run_results), fold_avg["correct"],
            len(run_results), fold_avg["accurate"],
            fold_avg["f1"],
        )

    if not fold_details:
        logger.warning("[%s] No fold results found — skipping metrics", sut)
        return {"sut": sut, "model": model, "temperature": temperature, "n_folds": 0, "n_runs": 0}

    return aggregate_sut_metrics(sut, fold_details, model=model, temperature=temperature)


# ── output helpers ────────────────────────────────────────────────────────────

def _write_fold_csv(sut: str, fold_details: list[dict], out_dir: str) -> str:
    path = os.path.join(out_dir, f"metrics_{sut}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FOLD_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in fold_details:
            writer.writerow({"sut": sut, **row})
    return path


def _write_run_csv(sut: str, fold_details: list[dict], out_dir: str) -> str:
    path = os.path.join(out_dir, f"metrics_{sut}_runs.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RUN_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for fold in fold_details:
            for run in fold.get("runs", []):
                writer.writerow(
                    {
                        "sut": sut,
                        "fold_id": fold["fold_id"],
                        "testname": fold["testname"],
                        **run,
                    }
                )
    return path


def _write_summary_csv(all_results: list[dict], out_dir: str) -> str:
    valid = [r for r in all_results if r.get("n_folds", 0) > 0]
    fields = _summary_fields(valid[0]["n_runs"] if valid else 1)
    path = os.path.join(out_dir, "metrics_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(valid)
    return path


def _write_excel(all_results: list[dict], out_dir: str) -> str:
    from ril2m.helpers import write_metrics_excel  # noqa: PLC0415

    valid = [r for r in all_results if r.get("n_folds", 0) > 0]
    n_runs = valid[0]["n_runs"] if valid else 1
    path = os.path.join(out_dir, "metrics.xlsx")
    write_metrics_excel(
        all_results,
        path,
        summary_fields=_summary_fields(n_runs),
        fold_fields=_FOLD_CSV_FIELDS,
        run_fields=_RUN_CSV_FIELDS,
    )
    return path


def _log_summary(all_results: list[dict]) -> None:
    valid = [r for r in all_results if r.get("n_folds", 0) > 0]
    if not valid:
        return
    n = valid[0]["n_runs"]
    cols = [f"M1@{n}", f"M2@{n}", f"M3@{n}", "M4 TP", "M5 TN",
            "M6 FP", "M6.1 FPr", "M6.2 Hall", "M7 FN", "M8 F1", "M8 F1h"]
    sep = "─" * 100
    logger.info(sep)
    logger.info("  METRICS SUMMARY  (n_runs=%d)", n)
    logger.info(sep)
    logger.info("  %-16s  %5s  %s", "SUT", "Folds", "  ".join(f"{c:>8}" for c in cols))
    logger.info("  %-16s  %5s  %s", "─" * 16, "─" * 5, "  ".join("─" * 8 for _ in cols))
    for r in valid:
        vals = [
            r[f"M1_correct_at_{n}"], r[f"M2_pass_at_{n}"], r[f"M3_acc_at_{n}"],
            r["M4_avg_tp"], r["M5_avg_tn"], r["M6_avg_fp"],
            r["M6_1_avg_fp_real"], r["M6_2_avg_fp_hall"],
            r["M7_avg_fn"], r["M8_f1"], r["M8_f1_hall"],
        ]
        logger.info(
            "  %-16s  %5d  %s",
            r["sut"], r["n_folds"], "  ".join(f"{v:8.4f}" for v in vals),
        )
    logger.info(sep)


# ── entry point ───────────────────────────────────────────────────────────────

_EXPERIMENT_DESCRIPTOR = ".experiment.json"


def _read_experiment_descriptor(experiment_dir: str) -> dict:
    """Read model/temperature from the `.experiment.json` written by crossvalidation.py."""
    path = os.path.join(experiment_dir, _EXPERIMENT_DESCRIPTOR)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def compute_all_metrics(
    context_dir: str | None = None,
    crossval_output_base: str = CROSSVAL_OUTPUT_BASE,
    output_dir: str = OUTPUT_METRICS_DIR,
) -> list[dict]:
    """Compute M1-M8 for every experiment × SUT, write CSV/JSON/Excel, and log summary.

    Directory structure expected:
        crossval_output_base/<experiment_tag>/<sut>/TC-XXX_run_NN.txt

    Each experiment directory may contain a ``.experiment.json`` with ``model``
    and ``temperature`` keys written by ``crossvalidation.py``.
    """
    from ril2m.core import CONTEXTS  # noqa: PLC0415

    if context_dir is None:
        context_dir = CONTEXTS

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(crossval_output_base):
        logger.error("Cross-validation output directory not found: %s", crossval_output_base)
        return []

    # Walk: experiment_tag → sut
    all_results: list[dict] = []
    for experiment_tag in sorted(os.listdir(crossval_output_base)):
        exp_dir = os.path.join(crossval_output_base, experiment_tag)
        if not os.path.isdir(exp_dir):
            continue

        descriptor = _read_experiment_descriptor(exp_dir)
        model = descriptor.get("model", "")
        temperature = descriptor.get("temperature", 0.0)
        exp_output_dir = os.path.join(output_dir, experiment_tag)
        os.makedirs(exp_output_dir, exist_ok=True)

        suts = [
            d for d in sorted(os.listdir(exp_dir))
            if os.path.isdir(os.path.join(exp_dir, d)) and not d.startswith(".")
        ]
        logger.info("Experiment: %s  model=%s  temperature=%s  SUTs=%s",
                    experiment_tag, model, temperature, suts)

        for sut in suts:
            logger.info("── %s / %s ──────────────────────────────────────────", experiment_tag, sut)
            result = compute_sut_metrics(sut, context_dir, exp_dir, model=model, temperature=temperature)
            all_results.append(result)

            if result.get("n_folds", 0) > 0:
                csv_path = _write_fold_csv(sut, result["fold_details"], exp_output_dir)
                run_csv_path = _write_run_csv(sut, result["fold_details"], exp_output_dir)
                logger.info("  Fold CSV  → %s", csv_path)
                logger.info("  Run CSV   → %s", run_csv_path)
                json_path = os.path.join(exp_output_dir, f"metrics_{sut}.json")
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(result, jf, indent=2)
                logger.info("  JSON      → %s", json_path)

        # Per-experiment summary and Excel
        exp_results = [r for r in all_results if r.get("model") == model
                       and r.get("temperature") == temperature]
        summary_path = _write_summary_csv(exp_results, exp_output_dir)
        excel_path = _write_excel(exp_results, exp_output_dir)
        logger.info("Summary CSV  → %s", summary_path)
        logger.info("Excel report → %s", excel_path)

    # Global Excel: Summary + All Results + per-model sheets
    from ril2m.helpers.excel_utils import (  # noqa: PLC0415
        append_fold_to_global_excel,
        init_global_metrics_excel,
        update_global_summary_in_excel,
    )

    global_excel = os.path.join(output_dir, _GLOBAL_EXCEL_NAME)
    init_global_metrics_excel(global_excel, _GLOBAL_SUMMARY_FIELDS, _FOLD_CSV_FIELDS)
    for r in all_results:
        model_name = r.get("model", "").replace(":", "-")
        for fold in r.get("fold_details", []):
            append_fold_to_global_excel(global_excel, model_name,
                                        {"sut": r["sut"], **fold}, _FOLD_CSV_FIELDS)
    global_summary_rows = [_to_global_summary_row(r)
                           for r in all_results if r.get("n_folds", 0) > 0]
    update_global_summary_in_excel(global_excel, global_summary_rows, _GLOBAL_SUMMARY_FIELDS)

    global_csv = os.path.join(output_dir, _GLOBAL_SUMMARY_CSV_NAME)
    with open(global_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_GLOBAL_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(global_summary_rows)

    logger.info("Global Excel → %s", global_excel)
    logger.info("Global CSV   → %s", global_csv)
    _log_summary(all_results)
    return all_results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, ".."))
    from ril2m.helpers.logging_config import setup_logging

    setup_logging()
    compute_all_metrics()
