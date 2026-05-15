# -*- coding: utf-8 -*-
"""
Leave-one-out cross-validation for the RETORCH annotation RAG pipeline.

For each SUT discovered in the context directory:
  1. Load the per-repo ragtestcases_<sut>.json.
  2. Build N ChromaDB databases — one per test case, each indexed with the N-1
     remaining cases (leave-one-out).  Already-indexed folds are skipped.
  3. For each fold, query the LLM *n_runs* times via the Jinja2 RAG pipeline to
     predict the @AccessMode annotations of the held-out test case.
  4. Save each run's prediction to ``outputs/crossval/<sut>/TC-XXX_run_<i>.txt``.
  5. Write a manifest.json with fold metadata and all run paths.

Configuration:
    Edit ``ril2m/input/config.json`` to change ``n_runs`` and ``top_k``.

Usage:
    poetry run python ril2m/crossvalidation.py
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_BASE = os.path.join(BASE_DIR, "chroma_db")
OUTPUT_BASE = os.path.join(BASE_DIR, "..", "outputs", "crossval")
_CONFIG_PATH = os.path.join(BASE_DIR, "input", "config.json")

_CONFIG_DEFAULTS = {"n_runs": 10, "top_k": 5, "base_seed": 42}


# ── configuration ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load pipeline parameters from ``ril2m/input/config.json``.

    Falls back to defaults if the file is missing or a key is absent.
    """
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        merged = {**_CONFIG_DEFAULTS, **cfg}
        logger.info("Config loaded from %s: %s", _CONFIG_PATH, merged)
        return merged
    logger.warning("config.json not found — using defaults: %s", _CONFIG_DEFAULTS)
    return dict(_CONFIG_DEFAULTS)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_sut_cases(sut: str, context_dir: str) -> list[dict]:
    """Load and return test cases for *sut* from its ragtestcases_<sut>.json."""
    path = os.path.join(context_dir, f"ragtestcases_{sut}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No test-case file for SUT '{sut}': {path}\n"
            "Run ril2m/input/fetch_testcases.py first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resource_file_for(sut: str, context_dir: str) -> str:
    """Return the SystemResources JSON path for *sut*, falling back to resourcefile.json."""
    sut_path = os.path.join(context_dir, f"systemresources_{sut}.json")
    fallback = os.path.join(context_dir, "resourcefile.json")
    if os.path.exists(sut_path):
        return sut_path
    logger.warning("systemresources_%s.json not found, using resourcefile.json", sut)
    return fallback


def _log_summary(sut: str, folds: list[dict], n_runs: int) -> None:
    sep = "─" * 72
    logger.info(sep)
    logger.info("  CROSS-VALIDATION COMPLETE — %s  (%d runs/fold)", sut, n_runs)
    logger.info(sep)
    logger.info("  %-10s  %-34s  %s", "Fold", "Excluded test case", "Runs saved")
    logger.info("  %-10s  %-34s  %s", "─" * 10, "─" * 34, "─" * 10)
    for fold in folds:
        logger.info(
            "  %-10s  %-34s  %d",
            fold["fold_id"], fold["excluded_testname"], len(fold.get("runs", [])),
        )
    logger.info(sep)


# ── public API ────────────────────────────────────────────────────────────────

def _available_suts(context_dir: str) -> list[str]:
    """Return all SUT short names discovered from ragtestcases_*.json files."""
    return [
        f[len("ragtestcases_"):-len(".json")]
        for f in sorted(os.listdir(context_dir))
        if f.startswith("ragtestcases_") and f.endswith(".json")
    ]


def run_crossvalidation(
    sut: str,
    context_dir: str | None = None,
    chroma_base: str = CHROMA_BASE,
    output_base: str = OUTPUT_BASE,
    n_runs: int | None = None,
    top_k: int | None = None,
    base_seed: int | None = None,
) -> list[dict]:
    """
    Run leave-one-out cross-validation for *sut*.

    For each test case in the SUT:
    - A ChromaDB is built (or reused) with the remaining N-1 cases.
    - The LLM is queried *n_runs* times with a deterministic seed per run
      (``base_seed + run_idx - 1``) so the experiment is reproducible.
    - Each run is saved to disk immediately after it completes.
    - The fold metadata JSON and the SUT manifest are updated after every
      completed run / fold so partial results survive a crash.

    Parameters
    ----------
    sut:        Short repository name (e.g. ``"fullteaching"``).
    context_dir: Directory containing context JSONs.
    chroma_base: Root directory for ChromaDB fold databases.
    output_base: Root directory for prediction outputs.
    n_runs:     LLM queries per fold. Defaults to ``config.json`` value.
    top_k:      Similar examples in the LLM prompt. Defaults to ``config.json`` value.
    base_seed:  Seed for run 1; subsequent runs use ``base_seed + run_idx - 1``.
                Defaults to ``config.json`` value.

    Returns
    -------
    List of fold-info dicts (also written to ``chroma_db/<sut>/manifest.json``).
    """
    from ril2m.core import CONTEXTS, DEFAULT_MODEL, TEMPERATURE, JavaTestRAG  # noqa: PLC0415
    from ril2m.metrics import (  # noqa: PLC0415
        OUTPUT_METRICS_DIR,
        _EXPERIMENT_DESCRIPTOR,
        aggregate_sut_metrics,
        append_fold_metrics,
        append_fold_to_global,
        append_run_metrics,
        flush_global_summary,
        flush_sut_summary,
        load_system_resources,
    )

    cfg = load_config()
    if n_runs is None:
        n_runs = cfg["n_runs"]
    if top_k is None:
        top_k = cfg["top_k"]
    if base_seed is None:
        base_seed = cfg["base_seed"]

    if context_dir is None:
        context_dir = CONTEXTS

    # Build experiment tag: <model-name>_t<temperature>
    model_name = DEFAULT_MODEL.replace(":", "-")
    temperature = TEMPERATURE
    experiment_tag = f"{model_name}_t{temperature}"

    metrics_dir = os.path.join(OUTPUT_METRICS_DIR, experiment_tag)
    os.makedirs(metrics_dir, exist_ok=True)

    cases = _load_sut_cases(sut, context_dir)
    resource_file = _resource_file_for(sut, context_dir)
    system_resource_ids = load_system_resources(sut, context_dir)
    n_folds = len(cases)
    logger.info(
        "[%s] Starting cross-validation — %d fold(s) × %d run(s)  base_seed=%d",
        sut, n_folds, n_runs, base_seed,
    )

    if n_folds < 2:
        logger.warning(
            "[%s] Skipping — leave-one-out requires at least 2 test cases (found %d).",
            sut, n_folds,
        )
        return []

    repo_dir = os.path.join(chroma_base, experiment_tag, sut)
    out_dir = os.path.join(output_base, experiment_tag, sut)
    exp_base = os.path.join(output_base, experiment_tag)
    manifest_path = os.path.join(repo_dir, "manifest.json")
    os.makedirs(repo_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Write experiment descriptor so metrics.py can read model/temperature
    descriptor_path = os.path.join(exp_base, _EXPERIMENT_DESCRIPTOR)
    if not os.path.exists(descriptor_path):
        with open(descriptor_path, "w", encoding="utf-8") as f:
            json.dump(
                {"model": DEFAULT_MODEL, "model_name": model_name,
                 "temperature": temperature, "experiment_tag": experiment_tag},
                f, indent=2,
            )

    folds: list[dict] = []
    completed_fold_details: list[dict] = []

    for fold_idx, excluded in enumerate(cases, 1):
        fold_id = excluded["id"]
        testname = excluded["testname"]
        db_path = os.path.join(repo_dir, f"fold_{fold_id}")
        training = [tc for tc in cases if tc["id"] != fold_id]
        meta_path = os.path.join(out_dir, f"{fold_id}.json")

        logger.info("[%s] Fold %d/%d — %s (%s)", sut, fold_idx, n_folds, fold_id, testname)

        # ── 1. Build / reuse ChromaDB ─────────────────────────────────────────
        rag = JavaTestRAG(persist_dir=db_path)
        if rag.store.count() > 0:
            logger.info("  Embedding: already indexed (%d docs), skipping", rag.store.count())
        else:
            logger.info("  Embedding: indexing %d training test cases", len(training))
            rag.index_test_cases(training)

        # ── 2. Query Ollama N times — save each run immediately ───────────────
        run_metrics_list: list[dict] = []
        for run_idx in range(1, n_runs + 1):
            seed = base_seed + run_idx - 1      # deterministic: 42, 43, 44, …
            logger.info("  Run %d/%d — seed=%d — querying LLM...", run_idx, n_runs, seed)
            response = rag.query(
                test_provided=excluded["code"],
                top_k=top_k,
                resource_file=resource_file,
                seed=seed,
            )

            # Save prediction text immediately
            run_filename = f"{fold_id}_run_{run_idx:02d}.txt"
            run_path = os.path.join(out_dir, run_filename)
            with open(run_path, "w", encoding="utf-8") as f:
                f.write(response)
            logger.info("  Run %d/%d — saved → %s", run_idx, n_runs, run_filename)

            # Compute + append run metrics to CSV immediately
            run_metrics = append_run_metrics(
                sut, excluded, run_idx, seed, response, system_resource_ids, metrics_dir,
                model=DEFAULT_MODEL, temperature=temperature,
            )
            run_metrics_list.append({"run_id": run_idx, "seed": seed,
                                     "filename": run_filename, **run_metrics})

            # Update fold metadata JSON after every run (partial results survive crashes)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sut": sut, "fold_id": fold_id, "testname": testname,
                        "training_size": len(training),
                        "n_runs_planned": n_runs, "n_runs_completed": run_idx,
                        "base_seed": base_seed, "runs": run_metrics_list,
                    },
                    f, indent=2,
                )

        # ── 3. After all runs: fold average → per-experiment CSV + Excel ─────────
        fold_detail = append_fold_metrics(
            sut, excluded, run_metrics_list, metrics_dir,
            model=DEFAULT_MODEL, temperature=temperature,
        )
        completed_fold_details.append(fold_detail)

        # Update global Excel (All Results + model sheet) immediately
        append_fold_to_global(sut, DEFAULT_MODEL, fold_detail, OUTPUT_METRICS_DIR)

        # Rebuild per-experiment Summary with partial results so far
        partial_summary = aggregate_sut_metrics(
            sut, completed_fold_details, model=DEFAULT_MODEL, temperature=temperature,
        )
        flush_sut_summary([partial_summary], metrics_dir)
        # Rebuild global Summary with all completed SUTs across all experiments
        flush_global_summary([partial_summary], OUTPUT_METRICS_DIR)

        fold_entry = {
            "fold_id": fold_id,
            "excluded_id": fold_id,
            "excluded_testname": testname,
            "db_path": db_path,
            "training_size": len(training),
            "n_runs": n_runs,
            "base_seed": base_seed,
            "runs": [{"run_id": r["run_id"], "seed": r["seed"], "filename": r["filename"]}
                     for r in run_metrics_list],
        }
        folds.append(fold_entry)

        # Update manifest after every completed fold
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(folds, f, indent=2)
        logger.info("  Manifest updated → %s", manifest_path)

    _log_summary(sut, folds, n_runs)
    return folds


def load_manifest(sut: str, chroma_base: str = CHROMA_BASE) -> list[dict]:
    """Load the fold manifest for *sut*."""
    path = os.path.join(chroma_base, sut, "manifest.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, ".."))
    from ril2m.helpers.logging_config import setup_logging
    from ril2m.core import CONTEXTS  # noqa: PLC0415

    setup_logging()
    suts = _available_suts(CONTEXTS)
    logger.info("SUTs discovered: %s", suts)
    for _sut in suts:
        run_crossvalidation(sut=_sut)
