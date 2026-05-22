# -*- coding: utf-8 -*-
"""
Leave-one-out cross-validation for the RETORCH annotation RAG pipeline.

For each SUT discovered in the context directory:
  1. Load the per-repo ragtestcases_<sut>.json.
  2. Build N ChromaDB databases — one per test case, each indexed with the N-1
     remaining cases (leave-one-out).  Already-indexed folds are skipped.
     ChromaDB databases are shared across LLM models (same embedding model).
  3. For each fold, query the LLM *n_runs* times via the Jinja2 RAG pipeline to
     predict the @AccessMode annotations of the held-out test case.
  4. Save each run's prediction to ``outputs/crossval/<experiment>/<sut>/TC-XXX_run_<i>.txt``.
  5. Write a manifest.json with fold metadata and all run paths.

Configuration:
    Edit ``ril2m/input/config.json`` to change ``n_runs``, ``top_k``, ``models``,
    and ``temperatures``.

Usage:
    poetry run python ril2m/crossvalidation.py
"""

import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_BASE = os.path.join(BASE_DIR, "chroma_db")
OUTPUT_BASE = os.path.join(BASE_DIR, "..", "outputs", "crossval")
_CONFIG_PATH = os.path.join(BASE_DIR, "input", "config.json")

_CONFIG_DEFAULTS = {
    "n_runs": 10,
    "top_k": 5,
    "base_seed": 42,
    "temperatures": [0.5],
    "models": ["gpt-oss:20b"],
}


# ── configuration ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load pipeline parameters from ``ril2m/input/config.json``.

    Falls back to defaults if the file is missing or a key is absent.
    """
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        merged = {**_CONFIG_DEFAULTS, **cfg}
        # back-compat: single "temperature" key → list
        if "temperature" in merged and "temperatures" not in cfg:
            merged["temperatures"] = [merged["temperature"]]
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


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string (e.g. '1h 23m 45s')."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


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


def _eta_str(durations: list[float], items_remaining: int) -> str:
    """Return a human-readable ETA string from past durations, or empty string if none."""
    if not durations:
        return ""
    avg = sum(durations) / len(durations)
    return f"  ETA ≈ {_fmt_duration(avg * items_remaining)}"


def _execute_run(
    rag,
    *,
    fold_idx: int,
    n_folds: int,
    run_idx: int,
    n_runs: int,
    seed: int,
    excluded: dict,
    top_k: int,
    resource_file: str,
    out_dir: str,
    fold_id: str,
    run_durations: list[float],
) -> tuple[str | None, str | None, float]:
    """Execute one LLM query and save the response to disk.

    Returns ``(response_text, run_filename, elapsed_seconds)``.
    ``response_text`` and ``run_filename`` are ``None`` on failure.
    """
    run_start = time.monotonic()
    eta = _eta_str(run_durations, n_runs - run_idx + 1)
    logger.info(
        "  TC %d/%d  repetition %d/%d — seed=%d — querying LLM...%s",
        fold_idx, n_folds, run_idx, n_runs, seed, eta,
    )
    try:
        response = rag.query(
            test_provided=excluded["code"],
            top_k=top_k,
            resource_file=resource_file,
            seed=seed,
        )
        elapsed = time.monotonic() - run_start
        run_filename = f"{fold_id}_run_{run_idx:02d}.txt"
        with open(os.path.join(out_dir, run_filename), "w", encoding="utf-8") as fh:
            fh.write(response)
        logger.info(
            "  TC %d/%d  repetition %d/%d — done in %s — saved → %s",
            fold_idx, n_folds, run_idx, n_runs, _fmt_duration(elapsed), run_filename,
        )
        return response, run_filename, elapsed
    except Exception:
        elapsed = time.monotonic() - run_start
        logger.exception(
            "  TC %d/%d  repetition %d/%d — seed=%d — unexpected error, skipping",
            fold_idx, n_folds, run_idx, n_runs, seed,
        )
        return None, None, elapsed


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
    model: str | None = None,
    context_dir: str | None = None,
    chroma_base: str = CHROMA_BASE,
    output_base: str = OUTPUT_BASE,
    n_runs: int | None = None,
    top_k: int | None = None,
    base_seed: int | None = None,
    temperature: float | None = None,
) -> list[dict]:
    """
    Run leave-one-out cross-validation for one SUT with one LLM model.

    ChromaDB fold databases are **shared across models** because embeddings are
    produced by a fixed embedding model (``EMBED_MODEL``), not by the generation
    LLM.  Rebuilding them for every model would be wasteful.

    Parameters
    ----------
    sut:         Short repository name (e.g. ``"fullteaching"``).
    model:       Ollama model tag for generation (e.g. ``"gpt-oss:20b"``).
                 Defaults to first entry in ``config.json → models``.
    context_dir: Directory containing context JSONs.
    chroma_base: Root directory for shared ChromaDB fold databases.
    output_base: Root directory for prediction outputs (nested by experiment).
    n_runs:      LLM queries per fold. Defaults to ``config.json`` value.
    top_k:       Similar examples in the LLM prompt. Defaults to ``config.json`` value.
    base_seed:   Seed for run 1; run i uses ``base_seed + i − 1``.
    temperature: Sampling temperature. Defaults to first entry in ``config.json → temperatures``.

    Returns
    -------
    List of fold-info dicts (also written to ``chroma_db/<sut>/manifest.json``).
    """
    from ril2m.core import CONTEXTS, JavaTestRAG  # noqa: PLC0415
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
    if model is None:
        model = cfg["models"][0]
    if temperature is None:
        temperature = cfg["temperatures"][0]

    if context_dir is None:
        context_dir = CONTEXTS

    # Build experiment tag: <model-name>_t<temperature>
    model_name = model.replace(":", "-")
    experiment_tag = f"{model_name}_t{temperature}"

    metrics_dir = os.path.join(OUTPUT_METRICS_DIR, experiment_tag)
    os.makedirs(metrics_dir, exist_ok=True)

    cases = _load_sut_cases(sut, context_dir)
    resource_file = _resource_file_for(sut, context_dir)
    system_resource_ids = load_system_resources(sut, context_dir)
    n_folds = len(cases)
    logger.info(
        "[%s] Starting cross-validation — model=%s  temperature=%s  %d fold(s) × %d run(s)  base_seed=%d",
        sut, model, temperature, n_folds, n_runs, base_seed,
    )

    if n_folds < 1:
        logger.warning("[%s] No test cases found — skipping.", sut)
        return []

    if n_folds == 1:
        logger.warning(
            "[%s] Only 1 test case — fold will have 0 training examples (zero-shot).",
            sut,
        )

    # ChromaDB is shared across models — path does NOT include experiment_tag
    repo_dir = os.path.join(chroma_base, sut)
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
                {"model": model, "model_name": model_name,
                 "temperature": temperature, "experiment_tag": experiment_tag},
                f, indent=2,
            )

    folds: list[dict] = []
    completed_fold_details: list[dict] = []
    fold_durations: list[float] = []   # elapsed seconds per completed fold

    for fold_idx, excluded in enumerate(cases, 1):
        fold_start = time.monotonic()
        fold_id = excluded["id"]
        testname = excluded["testname"]
        db_path = os.path.join(repo_dir, f"fold_{fold_id}")
        training = [tc for tc in cases if tc["id"] != fold_id]
        meta_path = os.path.join(out_dir, f"{fold_id}.json")

        logger.info(
            "[%s] TC %d/%d — %s (%s)%s",
            sut, fold_idx, n_folds, fold_id, testname,
            _eta_str(fold_durations, n_folds - fold_idx + 1) or "  ETA ≈ calculating...",
        )

        # ── 1. Build / reuse ChromaDB (shared across models) ──────────────────
        rag = JavaTestRAG(persist_dir=db_path, model=model, temperature=temperature)
        if rag.store.count() > 0:
            logger.info("  Embedding: already indexed (%d docs), skipping", rag.store.count())
        else:
            logger.info("  Embedding: indexing %d training test cases", len(training))
            rag.index_test_cases(training)

        # ── 2. Save prompt once per fold (for debugging) ──────────────────────
        prompt_path = os.path.join(out_dir, f"{fold_id}_prompt.txt")
        if not os.path.exists(prompt_path):
            prompt_text = rag.build_prompt(excluded["code"], top_k, resource_file)
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            logger.info("  Prompt saved → %s", f"{fold_id}_prompt.txt")

        # ── 3. Query Ollama N times — save each run immediately ───────────────
        run_metrics_list: list[dict] = []
        run_durations: list[float] = []   # elapsed seconds per completed run in this fold

        for run_idx in range(1, n_runs + 1):
            seed = base_seed + run_idx - 1      # deterministic: 42, 43, 44, …
            response, run_filename, run_elapsed = _execute_run(
                rag,
                fold_idx=fold_idx, n_folds=n_folds,
                run_idx=run_idx, n_runs=n_runs,
                seed=seed, excluded=excluded,
                top_k=top_k, resource_file=resource_file,
                out_dir=out_dir, fold_id=fold_id,
                run_durations=run_durations,
            )
            run_durations.append(run_elapsed)

            if response is not None:
                run_metrics = append_run_metrics(
                    sut, excluded, run_idx, seed, response, system_resource_ids, metrics_dir,
                    model=model, temperature=temperature,
                )
                run_metrics_list.append({"run_id": run_idx, "seed": seed,
                                         "filename": run_filename, **run_metrics})
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

        fold_elapsed = time.monotonic() - fold_start
        fold_durations.append(fold_elapsed)
        logger.info(
            "  TC %d/%d — %s complete in %s",
            fold_idx, n_folds, fold_id, _fmt_duration(fold_elapsed),
        )

        # ── 3. After all runs: fold average → per-experiment CSV + Excel ──────
        fold_detail = append_fold_metrics(
            sut, excluded, run_metrics_list, metrics_dir,
            model=model, temperature=temperature,
        )
        completed_fold_details.append(fold_detail)

        # Update global Excel (All Results + model sheet) immediately
        append_fold_to_global(sut, model, fold_detail, OUTPUT_METRICS_DIR)

        # Rebuild per-experiment Summary with partial results so far
        partial_summary = aggregate_sut_metrics(
            sut, completed_fold_details, model=model, temperature=temperature,
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


def _model_param_count(model_tag: str) -> float:
    """Return approximate parameter count from a model tag (e.g. 'llama3:7b' → 7.0).

    Models whose size cannot be parsed sort last (inf) so unknown models run after
    known smaller ones.  Mixture-of-experts tags like '8x7b' are treated as 56b.
    """
    tag = model_tag.split(":")[-1]
    moe = re.search(r'(\d+)x(\d+(?:\.\d+)?)b', tag, re.IGNORECASE)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    m = re.search(r'(\d+(?:\.\d+)?)b', tag, re.IGNORECASE)
    return float(m.group(1)) if m else float("inf")


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
    cfg = load_config()
    suts = _available_suts(CONTEXTS)
    models = sorted(cfg["models"], key=_model_param_count)
    temperatures = cfg["temperatures"]
    n_runs = cfg["n_runs"]

    n_experiments = len(models) * len(temperatures)
    n_total_sut_runs = n_experiments * len(suts)

    sep = "═" * 72
    logger.info(sep)
    logger.info("  EXPERIMENT PLAN")
    logger.info(sep)
    logger.info("  %-24s %s", "Models (least→most):", " | ".join(
        f"{m} (~{_model_param_count(m):.0f}B)" if _model_param_count(m) != float('inf') else m
        for m in models
    ))
    logger.info("  %-24s %s", "Temperatures:", " | ".join(str(t) for t in temperatures))
    logger.info("  %-24s %s", "SUTs:", " | ".join(suts) if suts else "(none found)")
    logger.info("  %-24s %d", "Runs per fold:", n_runs)
    logger.info("  %-24s %d  (%d models × %d temperatures)",
                "Total experiments:", n_experiments, len(models), len(temperatures))
    logger.info("  %-24s %d  (%d experiments × %d SUTs)",
                "Total SUT runs:", n_total_sut_runs, n_experiments, len(suts))
    logger.info(sep)

    exp_idx = 0
    experiment_durations: list[float] = []   # elapsed seconds per completed experiment
    for _model in models:
        for _temperature in temperatures:
            exp_idx += 1
            exp_start = time.monotonic()
            exp_eta = _eta_str(experiment_durations, n_experiments - exp_idx + 1) \
                or "  ETA ≈ calculating..."
            logger.info("")
            logger.info(sep)
            logger.info(
                "  EXPERIMENT %d/%d — model=%s  temperature=%s%s",
                exp_idx, n_experiments, _model, _temperature, exp_eta,
            )
            logger.info(sep)

            sut_durations: list[float] = []   # elapsed seconds per completed SUT in this experiment
            for sut_idx, _sut in enumerate(suts, 1):
                sut_start = time.monotonic()
                sut_eta = _eta_str(sut_durations, len(suts) - sut_idx + 1) \
                    or "  ETA ≈ calculating..."
                logger.info("")
                logger.info("  ── SUT %d/%d : %s%s",
                            sut_idx, len(suts), _sut, sut_eta)
                run_crossvalidation(sut=_sut, model=_model, temperature=_temperature)
                sut_elapsed = time.monotonic() - sut_start
                sut_durations.append(sut_elapsed)
                logger.info("  ── SUT %d/%d : %s  DONE in %s",
                            sut_idx, len(suts), _sut, _fmt_duration(sut_elapsed))

            exp_elapsed = time.monotonic() - exp_start
            experiment_durations.append(exp_elapsed)
            remaining_exp_eta = _eta_str(experiment_durations, n_experiments - exp_idx) \
                or "  (last experiment)"
            logger.info("")
            logger.info("  EXPERIMENT %d/%d COMPLETE in %s — model=%s  temperature=%s",
                        exp_idx, n_experiments, _fmt_duration(exp_elapsed),
                        _model, _temperature)
            logger.info("  Remaining experiments:%s", remaining_exp_eta)
            logger.info(sep)
