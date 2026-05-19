# CLAUDE.md — retorch-ril2m-rp

## Project overview

**retorch-ril2m-rp** is the replication package for a research paper on automated generation
of RETORCH resource-access annotations for Java end-to-end test cases using Retrieval-Augmented
Generation (RAG) and an LLM.

Given a new, unannotated Java test method the system:
1. Retrieves the most similar already-annotated test cases from a vector store (ChromaDB).
2. Builds a context-enriched prompt using a Jinja2 template.
3. Queries a local LLM (via Ollama) to suggest the missing `@AccessMode` annotations.

## Pipeline overview

```
fetch_testcases.py          →  crossvalidation.py          →  metrics.py
Fetch Java tests from GitHub   Build N ChromaDB databases      Compute M1-M8 metrics
Write per-repo JSONs           (leave-one-out per repo)        Write CSV, JSON, Excel
                               Query LLM n_runs × per fold
                               Save outputs + metrics live
```

## Repository layout

```
retorch-ril2m-rp/
├── ril2m/
│   ├── __init__.py
│   ├── core.py                        # RAG classes + querying entry point
│   ├── crossvalidation.py             # Leave-one-out cross-validation (all SUTs)
│   ├── metrics.py                     # M1-M8 metrics computation + CSV/Excel output
│   ├── chroma_db/                     # Auto-generated ChromaDB fold databases (shared across models)
│   │   └── <repo>/
│   │       ├── fold_TC-001/
│   │       └── manifest.json
│   ├── helpers/
│   │   ├── __init__.py                # Re-exports all helpers
│   │   ├── helpers.py                 # Backwards-compat re-export shim
│   │   ├── file_utils.py              # loadfile, save_output_to_file
│   │   ├── code_utils.py              # extract_snippets
│   │   ├── excel_utils.py             # write_metrics_excel, global Excel helpers
│   │   ├── logging_config.py          # Structured logging setup
│   │   └── ollamaClient.py            # Ollama HTTP client wrapper
│   └── input/
│       ├── fetch_testcases.py         # Fetches repos and writes per-repo JSONs
│       ├── config.json                # Experiment parameters (n_runs, top_k, base_seed)
│       ├── context/
│       │   ├── ragtestcases_<repo>.json    # Auto-generated test cases per SUT
│       │   ├── systemresources_<repo>.json # Auto-generated resource definitions per SUT
│       │   └── resourcefile.json           # Static resource definitions (legacy)
│       └── prompts/
│           └── generate_annotations.j2     # Jinja2 prompt template
├── outputs/
│   ├── crossval/
│   │   └── <experiment_tag>/              # e.g. gpt-oss-20b_t0.5
│   │       ├── .experiment.json           # model, temperature, experiment_tag
│   │       └── <sut>/
│   │           ├── TC-001_run_01.txt      # LLM prediction (saved immediately per run)
│   │           ├── TC-001.json            # Fold metadata (updated after every run)
│   │           └── …
│   └── metrics/
│       ├── metrics_global.xlsx            # Global Excel: Summary + All Results + per-model sheets
│       ├── metrics_global_summary.csv     # Global summary (one row per sut × experiment)
│       └── <experiment_tag>/
│           ├── metrics_<sut>.csv          # Per-fold metrics (averaged across runs)
│           ├── metrics_<sut>_runs.csv     # Per-run metrics (one row per fold × run)
│           ├── metrics_<sut>.json         # Full result dict with fold + run details
│           ├── metrics_summary.csv        # One row per SUT for this experiment
│           └── metrics.xlsx              # Summary + Fold Details + Run Details sheets
├── tests/
│   └── test_basic.py
├── Jenkinsfile
├── pyproject.toml
└── docker-compose.yml
```

## Key source files

### `ril2m/input/fetch_testcases.py`
Fetches Java test files from the configured GitHub repositories and writes **one JSON file
per repository** under `ril2m/input/context/`.

**Key public API:**

| Symbol | Purpose |
|--------|---------|
| `REPOS` | List of GitHub repo URLs to process — edit this to add/remove repos |
| `extract_test_cases_from_url(repo_url)` | Fetch & parse one repo; returns `(list[dict], system_resources_str)` |
| `generate_ragtestcases(repo_urls, context_dir)` | Generate all per-repo JSONs; returns `dict[short_name, cases]` |
| `load_all_ragtestcases(context_dir)` | Merge all `ragtestcases_*.json` into one list (used by `core.py`) |

**Output file naming** — the short name is the last segment after the final `-` in the repo name:

| Repository | Short name | Test cases file | Resources file |
|------------|-----------|-----------------|----------------|
| `retorch-st-petclinic` | `petclinic` | `ragtestcases_petclinic.json` | `systemresources_petclinic.json` |
| `retorch-st-fullteaching` | `fullteaching` | `ragtestcases_fullteaching.json` | `systemresources_fullteaching.json` |
| `retorch-st-eShopContainers` | `eShopContainers` | `ragtestcases_eShopContainers.json` | `systemresources_eShopContainers.json` |

IDs are sequential and **scoped per repo** (each file starts at `TC-001`).

`extract_test_cases_from_url(repo_url)` returns `(test_cases, system_resources_content)`:
- Accepts any public GitHub repository URL.
- In a single tree-listing API call, discovers all `.java` files under `src/test/` **and**
  the `*SystemResources.json` file under `.retorch/configurations/`.
- Extracts methods annotated with `@Test` or `@ParameterizedTest`.
- Separates `@AccessMode` annotations (→ `annotations` field) from the rest of the method
  code (→ `code` field). Private helper methods between consecutive test methods are
  included in the preceding test's `code` block.
- Skips methods and classes marked `@Disabled`.
- Logs a per-repo summary: file count, test names, and `@AccessMode` count statistics
  (min / max / average per test case).

Set `GITHUB_TOKEN` to avoid GitHub API rate limits (unauthenticated: 60 req/hour; raw
file fetches are unlimited and do not count against this quota).

### `ril2m/input/config.json`
Single file controlling all experiment parameters:

```json
{ "n_runs": 10, "top_k": 5, "base_seed": 42 }
```

| Key | Default | Description |
|-----|---------|-------------|
| `n_runs` | 10 | LLM queries per fold |
| `top_k` | 5 | Similar examples included in the RAG prompt |
| `base_seed` | 42 | Seed for run 1; run *i* uses `base_seed + i − 1` (deterministic) |
| `models` | `["gpt-oss:20b"]` | Ollama model tags to evaluate; crossvalidation iterates over all |
| `temperatures` | `[0.5]` | Sampling temperatures to evaluate; crossvalidation iterates over all |

### `ril2m/crossvalidation.py`
Runs the full leave-one-out cross-validation loop for all SUTs.

**Key public API:**

| Symbol | Purpose |
|--------|---------|
| `load_config()` | Load `n_runs`, `top_k`, `base_seed`, `models`, `temperatures` from `ril2m/input/config.json` |
| `run_crossvalidation(sut, model, context_dir, chroma_base, output_base, n_runs, top_k, base_seed, temperature)` | Full build + query loop for one model×temperature; returns list of fold-info dicts |
| `_model_param_count(model_tag)` | Parse parameter count from a model tag (e.g. `llama3:7b` → 7.0) for ordering |
| `load_manifest(sut, chroma_base)` | Load the fold manifest for a SUT |

**How it works (per fold):**
1. Load `ragtestcases_<sut>.json` and `systemresources_<sut>.json`.
2. For each test case `TC-k`: create a `JavaTestRAG` backed by `chroma_db/<sut>/fold_TC-k/`
   (shared across models — same embeddings) and index the N-1 training cases (skipped if already indexed).
3. Query the LLM `n_runs` times — each run uses seed `base_seed + run_idx - 1`.
   **Each run is wrapped in try/except — a failed run logs a full traceback and is skipped; the experiment continues.**
4. Save each run immediately to `outputs/crossval/<experiment>/<sut>/TC-k_run_NN.txt`.
5. Append run metrics to `outputs/metrics/<experiment>/metrics_<sut>_runs.csv`.
6. After all runs: compute fold average, append to fold CSV, update per-experiment and global Excel.
7. After all folds: write manifest and log summary.

The `__main__` block:
- **Sorts models by parameter count (ascending)** so less powerful models (more prone to hallucinations) run first.
- Logs a full **experiment plan summary table** at startup (models, temperatures, SUTs, runs/fold, total experiments).
- Logs clear banners on each **model/temperature change** and each **SUT change**, with progress counters (e.g. `SUT 2/3`, `EXPERIMENT 2/4`).
- Iterates `models × temperatures × suts` from config so the full grid runs unattended.

All output files are written **incrementally** — if the job is cancelled, results up to
the last completed run are preserved in CSV and Excel.

### `ril2m/metrics.py`
Computes M1-M8 metrics across all SUTs and writes output files.

| Metric | Description |
|--------|-------------|
| M1 Correct@N | Proportion of runs where all predicted `@AccessMode` syntactically valid |
| M2 Pass@N | Proportion of runs where all predicted `resID`s existing in SystemResources |
| M3 Acc@N | Proportion of runs where predicted resource set exactly matches ground truth |
| M4 Avg TP | Average resources correctly tagged |
| M5 Avg TN | Average resources correctly NOT tagged |
| M6 Avg FP | Average resources incorrectly tagged (M6.1 real FP + M6.2 hallucinations) |
| M7 Avg FN | Average resources missed |
| M8 F1 / F1_Hall | Macro-averaged F1; F1_Hall counts only hallucinations as FP |

N = `n_runs` from config. All CSVs and Excel sheets include `model` and `temperature` columns.

**Annotation parsing (`parse_access_modes`):**
- Guards against `None` / empty text input (returns `[]`).
- `concurrency` field: converted to `int` via `_to_int`; non-integer LLM values (e.g. `concurrency="high"`) produce `concurrency=None` (→ syntactically incorrect in M1) **but are preserved in `concurrency_raw`** for hallucination analysis.
- A `WARNING` log line is emitted when a non-integer concurrency is detected, including the raw value.
- The `raw` field stores the full `@AccessMode(...)` string as emitted by the LLM.

**Annotation dict fields:**

| Field | Type | Notes |
|-------|------|-------|
| `resID` | `str \| None` | Resource identifier |
| `accessMode` | `str \| None` | One of READONLY / READWRITE / NOACCESS / DYNAMIC |
| `concurrency` | `int \| None` | Parsed integer; `None` if absent or non-integer |
| `concurrency_raw` | `str \| None` | Original LLM string (e.g. `"high"`); `None` if field absent |
| `sharing` | `bool \| None` | `True`/`False`; `None` if field absent |
| `raw` | `str` | Full `@AccessMode(...)` text |

**Output layout:**

| File | Location | Description |
|------|----------|-------------|
| `metrics_global.xlsx` | `outputs/metrics/` | **Summary** + **All Results** + one sheet per model |
| `metrics_global_summary.csv` | `outputs/metrics/` | One row per sut × experiment (normalized metric names) |
| `metrics_<sut>.csv` | `outputs/metrics/<experiment>/` | Per-fold metrics (averaged across runs) |
| `metrics_<sut>_runs.csv` | `outputs/metrics/<experiment>/` | Per-run metrics |
| `metrics_<sut>.json` | `outputs/metrics/<experiment>/` | Full dict including fold + run details |
| `metrics_summary.csv` | `outputs/metrics/<experiment>/` | One row per SUT for this experiment |
| `metrics.xlsx` | `outputs/metrics/<experiment>/` | Summary + Fold Details + Run Details sheets |

### `ril2m/helpers/`
Split by responsibility:

| Module | Contents |
|--------|----------|
| `file_utils.py` | `loadfile`, `save_output_to_file` |
| `code_utils.py` | `extract_snippets` |
| `excel_utils.py` | `write_metrics_excel`, global Excel helpers (openpyxl) |
| `logging_config.py` | `setup_logging` |
| `ollamaClient.py` | `OllamaClient` — `chat()` and `embed()` guard response key access; unexpected Ollama response structure raises `ValueError` with a full traceback in the log |
| `helpers.py` | Backwards-compatible re-export shim |

### `ril2m/core.py`
Houses the `TestCaseVectorStore` and `JavaTestRAG` classes used by both the
cross-validation builder and the querying phase.

`TestCaseVectorStore` — wraps ChromaDB; provides `add_test_cases()` and `search()`.
Embeddings are computed **exclusively from the `code` field** so similarity search is
based on test logic, not on existing annotations.

`JavaTestRAG` — full RAG pipeline: retrieve similar cases → build prompt → query LLM → return response.
`query(test_provided, top_k, resource_file, seed)` accepts a `seed` for reproducible generation.

### `ril2m/input/context/ragtestcases_<name>.json`
Auto-generated, one file per repository. Each entry has:
```json
{
  "id": "TC-001",
  "code": "<non-@AccessMode annotations + method signature + body>",
  "annotations": "<@AccessMode annotations, one per line>",
  "testname": "<MethodName with first letter uppercased>"
}
```

### `ril2m/input/context/systemresources_<name>.json`
Auto-generated, one file per repository. Contains the raw `*SystemResources.json` from
`.retorch/configurations/` — the resource definitions (type, capacities, Docker image,
elasticity model) used by that SUT.

Do not edit these files manually; run `fetch_testcases.py` to regenerate them.

## Running locally

```bash
# 1. Install dependencies
poetry install

# 2. Start Ollama (GPU container or local install)
docker compose up ollama-gpu --detach

# 3. Fetch test cases from GitHub (writes per-repo JSONs)
poetry run python ril2m/input/fetch_testcases.py

# 4. Build leave-one-out cross-validation ChromaDB databases + query LLM
poetry run python ril2m/crossvalidation.py

# 5. (optional) Recompute all metrics from saved predictions
poetry run python ril2m/metrics.py

# 6. Run tests
poetry run pytest
```

## Environment variables

| Variable       | Description                                           |
|----------------|-------------------------------------------------------|
| `GITHUB_TOKEN` | Optional GitHub personal access token (avoids rate limits) |
| `OLLAMAIP`     | IP/host of the Ollama instance (used when not in CI)  |
| `CI_ENV`       | When set, Ollama connects to `ollama-gpu:11434`       |

## CI/CD (Jenkinsfile)

The pipeline runs on a `slave-xg` node with GPU access:

| Stage              | Action                                                              |
|--------------------|---------------------------------------------------------------------|
| Init               | Clean workspace, checkout SCM                                       |
| Build              | `poetry install`                                                    |
| Deploy             | Start the Ollama Docker container and download models               |
| Generate Input     | Run `fetch_testcases.py`; prints generation summary                 |
| Cross-Validation   | Run `crossvalidation.py`; embed folds + query LLM (all SUTs)       |
| Compute Metrics    | Run `metrics.py`; writes CSV, JSON, and Excel reports               |
| Tear-down          | Store container logs, stop Docker                                   |
| Archive results    | Always runs (even on cancellation) — archives all outputs           |
| test               | `poetry run pytest`                                                 |

Artifacts archived: `ril2m/input/context/*.json`, `ril2m/chroma_db/**/manifest.json`,
`outputs/crossval/**/*`, `outputs/metrics/**/*`.

## Adding a new repository

Append the repository URL to `REPOS` in `ril2m/input/fetch_testcases.py`:

```python
REPOS = [
    "https://github.com/giis-uniovi/retorch-st-petclinic",
    "https://github.com/giis-uniovi/retorch-st-fullteaching",
    "https://github.com/giis-uniovi/retorch-st-eShopContainers",
    "https://github.com/your-org/retorch-st-myapp",   # ← add here
]
```

A new file `ragtestcases_myapp.json` will be created automatically on the next run.
The short name is derived from the last `-`-separated segment of the repo name.

Requirements:
- Public GitHub repository (or set `GITHUB_TOKEN` for private repos)
- Java test files under `src/test/`
- Test methods annotated with `@Test` or `@ParameterizedTest`
- Resource-access annotations using `@AccessMode` from `giis.retorch.annotations`
