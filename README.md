# Replication Package for *'Automating Test Case Resource Identification in System Testing Using Large Language Models'*

This repository contains the replication package of the paper *Automating Test Case Resource Identification in System Testing Using Large Language Models*
published at *TO-DO*

This replication package includes the Python scripts, Jenkinsfile, docker-compose configuration, and all generated
data required to reproduce the evaluation. The replication package structure is depicted as follows:

```
📁 /
├── 📜 .gitignore
├── 🤝 LICENSE
├── 🏗️ pyproject.toml
├── 📖 README.md
├── 🐳 docker-compose.yml
├── 🚀 Jenkinsfile
│
├── 📦 ril2m/
│   ├── 🐍 core.py
│   ├── 🐍 crossvalidation.py
│   ├── 🐍 metrics.py
│   ├── 🐍 __init__.py
│   ├── 📦 helpers/
│   │   ├── 🐍 file_utils.py
│   │   ├── 🐍 code_utils.py
│   │   ├── 🐍 excel_utils.py
│   │   ├── 🐍 logging_config.py
│   │   └── 🐍 ollamaClient.py
│   └── 📁 input/
│       ├── 🐍 fetch_testcases.py
│       ├── ⚙️ config.json
│       ├── 📁 context/
│       │   ├── 📋 ragtestcases_<sut>.json
│       │   └── 📋 systemresources_<sut>.json
│       └── 📁 prompts/
│           └── 📝 generate_annotations.j2
│
└── 📊 outputs/
    ├── 📁 crossval/
    │   └── 📁 <model>_t<temp>/
    │       ├── 📋 .experiment.json
    │       └── 📁 <sut>/
    │           ├── 📝 TC-001_run_01.txt
    │           └── 📋 TC-001.json
    └── 📁 metrics/
        ├── 📊 metrics_global.xlsx
        ├── 📋 metrics_global_summary.csv
        └── 📁 <model>_t<temp>/
            ├── 📊 metrics.xlsx
            ├── 📋 metrics_<sut>.csv
            ├── 📋 metrics_<sut>_runs.csv
            └── 📋 metrics_summary.csv
```

- `📦 ril2m/` contains the Python scripts implementing the RAG annotation-generation pipeline and its evaluation.
  It includes `🐍 core.py`, which houses the `TestCaseVectorStore` and `JavaTestRAG` classes; `🐍 crossvalidation.py`,
  which runs the leave-one-out cross-validation loop over all SUTs; `🐍 metrics.py`, which computes M1-M8 metrics and
  writes incremental CSV and Excel reports; a `📦 helpers/` folder with supporting utilities; and an `📁 input/` folder
  with the test-case context data and the Jinja2 prompt template.
  During execution, an `📊 outputs/` folder is dynamically generated to store predictions and metric reports.

- `📊 outputs/` accumulates results during the pipeline run.
  - `📁 crossval/<model>_t<temp>/<sut>/` stores one `.txt` file per LLM query (saved immediately)
    and a `.json` metadata file updated after every run so that results are recoverable if the job is cancelled.
  - `📁 metrics/<model>_t<temp>/` stores per-fold and per-run CSVs and an Excel workbook with
    Summary, Fold Details, and Run Details sheets, also updated incrementally.
  - `📊 metrics_global.xlsx` in the metrics root combines all experiments into a single workbook:
    a **Summary** sheet, an **All Results** sheet, and one sheet per model tested.

The replication package data is also archived on [Zenodo](https://doi.org/TO-DO)

## Experimental Subjects

The evaluation is performed on three real-world Java end-to-end test suites, each developed following the
RETORCH resource-aware testing methodology:

| SUT | Description | Repository |
|-----|-------------|-----------|
| **PetClinic** | Spring Boot veterinary clinic demo | [retorch-st-petclinic](https://github.com/giis-uniovi/retorch-st-petclinic) |
| **FullTeaching** | Online education platform (ElasTest demonstrator) | [retorch-st-fullteaching](https://github.com/giis-uniovi/retorch-st-fullteaching) |
| **eShopOnContainers** | Microservices-based e-commerce reference app | [retorch-st-eShopContainers](https://github.com/giis-uniovi/retorch-st-eShopContainers) |

Each SUT provides Java test cases annotated with `@AccessMode` from `giis.retorch.annotations` and a
`*SystemResources.json` file describing the available test resources.

## Treatment Replication Procedure

Python 3.12 or later and [Poetry](https://python-poetry.org/) are required.
An Ollama instance with GPU access is needed to run the LLM queries.

1. **Install dependencies:**

   ```bash
   poetry install
   ```

2. **Start the Ollama container:**

   ```bash
   docker compose up ollama-gpu --detach
   ```

3. **(Optional) Configure the experiment** by editing `ril2m/input/config.json`:

   ```json
   { "n_runs": 10, "top_k": 5, "base_seed": 42 }
   ```

4. **Fetch test cases** from the SUT repositories (requires internet access):

   ```bash
   poetry run python ril2m/input/fetch_testcases.py
   ```

5. **Run the cross-validation** (embedding + LLM querying for all SUTs):

   ```bash
   poetry run python ril2m/crossvalidation.py
   ```

6. **(Optional) Recompute metrics** from saved prediction files:

   ```bash
   poetry run python ril2m/metrics.py
   ```

7. **Run the test suite:**

   ```bash
   poetry run pytest
   ```

The pipeline writes results **incrementally**: cancelling at any point preserves all completed predictions and
their corresponding CSV/Excel metric rows.

## Contributing

See the general contribution policies and guidelines for *giis-uniovi* at
[CONTRIBUTING.md](https://github.com/giis-uniovi/.github/blob/main/profile/CONTRIBUTING.md).

## Contact

Contact any of the researchers who authored the paper; their affiliation and contact information are provided in the
paper itself.

## Citing this work

TO-DO

## Acknowledgments

This work was supported in part by the project EQUAVEL (PID2022-137646OB-C32) funded by MCIN/AEI/10.13039/501100011033/FEDER, UE and in part by the European [HORIZON-KDT-JU research project MATISSE](https://matisse-kdt.eu/): *"Model-based engineering of Digital Twins for early verification and validation of Industrial Systems"*, HORIZON-KDT-JU-2023-2-RIA, Proposal number: 101140216-2, KDT232RIA_00017, and also by the (partial) support of the PNRR MUR project [FAIR (PE0000013)](https://www.mur.gov.it/sites/default/files/2023-02/D.D.%20341%20_PE0000013_rev181022NF.pdf).
This paper has been also partially supported by the Italian MUR PRIN 2022 Project: Domain (Grant Agreement #2022TSYYKJ) financed by NextGenEu.
