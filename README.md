<div align="center">

# ML Model Benchmarking

### A reproducible, leakage-aware machine-learning evaluation toolkit

**Offline dataset · Stratified cross-validation · Transparent model selection · Held-out testing · Automated reports**

[![Python](https://img.shields.io/badge/Python-3.15.2-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3152/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Focus-Reproducible%20ML-blueviolet)](docs/METHODOLOGY.md)

**[Quick Start](#quick-start--windows--vs-code) · [Methodology](#experimental-design) · [Results](#generated-research-artifacts) · [Reproducibility](#reproducibility--testing) · [Limitations](#limitations--responsible-use)**

</div>

---

## Abstract

Machine-learning results are only useful when the evaluation protocol is transparent and repeatable. **ML Model Benchmarking** provides an end-to-end research-oriented experiment comparing four classification approaches under a consistent protocol: a naive majority-class baseline, logistic regression, random forest, and histogram gradient boosting.

The project intentionally prioritizes **sound experimental practice over inflated headline scores**. All preprocessing is contained in cross-validation pipelines where appropriate; model selection uses training-partition cross-validation; only the selected model is evaluated once on the untouched test set. Every run exports machine-readable metrics, presentation-ready figures, and a short technical report.

**Primary research question:** *How do conventional supervised classifiers compare in predictive performance under a controlled, leakage-aware evaluation procedure?*

> **Status:** Educational research implementation. Results must be generated locally; this repository does not present invented or precomputed benchmark scores.

## What makes this different?

| Research practice | Implementation |
|:--|:--|
| Reproducible randomization | Explicit seed for splitting, CV and estimators |
| Data leakage prevention | StandardScaler is placed inside a Pipeline |
| Fair comparison | Same stratified train/test split and CV folds for every model |
| Meaningful baseline | DummyClassifier(strategy="prior") |
| Honest model selection | Highest mean *training CV ROC AUC*, tie-broken by macro F1 |
| Independent final assessment | One held-out test evaluation for the selected model |
| Beyond accuracy | Accuracy, balanced accuracy, macro F1, ROC AUC |
| Auditable outputs | CSV, JSON (with versions), Markdown report, PNG charts |
| Reuse and quality | Command-line interface, unittest, and GitHub Actions CI |
| Offline-first | Dataset distributed with scikit-learn; no API key or download |

## Quick Start — Windows + VS Code

**Prerequisites:** Python **3.15.2** (64-bit), VS Code, and the Microsoft **Python** extension; internet needed only for first-time package installation.

1. Extract the ZIP, then choose **File → Open Folder** in VS Code and select `ml-model-benchmarking`.
2. Open **Terminal → New Terminal** and use **PowerShell**.
3. Execute the commands below, one at a time:

```powershell
py -3.15 --version
py -3.15 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m ml_benchmark
```

> `py -3.15 --version` should report `Python 3.15.2` if that exact version is installed. If the `py` launcher is missing, use `python --version`, then replace `py -3.15` with the exact executable for Python 3.15.2.

**Output:** On successful completion the `results/` folder contains graphs, metrics, and a report. You can also run the **Run ML Benchmark** VS Code launch configuration once `.venv` is selected as the Python interpreter (Ctrl+Shift+P → **Python: Select Interpreter**).

### macOS / Linux

```bash
python3.15 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m ml_benchmark
```

### Example CLI commands

```bash
python -m ml_benchmark --help
python -m ml_benchmark --seed 42 --folds 5 --repeats 2 --output results
python -m unittest discover -s tests -v
```

If the environment has been activated, `python` will resolve to the `.venv` interpreter. Without activation on Windows, use `.\.venv\Scripts\python.exe` as in the quick start.

## Dataset

The repository uses the **Wisconsin Diagnostic Breast Cancer** dataset made available by `sklearn.datasets.load_breast_cancer()`:

- **569** samples and **30** real-valued features.
- Binary targets, malignant and benign (original scikit-learn label mapping is recorded in `results.json`).
- No file downloads, authentication, cloud resources, or user-supplied data required.
- Reference: [scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

**Ethical boundary:** This is a pedagogical dataset and benchmark, **not** a validated clinical decision support system.

## Experimental design

```text
                          Original dataset
                                 │
                 Stratified 80% / 20% split (seed 42)
                                 │
             ┌───────────────────┴───────────────────┐
             │                                       │
          TRAIN                                    TEST
             │                                  untouched
       Repeated stratified CV                         │
         (5 folds × 2 repeats)                        │
             │                                        │
  Fit 4 candidate models, same splits                 │
             │                                        │
    Compare mean CV ROC AUC                           │
             │                                        │
        Select model ── refit on full training ────────┘
                                                    │
                                      One final test assessment
                                                    │
                                        CSV · JSON · PNG · MD
```

**Models:** Dummy (prior), Logistic Regression (standardized), Random Forest, Histogram Gradient Boosting. Hyperparameters are deliberately specified in source and not tuned on the test partition. With `--folds 5 --repeats 2`, 10 fold scores per model are summarized using the mean and sample SD. **The fold-score SD is not a confidence interval** because repeated folds share data.

See [full research methodology](docs/METHODOLOGY.md) for threats to validity, future extensions, and evaluation choices.

## Generated research artifacts

```text
results/
├── cv_results.csv           # Cross-validation metrics for all models
├── results.json             # Configuration, scores, library versions, caveats
├── model_comparison.png     # CV ROC AUC + descriptive error bars
├── confusion_matrix.png     # Held-out predictions for selected model
└── REPORT.md                # Readable methods, results and limitations
```

> Outputs are created only after execution. Generated results are excluded from Git by default to prevent mixing results from different machines or dependency versions. To publish reproducible evidence, copy a verified run to a separate versioned `reports/` directory and document its environment; do not fabricate scores.

## Repository structure

```text
ml-model-benchmarking/
├── .github/workflows/ci.yml      # CI for Python 3.14 and 3.15
├── .vscode/launch.json           # VS Code debug/run configuration
├── .vscode/settings.json
├── docs/METHODOLOGY.md
├── src/ml_benchmark/
│   ├── __init__.py
│   ├── __main__.py
│   └── benchmark.py
├── tests/test_benchmark.py       # Unit + end-to-end tests
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Reproducibility & testing

- Single entry point: `python -m ml_benchmark`.
- Identical data, CV assignments and random seeds are reused across models.
- Critical environment metadata is stored in `results.json`.
- End-to-end tests verify artifacts, valid score bounds, and deterministic outputs across two runs in the **same environment**.
- CI installs the project and runs tests plus a reduced-fold smoke test.

**Important:** Bit-for-bit reproducibility across different operating systems or dependency releases is *not* guaranteed. Record exact installed package versions for a formal research comparison:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > environment-lock.txt
```

Use the generated lock file when reporting a run; don't treat broad declared version ranges as an exact reproducibility lock.

## Limitations & responsible use

1. **One dataset:** Rankings do not generalize to other datasets, especially larger or shifted populations.
2. **One held-out partition:** A separate external cohort would be needed for serious validation.
3. **No statistical significance claim:** Fold-score SD is descriptive, not an independent-sample estimate.
4. **No optimization search:** Models use fixed hyperparameters; this is a controlled demonstration, not exhaustive tuning.
5. **Not medical software:** No diagnostic or treatment decisions should rely on this benchmark.

## Potential research extensions

- Add nested cross-validation for unbiased hyperparameter tuning comparisons.
- Evaluate class imbalance, calibration, and decision-curve tradeoffs.
- Add multiple public datasets and paired uncertainty analysis.
- Compare model size and inference latency under controlled hardware.
- Investigate robustness to corrupted inputs and domain shifts.

## Upload to GitHub

Create a **new empty** repository called `ml-model-benchmarking` on GitHub (**do not** initialize it with another README). Then run in the VS Code terminal:

```powershell
git init
git add .
git commit -m "Initial release: reproducible ML benchmarking toolkit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ml-model-benchmarking.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your own GitHub username. Git must be installed and you must be authenticated with GitHub; HTTPS pushes may require a Git credential manager or personal access token. The remote command shown is a **template**, not an already-created repository.

## Citation and acknowledgments

This project uses scikit-learn and the Wisconsin Diagnostic Breast Cancer dataset. Cite the dataset and scikit-learn when reusing this work in an academic report. See the [original UCI dataset record](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) and [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

## License

Released under the [MIT License](LICENSE). Replace `Project Maintainer` in `pyproject.toml` and `LICENSE` with your own name **before publishing**. Do not imply clinical approval or unpublished research findings.

---

<div align="center"><strong>Research quality begins with transparent experimental design—not just a high accuracy score.</strong></div>
