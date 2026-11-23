"""Leakage-aware model comparison on an offline, public benchmark dataset."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

METRICS = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
}


def make_models(seed: int) -> dict[str, object]:
    """Create fresh, deterministic estimators, including a chance-level baseline."""
    return {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()),
             ("classifier", LogisticRegression(max_iter=2000, random_state=seed))]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, learning_rate=0.08,
            random_state=seed,
        ),
    }


def run_benchmark(*, output: Path, seed: int = 42, folds: int = 5,
                  repeats: int = 2, test_size: float = 0.20) -> dict:
    if folds < 2 or repeats < 1:
        raise ValueError("folds must be >= 2 and repeats >= 1")
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=test_size,
        stratify=dataset.target, random_state=seed,
    )
    if min(np.bincount(y_train)) < folds:
        raise ValueError("folds exceeds available samples in a training class")

    cv = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=repeats, random_state=seed
    )
    models = make_models(seed)
    rows: list[dict] = []
    for name, estimator in models.items():
        scores = cross_validate(
            estimator, x_train, y_train, scoring=METRICS, cv=cv,
            n_jobs=1, error_score="raise", return_train_score=False,
        )
        row = {"model": name}
        for metric in METRICS:
            values = scores[f"test_{metric}"]
            row[f"cv_{metric}_mean"] = float(np.mean(values))
            row[f"cv_{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
        print(f"  {name:29} CV ROC AUC = {row['cv_roc_auc_mean']:.4f} ± {row['cv_roc_auc_std']:.4f}")

    # Make the selection using *training CV only*. Never inspect test labels to choose.
    winner = max(rows, key=lambda r: (r["cv_roc_auc_mean"], r["cv_f1_macro_mean"]))
    chosen = models[winner["model"]]
    chosen.fit(x_train, y_train)
    predicted = chosen.predict(x_test)
    # All included classifiers implement predict_proba, with class index 1.
    probabilities = chosen.predict_proba(x_test)[:, list(chosen.classes_).index(1)]
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "f1_macro": float(f1_score(y_test, predicted, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    print(f"\nSelected (CV only): {winner['model']}")
    print(f"Held-out test ROC AUC: {test_metrics['roc_auc']:.4f}")

    fields = ["model"] + [f"cv_{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]
    with (output / "cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "scikit-learn breast cancer Wisconsin (diagnostic)",
        "dataset_source": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
        "dataset_size": int(len(dataset.target)),
        "features": int(dataset.data.shape[1]),
        "class_labels": {str(i): name for i, name in enumerate(dataset.target_names)},
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "seed": seed, "folds": folds, "repeats": repeats,
        "selection_metric": "cv_roc_auc_mean",
        "selected_model": winner["model"],
        "cv_results": rows,
        "held_out_test_metrics": test_metrics,
        "test_classification_report": classification_report(
            y_test, predicted, target_names=dataset.target_names,
            output_dict=True, zero_division=0,
        ),
        "environment": {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "caveats": [
            "Educational research benchmark, not a clinical diagnostic tool.",
            "Repeated CV fold scores are dependent; their standard deviation is descriptive, not a confidence interval.",
            "A single external test split does not estimate deployment performance across institutions.",
            "Hyperparameters are prespecified; model selection uses training CV only.",
        ],
    }
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    ordered = sorted(rows, key=lambda r: r["cv_roc_auc_mean"], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([r["model"] for r in reversed(ordered)],
            [r["cv_roc_auc_mean"] for r in reversed(ordered)],
            xerr=[r["cv_roc_auc_std"] for r in reversed(ordered)], capsize=3)
    ax.set(xlabel="Mean training CV ROC AUC (error bars: fold-score SD)",
           title="Model comparison on the training partition", xlim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted, display_labels=dataset.target_names,
        values_format="d", ax=ax, cmap="Blues", colorbar=False,
    )
    ax.set_title(f"Held-out test confusion matrix: {winner['model']}")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    md_rows = "\n".join(
        f"| {r['model']} | {r['cv_roc_auc_mean']:.4f} | {r['cv_roc_auc_std']:.4f} | {r['cv_f1_macro_mean']:.4f} |"
        for r in ordered
    )
    report_md = f"""# Experiment Report

> Auto-generated from a local run. Outputs depend on software versions and settings.

## Research question

How do four classical classification baselines compare under a reproducible,
leakage-aware evaluation protocol on the Wisconsin Diagnostic Breast Cancer dataset?

## Methods

- **Data:** {len(dataset.target)} samples, {dataset.data.shape[1]} numeric features; no external downloads.
- **Split:** stratified train/test ({len(y_train)}/{len(y_test)}); seed `{seed}`.
- **Model selection:** {folds}-fold stratified CV repeated {repeats} time(s), *training data only*.
- **Selection metric:** mean ROC AUC; tie-breaker mean macro F1.
- **Preprocessing:** standardization fitted inside logistic-regression CV folds via a pipeline.
- **Hyperparameters:** fixed a priori (see source); no test-set tuning.

## Cross-validation results

| Model | Mean ROC AUC | SD of fold scores | Mean macro F1 |
|---|---:|---:|---:|
{md_rows}

![Cross-validation comparison](model_comparison.png)

## Held-out test results (selected model only)

**Selected model:** {winner['model']}

| Metric | Test score |
|---|---:|
""" + "\n".join(f"| {name} | {value:.4f} |" for name, value in test_metrics.items()) + f"""

![Held-out confusion matrix](confusion_matrix.png)

## Limitations and research ethics

- This is **not** a clinical tool and must not be used to diagnose patients.
- Results are from one small historical dataset, not a multi-site external validation.
- CV fold scores are correlated; displayed SD is descriptive, not a confidence interval.
- Only the CV-selected model is evaluated on the held-out test set.
- No claims of state-of-the-art performance or publication-quality clinical validity.

## Reproduction

```bash
python -m ml_benchmark --seed {seed} --folds {folds} --repeats {repeats} --output results
```

See `results.json` for exact environment versions and machine-readable scores.
"""
    (output / "REPORT.md").write_text(report_md, encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args(argv)
    print("ML Model Benchmarking | leakage-aware offline research demo\n")
    run_benchmark(output=args.output, seed=args.seed, folds=args.folds,
                  repeats=args.repeats, test_size=args.test_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
