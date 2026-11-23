# Methodology and Research Notes

## Aim

Compare simple classification approaches with a reproducible and auditable evaluation protocol, while exposing important threats to validity.

## Experimental protocol

1. Load the packaged Wisconsin Diagnostic Breast Cancer dataset; no network activity occurs.
2. Reserve a stratified held-out test sample with a fixed random seed.
3. On training data only, run `RepeatedStratifiedKFold` for every candidate estimator using identical folds.
4. Fit logistic regression's scaler **within** its cross-validation training fold, not before splitting.
5. Rank candidates by mean training CV ROC AUC, with macro F1 as a deterministic tie-breaker.
6. Refit the winning estimator on the complete training subset.
7. Evaluate **only the selected model** once on the held-out test subset.
8. Export CV results, held-out test metrics, class report, plots, configuration, and software versions.

## Metric rationale

- **ROC AUC**: discrimination across probability thresholds; primary selection objective.
- **Macro F1**: equally weights both classes in precision/recall balance.
- **Balanced accuracy**: averages sensitivity across target classes.
- **Accuracy**: intuitive but sensitive to class prevalence.

## Why include DummyClassifier?

An uninformed majority-prior model provides an essential baseline. Without it, even a high raw accuracy may be misleading.

## Precommitted design decisions

Candidate models and hyperparameters are predefined; there is no repeated testing of alternatives against held-out labels. If modifying the feature set, model family or hyperparameters after seeing test results, reserve a *new* external dataset for final evaluation rather than claiming the original test is pristine.

## Validity concerns

- Sampling variance: this is one train/test split.
- Dataset representativeness: old, small, curated benchmark.
- Dependence of CV folds: fold SD should not be interpreted as an uncertainty interval for generalization.
- Computation: timing and memory are not tracked and cannot be compared from reported scores.
- Generalizability: neither clinical reliability nor transportability is evaluated.

## Research roadmap

**Experiment A:** multiple seeds on the train/test split to test rank stability (requires a prespecified protocol).

**Experiment B:** nested cross-validation for unbiased comparison of tuned models.

**Experiment C:** calibration curves, probability error and threshold sensitivity.

**Experiment D:** additional open datasets and explicit domain shift.

**Experiment E:** latency, memory and energy profiling with hardware metadata.

## Suggested academic references

- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
- Wolpert, D. H. (1996). The Lack of A Priori Distinctions Between Learning Algorithms. *Neural Computation*, 8(7), 1341–1390.
- Original dataset record: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

If extending the repository for a formal paper, provide a complete citation list and follow the target venue's reproducibility checklist.
