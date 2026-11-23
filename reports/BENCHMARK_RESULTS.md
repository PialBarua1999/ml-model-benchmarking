# Machine Learning Benchmarking — Experimental Results

**Project:** Reproducible Machine Learning Model Benchmarking  
**Version:** 1.0.0  
**Python:** 3.15.2  
**Repository:** https://github.com/PialBarua1999/ml-model-benchmarking

## 1. Abstract

This report presents an experimental comparison of four machine learning classification algorithms using the Wisconsin Diagnostic Breast Cancer dataset.

The study evaluates model discrimination through repeated stratified cross-validation and a separate held-out test set.

Among the evaluated models, Logistic Regression achieved the highest mean cross-validation ROC AUC of 0.9957 and a held-out test ROC AUC of 0.9954.

The experiment demonstrates a reproducible benchmarking workflow with leakage-aware model evaluation, automated testing, and systematic model comparison.

## 2. Research Objectives

The objectives of this experiment are to:

1. Compare the discrimination performance of multiple classification algorithms.
2. Evaluate model performance using cross-validation.
3. Select the best-performing model without using the held-out test set for model selection.
4. Assess the selected model on unseen test data.
5. Demonstrate a reproducible machine learning experimentation workflow.

## 3. Dataset

**Dataset:** Wisconsin Diagnostic Breast Cancer Dataset

**Source:** UCI Machine Learning Repository; available through scikit-learn.

**Characteristics:**

- Total samples: 569
- Numerical input features: 30
- Classification problem: Binary
- Target classes: Malignant and Benign

This dataset is used for methodological benchmarking and educational research. The results are not intended for clinical diagnosis.

## 4. Experimental Methodology

The experimental workflow involves:

1. Dataset loading and preparation.
2. Training and test data separation.
3. Model preprocessing and fitting.
4. Repeated stratified cross-validation.
5. Comparison of cross-validation ROC AUC scores.
6. Selection of the highest-performing model based on cross-validation.
7. Final evaluation using the held-out test set.

The held-out test set is reserved for final evaluation rather than model selection.

## 5. Evaluated Models

Four classification approaches were evaluated:

- Dummy Classifier (Prior)
- Logistic Regression
- Random Forest
- Histogram Gradient Boosting

The Dummy Classifier provides a baseline against which the other algorithms can be compared.

## 6. Experimental Results

### 6.1 Cross-Validation Performance

| Model | Mean ROC AUC | Standard Deviation |
|---|---:|---:|
| Dummy Classifier (Prior) | 0.5000 | 0.0000 |
| Logistic Regression | **0.9957** | 0.0053 |
| Random Forest | 0.9903 | 0.0075 |
| Histogram Gradient Boosting | 0.9931 | 0.0051 |

**Best model by mean cross-validation ROC AUC: Logistic Regression**

### 6.2 Held-Out Test Performance

The selected Logistic Regression model achieved:

**Held-out Test ROC AUC: 0.9954**

The reported test score represents the performance obtained during the documented local benchmark execution.

## 7. Interpretation

Logistic Regression achieved the highest mean cross-validation ROC AUC among the evaluated algorithms.

Random Forest and Histogram Gradient Boosting also produced high ROC AUC scores.

However, small numerical differences in cross-validation performance do not necessarily establish statistically significant superiority.

The reported ROC AUC scores demonstrate strong discrimination on this dataset but do not establish clinical suitability or performance on independent patient populations.

## 8. Reproducibility and Testing

The benchmark was successfully executed locally with Python 3.15.2.

**Recorded automated test results:**

- Tests passed: 3
- Tests failed: 0
- Test execution time: 23.46 seconds

The model evaluation was completed without runtime errors.

To reproduce the experiment, follow the installation and execution instructions provided in the repository's main README.

## 9. Research Limitations

This experiment has several limitations:

- Only one publicly available classification dataset was evaluated.
- No external clinical validation was performed.
- Statistical significance between model performance differences was not established.
- Model calibration was not independently assessed.
- Robustness under distribution shifts was not evaluated.
- ROC AUC alone does not demonstrate clinical effectiveness.
- Performance may vary across software versions and computational environments.

## 10. Future Research Directions

1. **Additional datasets:** Evaluate model generalization across multiple classification datasets.
2. **Probability calibration:** Compare calibration curves, Brier scores, and calibration errors.
3. **Statistical analysis:** Investigate model performance differences using appropriate paired statistical comparisons.
4. **Explainability:** Explore permutation importance, SHAP, and other interpretation methods.
5. **Robustness:** Evaluate model sensitivity to noise, missing values, and distribution shifts.

## 11. Conclusion

This project demonstrates a reproducible machine learning benchmarking workflow using multiple classification algorithms, cross-validation-based model selection, and final held-out evaluation.

Logistic Regression achieved the highest mean cross-validation ROC AUC of 0.9957 and a held-out test ROC AUC of 0.9954.

The findings are specific to the selected dataset and experimental configuration. Further evaluation would be required to establish broader generalizability.

---

**Research Area:** Machine Learning, Model Evaluation, Reproducible Research

**Report Status:** Results recorded from successful local execution using Python 3.15.2.

**Repository:** https://github.com/PialBarua1999/ml-model-benchmarking
