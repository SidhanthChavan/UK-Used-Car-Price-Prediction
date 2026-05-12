# UK Used Car Price Prediction

> End-to-end regression project showing progression from baseline models to advanced ensemble methods.
> **Part 1: R² = 0.88 (Decision Tree) → Part 2: R² = 0.94 (Stacking Ensemble)**

## The Story

This project spans two semesters of MSc Data Science — the same dataset, increasingly sophisticated approaches.

Part 1 established baselines comparing Linear Regression, KNN, and Decision Tree on 400,000 UK used car listings. Part 2 advanced to stacking ensembles, SHAP interpretability, and dimensionality reduction — pushing R² from 0.88 to 0.94.

---

## Part 1 — Regression Baselines

### Dataset

- 400,000 UK used car listings
- Features: mileage, year, brand, fuel type, transmission, body type, condition
- Target: sale price

### Results

| Model | R² | RMSE |
|-------|-----|------|
| Decision Tree | 0.880 | 0.341 |
| KNN | 0.853 | 0.385 |
| Linear Regression | 0.511 | 0.701 |

### Methodology

- Missing value imputation (median for numerical, mode for categorical)
- IQR-based outlier removal
- Feature engineering: VehicleAge, PricePerMile
- One-hot encoding (low cardinality) + frequency encoding (high cardinality)
- StandardScaler normalisation
- GridSearchCV hyperparameter tuning with 5-fold cross-validation
- Feature importance analysis — vehicle age and mileage are top predictors

---

## Part 2 — Advanced Ensemble Methods

### Results

| Model | R² |
|-------|-----|
| Stacking Ensemble | 0.94 |
| Random Forest baseline | ~0.88 |

### Methodology

- Stacking ensemble: Random Forest + Gradient Boosting + Ridge meta-learner
- RFECV feature selection
- RandomizedSearchCV hyperparameter tuning
- SHAP TreeExplainer for model interpretability
- PCA + t-SNE dimensionality reduction
- K-Means clustering as feature engineering technique
- Tableau dashboard for stakeholder-ready visualisation

---

## Project Structure
---

## Tech Stack

Python · Pandas · NumPy · Scikit-Learn · SHAP · Matplotlib · Seaborn · Tableau

---

## Author

Sidhanth Chavan — MSc Data Science, Manchester Metropolitan University
linkedin.com/in/sidhanth-chavan · github.com/SidhanthChavan
