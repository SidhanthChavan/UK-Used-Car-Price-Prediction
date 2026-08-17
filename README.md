# UK Used-Car Price Modelling

Two MSc coursework notebooks exploring regression techniques on approximately 402,000 UK vehicle listings. The project documents a progression from baseline models to tree ensembles, model interpretation and unsupervised feature experiments.

The repository is best read as a record of exploratory modelling decisions. Its saved outputs are not presented as a production benchmark because the current notebooks perform some preprocessing before the train/test split and contain machine-specific data paths.

## Project structure

```text
.
├── part1-regression/
│   ├── car_price_regression.ipynb
│   └── ML_Report.pdf
├── part2-ensemble/
│   └── car_price_ensemble.ipynb
├── requirements.txt
└── README.md
```

## Part 1: regression baselines

The first notebook covers:

- missing-value inspection and basic exploratory analysis
- vehicle-age feature engineering
- categorical encoding and feature scaling
- Linear Regression, K-Nearest Neighbours and Decision Tree models
- residual review and tree feature importance

## Part 2: ensemble experiments

The second notebook extends the work with:

- Random Forest and Gradient Boosting regression
- a stacking regressor with a Ridge meta-model
- RFECV and univariate feature selection
- SHAP-based interpretation
- PCA, t-SNE and K-Means experiments

These techniques are exploratory. The notebook needs a clean, single execution path before its metrics can be considered reproducible.

## Dataset

The saved notebook output shows 402,005 rows and 12 source columns, including mileage, registration year, make, model, condition, body type, fuel type and price.

The dataset is not included in this repository. Both notebooks currently reference local file locations. To reproduce the work, obtain the original `adverts.csv` file and update the first `read_csv` cell in each notebook.

## Local setup

```bash
git clone https://github.com/SidhanthChavan/UK-Used-Car-Price-Prediction.git
cd UK-Used-Car-Price-Prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Reproducibility work still required

- Split the raw data before fitting imputers, encoders or scalers.
- Replace the random dummy-target encoding in the ensemble notebook.
- Remove stale outputs and execute every cell once from a fresh kernel.
- Keep all transformations inside scikit-learn pipelines.
- Add a public dataset reference or a documented download procedure.
- Save final metrics in original currency units and distinguish MSE from RMSE.

Until that work is complete, the existing scores should be treated as exploratory notebook outputs rather than verified model performance.

## Stack

Python, Pandas, NumPy, scikit-learn, SHAP, Matplotlib and Seaborn.
