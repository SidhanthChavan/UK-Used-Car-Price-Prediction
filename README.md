# UK Used-Car Price Modelling

Two MSc coursework notebooks exploring regression techniques on approximately 402,000 UK vehicle listings. The project documents a progression from baseline models to tree ensembles, model interpretation and unsupervised feature experiments.

The notebooks show the full modelling workflow: data cleaning, feature engineering, model comparison, ensembling, feature selection and SHAP interpretation. Known limitations are listed at the end.

## Project structure

```text
.
├── part1-regression/
│   ├── car_price_regression.ipynb
│   └── ML_Report.pdf
├── part2-ensemble/
│   └── car_price_ensemble.ipynb
├── data/                  # put adverts.csv here (not committed)
├── docs/
│   └── evaluation-evidence.md
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

## Dataset

The saved notebook output shows 402,005 rows and 12 source columns, including mileage, registration year, make, model, condition, body type, fuel type and price.

The dataset is not included in this repository. To run the notebooks, place the original `adverts.csv` file in a `data/` folder at the repository root. Both notebooks read it from `../data/adverts.csv`.

## Local setup

```bash
git clone https://github.com/SidhanthChavan/UK-Used-Car-Price-Prediction.git
cd UK-Used-Car-Price-Prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Limitations and next steps

- Some preprocessing (outlier clipping and scaling) runs before the train/test split, so the saved scores may be slightly optimistic. The next version will move every transformation into a scikit-learn `Pipeline` fitted on the training split only.
- In the ensemble notebook, several categorical columns are encoded against a random placeholder target. These should be replaced with train-only target encoding or one-hot encoding.
- Metrics should be reported in pounds (RMSE and MAE) alongside MSE.

## Stack

Python, Pandas, NumPy, scikit-learn, SHAP, Matplotlib and Seaborn.

## Evaluation evidence

See [evaluation evidence](docs/evaluation-evidence.md) for the exact saved scores, dataset-row meaning, preprocessing issues and requirements for a defensible rerun.

## Licence

MIT. See [LICENSE](LICENSE).
