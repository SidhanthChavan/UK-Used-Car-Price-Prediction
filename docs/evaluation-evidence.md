# Evaluation evidence and limitations

Reviewed against the committed notebooks on 27 October 2026. This note explains
saved outputs; the experiment has not been rerun because `adverts.csv` is absent.

## What 402,005 means

The ensemble notebook's initial `shape` output is `(402005, 12)`. These are
vehicle-advert rows in the input CSV, not verified unique vehicles, customers,
sales or test examples. Columns include an advert reference, mileage, make,
model, registration year, vehicle condition, body type, fuel and asking price.

## How the saved score was calculated

One experiment splits the transformed data using `random_state=42`: 70% train,
15% validation and 15% test. A stacking regressor combines Random Forest and
Gradient Boosting with a Ridge final estimator. The saved `r2_score(y_test,
y_pred_stack)` is `0.9446571696497522`, which rounds to 0.945.

However, a later stacking cell saves R-squared `0.924750660144111` and RMSE
`2304.3498096419394`. These outputs do not establish one reproducible final run.

## Why neither is a validated benchmark

- Outlier clipping, including the target price, occurs before splitting.
- Scaling is fitted before splitting, including on the target.
- Several categorical columns are encoded using an unseeded random dummy
  target. This does not provide a defensible price-based encoding.
- Saved outputs reflect different runs or data states; paths are machine-specific.
- One cell labels MSE as RMSE. Later cells calculate its square root correctly.

## Required rerun

Obtain and document the original dataset and its licence; preserve an untouched
test set before fitting transformations. Drop advert identifiers, use train-only
preprocessing within scikit-learn pipelines, remove dummy-target encoding and
evaluate an unchanged price target in currency units. Save the random seed,
split indices, environment, model parameters and metrics from a fresh execution.
Use a training-median baseline and report MAE, RMSE and R-squared. Dataset absence
currently blocks a genuine rerun; no replacement score has been invented.
