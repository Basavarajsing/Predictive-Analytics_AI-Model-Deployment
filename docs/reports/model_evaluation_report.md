# Model Evaluation

- Selected by lowest mean rolling-validation MAE: **Random Forest**.
- Chronological holdout: 2011-07-20 through 2011-12-09 (143 days).
- Rolling validation: 4 time-ordered folds.
- Holdout MAE: 9,447.34
- Holdout RMSE: 14,352.50
- Holdout R²: 0.6159
- Holdout WAPE: 27.21%

Metrics are one-day-ahead estimates: at each test date, prior observed sales are available for the lag features. They do not establish equivalent accuracy for a multi-week recursive forecast. See `../outputs/model_comparison.csv` for every candidate and `../outputs/evaluation_summary.json` for the selected-model metrics.

## Candidate comparison

| model | validation_mae | mae | rmse | r2 | wape_percent | holdout_start | holdout_end |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 7484.486 | 9447.344 | 14352.497 | 0.616 | 27.209 | 2011-07-20 | 2011-12-09 |
| HistGradientBoosting | 7592.558 | 9983.761 | 14710.039 | 0.597 | 28.754 | 2011-07-20 | 2011-12-09 |
| Ridge Regression | 7958.889 | 11302.034 | 15213.231 | 0.569 | 32.551 | 2011-07-20 | 2011-12-09 |
| Seasonal Naive (7-day) | 9152.094 | 11882.262 | 17320.902 | 0.441 | 34.222 | 2011-07-20 | 2011-12-09 |
