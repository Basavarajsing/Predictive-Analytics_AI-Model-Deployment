# Architecture

## Data flow

```text
Online Retail II workbook
  -> src/predictive_analytics/data.py
  -> cross-sheet reconciliation and validation
  -> daily net-sales series
  -> leakage-safe lag/rolling/calendar features
  -> chronological model comparison
  -> Joblib artifact
  -> FastAPI API
  -> Streamlit dashboard
```

## Revenue definition

The transaction target is quantity × price. `C`-prefixed cancellation invoices remain negative sales. `A`-prefixed adjustments are identified and excluded from the main sales target. Zero-price and extreme-value records are not silently removed; the workbook and reports identify their counts.

## Modules

- `src/predictive_analytics/data.py`: loading, validation, reconciliation and daily aggregation.
- `src/predictive_analytics/features.py`: lag, rolling and calendar features.
- `src/predictive_analytics/train.py`: chronological split, cross-validation, model comparison and reports.
- `api/main.py`: model health, data, metrics, holdout and forecast endpoints.
- `dashboard/app.py`: overview, sales, performance and prediction interface.
