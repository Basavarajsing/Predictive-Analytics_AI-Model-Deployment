import copy
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ARTIFACT_PATH, DATA_PATH, OUTPUT_DIR, RANDOM_STATE, REPORT_DIR
from .data import build_daily_sales, create_data_summary, load_transactions, prepare_transactions
from .features import FEATURE_COLUMNS, make_supervised_frame
from .forecasting import forecast_values


def _models() -> dict[str, Any | None]:
    return {
        "Seasonal Naive (7-day)": None,
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def _predict(model_name: str, estimator: Any, x: pd.DataFrame) -> np.ndarray:
    if model_name == "Seasonal Naive (7-day)":
        return x["lag_7"].to_numpy(dtype=float)
    return estimator.predict(x[FEATURE_COLUMNS])


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    denominator = float(np.abs(actual).sum())
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "wape_percent": float(np.abs(actual - predicted).sum() / denominator * 100)
        if denominator
        else float("nan"),
    }


def train_pipeline(
    data_path: str | Path = DATA_PATH,
    artifact_path: str | Path = ARTIFACT_PATH,
    output_dir: str | Path = OUTPUT_DIR,
    report_dir: str | Path = REPORT_DIR,
) -> dict[str, Any]:
    raw, load_summary = load_transactions(data_path)
    transactions = prepare_transactions(raw)
    data_summary = create_data_summary(raw, transactions, load_summary)
    daily_sales = build_daily_sales(transactions)
    supervised = make_supervised_frame(daily_sales)

    split_at = int(len(supervised) * 0.8)
    train_frame, test_frame = supervised.iloc[:split_at], supervised.iloc[split_at:]
    if len(test_frame) < 30:
        raise ValueError("The chronological holdout must contain at least 30 days.")

    x_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame["target"].to_numpy(dtype=float)
    x_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame["target"].to_numpy(dtype=float)
    splitter = TimeSeriesSplit(n_splits=4, test_size=max(30, len(train_frame) // 10))
    results: list[dict[str, Any]] = []
    estimators = _models()

    for model_name, prototype in estimators.items():
        fold_mae: list[float] = []
        for fold_train, fold_valid in splitter.split(x_train):
            estimator = None if prototype is None else copy.deepcopy(prototype)
            if estimator is not None:
                estimator.fit(x_train.iloc[fold_train], y_train[fold_train])
            predictions = _predict(model_name, estimator, x_train.iloc[fold_valid])
            fold_mae.append(float(mean_absolute_error(y_train[fold_valid], predictions)))

        validation_mae = float(np.mean(fold_mae))
        final_estimator = None if prototype is None else copy.deepcopy(prototype)
        if final_estimator is not None:
            final_estimator.fit(x_train, y_train)
        holdout_prediction = _predict(model_name, final_estimator, x_test)
        metrics = _metrics(y_test, holdout_prediction)
        results.append(
            {
                "model": model_name,
                "validation_mae": validation_mae,
                **metrics,
                "holdout_start": test_frame.index.min().date().isoformat(),
                "holdout_end": test_frame.index.max().date().isoformat(),
            }
        )

    comparison = pd.DataFrame(results).sort_values("validation_mae", kind="stable")
    best_model_name = str(comparison.iloc[0]["model"])
    best_prototype = estimators[best_model_name]
    best_estimator = None if best_prototype is None else copy.deepcopy(best_prototype)
    if best_estimator is not None:
        best_estimator.fit(supervised[FEATURE_COLUMNS], supervised["target"].to_numpy(dtype=float))

    best_result = next(row for row in results if row["model"] == best_model_name)
    best_holdout = _predict(best_model_name, None if best_model_name == "Seasonal Naive (7-day)" else _fit_for_holdout(
        estimators[best_model_name], x_train, y_train
    ), x_test)
    holdout = [
        {
            "date": date.date().isoformat(),
            "actual_sales": float(actual),
            "predicted_sales": float(predicted),
        }
        for date, actual, predicted in zip(test_frame.index, y_test, best_holdout, strict=True)
    ]

    artifact: dict[str, Any] = {
        "model_name": best_model_name,
        "estimator": best_estimator,
        "feature_columns": FEATURE_COLUMNS,
        "history": [
            {"date": date.date().isoformat(), "sales": float(value)}
            for date, value in daily_sales.items()
        ],
        "metrics": {
            **best_result,
            "rows_used_for_training": int(len(supervised)),
            "holdout_rows": int(len(test_frame)),
            "cv_folds": 4,
            "models_compared": len(results),
        },
        "holdout_predictions": holdout,
        "data_summary": data_summary,
    }

    artifact_file = Path(artifact_path)
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_file)
    _write_outputs(transactions, daily_sales, data_summary, comparison, artifact, output_dir, report_dir)
    return artifact


def _fit_for_holdout(prototype: Any, x_train: pd.DataFrame, y_train: np.ndarray) -> Any:
    estimator = copy.deepcopy(prototype)
    estimator.fit(x_train, y_train)
    return estimator


def _write_outputs(
    transactions: pd.DataFrame,
    daily_sales: pd.Series,
    data_summary: dict[str, Any],
    comparison: pd.DataFrame,
    artifact: dict[str, Any],
    output_dir: str | Path,
    report_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    reports_path = Path(report_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_path / "model_comparison.csv", index=False)
    monthly = daily_sales.resample("MS").agg(["sum", "mean", "count"]).rename(
        columns={"sum": "net_sales", "mean": "average_daily_sales", "count": "calendar_days"}
    )
    monthly.to_csv(output_path / "monthly_sales.csv", index_label="month")
    with (output_path / "eda_summary.json").open("w", encoding="utf-8") as file:
        json.dump(data_summary, file, indent=2, allow_nan=False)
    with (output_path / "evaluation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"selected_model": artifact["model_name"], **artifact["metrics"]},
            file,
            indent=2,
            allow_nan=False,
        )

    report = [
        "# Exploratory Data Analysis",
        "",
        f"- Source workbook: `{data_summary['source_file']}`",
        f"- Rows after sheet-overlap reconciliation: {data_summary['rows_after_overlap_reconciliation']:,}",
        f"- Date coverage: {data_summary['date_start'][:10]} through {data_summary['date_end'][:10]}",
        f"- Countries: {data_summary['countries']}; unique products: {data_summary['unique_products']:,}",
        f"- Missing customer IDs: {data_summary['missing_customer_id_rows']:,} "
        f"({data_summary['missing_values']['Customer ID']['percent']}%)",
        f"- Missing descriptions: {data_summary['missing_description_rows']:,} "
        f"({data_summary['missing_values']['Description']['percent']}%)",
        f"- Cancellation-marked rows: {data_summary['cancellation_rows']:,}; "
        f"negative quantity rows: {data_summary['negative_quantity_rows']:,}",
        f"- Adjustment rows excluded from sales target: {data_summary['adjustment_rows']:,}",
        f"- Net line sales (quantity × price, excluding A-prefixed adjustments): "
        f"{data_summary['net_sales_revenue']:,.2f}",
        "",
        "Daily totals include recorded returns/cancellations. Calendar dates without transaction rows are represented as zero recorded net sales. Exact repeated rows within an individual sheet are retained because this workbook has no line identifier to distinguish a repeated legitimate line from a duplicate; cross-sheet overlap is reconciled by preserving the largest exact-row multiplicity found in any sheet.",
    ]
    (reports_path / "eda_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    model_report = [
        "# Model Evaluation",
        "",
        f"- Selected by lowest mean rolling-validation MAE: **{artifact['model_name']}**.",
        f"- Chronological holdout: {artifact['metrics']['holdout_start']} through "
        f"{artifact['metrics']['holdout_end']} ({artifact['metrics']['holdout_rows']} days).",
        f"- Rolling validation: {artifact['metrics']['cv_folds']} time-ordered folds.",
        f"- Holdout MAE: {artifact['metrics']['mae']:,.2f}",
        f"- Holdout RMSE: {artifact['metrics']['rmse']:,.2f}",
        f"- Holdout R²: {artifact['metrics']['r2']:.4f}",
        f"- Holdout WAPE: {artifact['metrics']['wape_percent']:.2f}%",
        "",
        "Metrics are one-day-ahead estimates: at each test date, prior observed sales are available for the lag features. They do not establish equivalent accuracy for a multi-week recursive forecast. See `../outputs/model_comparison.csv` for every candidate and `../outputs/evaluation_summary.json` for the selected-model metrics.",
        "",
        "## Candidate comparison",
        "",
        _comparison_markdown(comparison),
    ]
    (reports_path / "model_evaluation_report.md").write_text(
        "\n".join(model_report) + "\n", encoding="utf-8"
    )


def _comparison_markdown(comparison: pd.DataFrame) -> str:
    columns = list(comparison.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in comparison.iterrows():
        formatted = []
        for column in columns:
            value = row[column]
            formatted.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    trained = train_pipeline()
    print(
        json.dumps(
            {
                "selected_model": trained["model_name"],
                "metrics": trained["metrics"],
                "artifact": str(ARTIFACT_PATH),
            },
            indent=2,
        )
    )
