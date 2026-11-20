import pandas as pd
import pytest

from src.predictive_analytics.data import (
    build_daily_sales,
    load_transactions,
    prepare_transactions,
)


def test_prepare_transactions_labels_returns_and_adjustments():
    frame = pd.DataFrame(
        {
            "Invoice": ["500001", "C500002", "A500003"],
            "StockCode": ["A", "B", "BANK"],
            "Description": ["Product", "Return", "Adjustment"],
            "Quantity": [2, -1, 1],
            "InvoiceDate": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "Price": [10.0, 10.0, -20.0],
            "Customer ID": [12345.0, 12345.0, None],
            "Country": ["United Kingdom"] * 3,
        }
    )
    prepared = prepare_transactions(frame)
    assert prepared["IsCancellation"].tolist() == [False, True, False]
    assert prepared["IsAdjustment"].tolist() == [False, False, True]
    assert prepared["SalesRevenue"].tolist() == [20.0, -10.0, 0.0]


def test_daily_sales_contains_missing_calendar_dates():
    frame = pd.DataFrame(
        {
            "Invoice": ["1", "2"],
            "StockCode": ["A", "A"],
            "Description": ["A", "A"],
            "Quantity": [1, 1],
            "InvoiceDate": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "Price": [5.0, 7.0],
            "Customer ID": [1.0, 1.0],
            "Country": ["United Kingdom"] * 2,
        }
    )
    daily = build_daily_sales(prepare_transactions(frame))
    assert daily.index.equals(pd.date_range("2024-01-01", "2024-01-03", freq="D"))
    assert daily.tolist() == [5.0, 0.0, 7.0]


def test_missing_workbook_raises_clear_error(tmp_path):
    missing = tmp_path / "missing.xlsx"
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_transactions(missing)
