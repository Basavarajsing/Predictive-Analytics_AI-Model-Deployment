from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]
ROW_KEY = REQUIRED_COLUMNS


def load_transactions(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load workbook sheets and remove only duplicated rows caused by sheet overlap."""
    workbook_path = Path(path)
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {workbook_path}")

    frames: list[pd.DataFrame] = []
    sheet_summary: list[dict[str, Any]] = []
    with pd.ExcelFile(workbook_path, engine="openpyxl") as workbook:
        for sheet_name in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet_name)
            missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
            if missing_columns:
                raise ValueError(f"Sheet {sheet_name!r} is missing required columns: {missing_columns}")
            frame = frame[REQUIRED_COLUMNS].copy()
            frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
            if frame["InvoiceDate"].isna().any():
                raise ValueError(f"Sheet {sheet_name!r} contains invalid InvoiceDate values.")
            if frame[["Quantity", "Price"]].isna().any().any():
                raise ValueError(f"Sheet {sheet_name!r} contains missing Quantity or Price values.")
            frame["_source_sheet"] = sheet_name
            sheet_summary.append(
                {
                    "sheet": sheet_name,
                    "rows": int(len(frame)),
                    "exact_duplicate_rows": int(frame.duplicated(subset=REQUIRED_COLUMNS).sum()),
                    "date_start": frame["InvoiceDate"].min().isoformat(),
                    "date_end": frame["InvoiceDate"].max().isoformat(),
                }
            )
            frames.append(frame)

    transactions = pd.concat(frames, ignore_index=True)
    initial_rows = len(transactions)
    transactions = _remove_cross_sheet_overlap(transactions)
    overlap_rows_removed = initial_rows - len(transactions)
    transactions = transactions.drop(columns="_source_sheet")
    transactions = transactions.reset_index(drop=True)
    return transactions, {
        "source_file": workbook_path.name,
        "source_rows": int(initial_rows),
        "rows_after_sheet_overlap_reconciliation": int(len(transactions)),
        "cross_sheet_rows_removed": int(overlap_rows_removed),
        "sheets": sheet_summary,
    }


def _remove_cross_sheet_overlap(transactions: pd.DataFrame) -> pd.DataFrame:
    """Keep the greatest within-sheet multiplicity of each exact row across sheets."""
    sheet_counts = (
        transactions.groupby(ROW_KEY + ["_source_sheet"], dropna=False, sort=False)
        .size()
        .rename("sheet_occurrences")
        .reset_index()
    )
    max_counts = (
        sheet_counts.groupby(ROW_KEY, dropna=False, sort=False)["sheet_occurrences"]
        .max()
        .rename("_max_occurrences")
        .reset_index()
    )
    rows = transactions.copy()
    rows["_row_order"] = np.arange(len(rows))
    rows = rows.merge(max_counts, on=ROW_KEY, how="left", sort=False, validate="many_to_one")
    rows["_occurrence"] = rows.groupby(ROW_KEY, dropna=False, sort=False).cumcount()
    rows = rows.loc[rows["_occurrence"] < rows["_max_occurrences"]]
    return rows.sort_values("_row_order", kind="stable").drop(
        columns=["_max_occurrences", "_occurrence", "_row_order"]
    )


def prepare_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add transaction-level labels and a transparent net line-sales measure."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(transactions.columns))
    if missing_columns:
        raise ValueError(f"Transactions are missing required columns: {missing_columns}")
    prepared = transactions.copy()
    prepared["InvoiceDate"] = pd.to_datetime(prepared["InvoiceDate"], errors="coerce")
    if prepared["InvoiceDate"].isna().any():
        raise ValueError("Transactions contain invalid InvoiceDate values.")
    for column in ("Quantity", "Price"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        if prepared[column].isna().any():
            raise ValueError(f"Transactions contain invalid {column} values.")

    invoice = prepared["Invoice"].astype("string").str.strip()
    prepared["IsCancellation"] = invoice.str.startswith("C", na=False)
    prepared["IsAdjustment"] = invoice.str.startswith("A", na=False)
    prepared["LineRevenue"] = prepared["Quantity"] * prepared["Price"]
    prepared["SalesRevenue"] = prepared["LineRevenue"].where(~prepared["IsAdjustment"], 0.0)
    prepared["InvoiceDateDay"] = prepared["InvoiceDate"].dt.normalize()
    return prepared


def build_daily_sales(transactions: pd.DataFrame, country: str | None = None) -> pd.Series:
    """Aggregate net line revenue to a complete calendar-day index."""
    selected = transactions
    if country is not None:
        selected = selected.loc[selected["Country"].eq(country)]
    if selected.empty:
        raise ValueError(f"No transaction rows found for country {country!r}.")

    daily = selected.groupby("InvoiceDateDay", sort=True)["SalesRevenue"].sum()
    full_dates = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="date")
    return daily.reindex(full_dates, fill_value=0.0).astype(float).rename("sales")


def create_data_summary(
    raw_transactions: pd.DataFrame,
    transactions: pd.DataFrame,
    load_summary: dict[str, Any],
) -> dict[str, Any]:
    """Create serializable data-quality and business-summary statistics."""
    row_count = len(raw_transactions)
    missing_values = {
        str(column): {
            "count": int(count),
            "percent": round(float(count / row_count * 100), 3) if row_count else 0.0,
        }
        for column, count in raw_transactions.isna().sum().items()
    }
    daily = build_daily_sales(transactions)
    country_rows = transactions["Country"].value_counts(dropna=False)
    return {
        **load_summary,
        "rows_after_overlap_reconciliation": int(len(transactions)),
        "columns": [
            {"name": str(column), "dtype": str(dtype)}
            for column, dtype in raw_transactions.dtypes.items()
        ],
        "missing_values": missing_values,
        "exact_duplicate_rows_after_overlap_reconciliation": int(
            transactions.duplicated(subset=REQUIRED_COLUMNS).sum()
        ),
        "invalid_dates": int(pd.to_datetime(raw_transactions["InvoiceDate"], errors="coerce").isna().sum()),
        "unique_invoices": int(transactions["Invoice"].nunique()),
        "unique_customers": int(transactions["Customer ID"].nunique()),
        "unique_products": int(transactions["StockCode"].nunique()),
        "countries": int(transactions["Country"].nunique()),
        "uk_row_share_percent": round(
            float(transactions["Country"].eq("United Kingdom").mean() * 100), 2
        ),
        "date_start": transactions["InvoiceDate"].min().isoformat(),
        "date_end": transactions["InvoiceDate"].max().isoformat(),
        "days_with_transactions": int(transactions["InvoiceDateDay"].nunique()),
        "calendar_days": int(len(daily)),
        "missing_description_rows": int(transactions["Description"].isna().sum()),
        "missing_customer_id_rows": int(transactions["Customer ID"].isna().sum()),
        "cancellation_rows": int(transactions["IsCancellation"].sum()),
        "adjustment_rows": int(transactions["IsAdjustment"].sum()),
        "negative_quantity_rows": int(transactions["Quantity"].lt(0).sum()),
        "zero_price_rows": int(transactions["Price"].eq(0).sum()),
        "negative_price_rows": int(transactions["Price"].lt(0).sum()),
        "quantity_over_1000_rows": int(transactions["Quantity"].gt(1000).sum()),
        "price_over_1000_rows": int(transactions["Price"].gt(1000).sum()),
        "net_sales_revenue": round(float(transactions["SalesRevenue"].sum()), 2),
        "adjustment_revenue_excluded": round(
            float(transactions.loc[transactions["IsAdjustment"], "LineRevenue"].sum()), 2
        ),
        "top_countries_by_rows": [
            {"country": str(country), "rows": int(count)}
            for country, count in country_rows.head(10).items()
        ],
    }
