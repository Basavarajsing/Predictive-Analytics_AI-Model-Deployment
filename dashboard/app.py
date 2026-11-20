import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictive_analytics.config import ARTIFACT_PATH

API_URL = os.getenv("PREDICTIVE_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Retail Forecast Analytics", page_icon="📈", layout="wide")
st.title("Retail Net-Sales Analytics & Forecasting")
st.caption("Online Retail II | Daily net-sales forecasting")


@st.cache_data(ttl=60)
def api_get(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_resource
def load_artifact():
    import joblib

    return joblib.load(ARTIFACT_PATH)


try:
    artifact = load_artifact()
except Exception as exc:
    st.error(f"Model artifact is unavailable. Run `python -m src.predictive_analytics.train`. ({exc})")
    st.stop()

summary = artifact["data_summary"]
comparison = pd.read_csv(PROJECT_ROOT / "outputs" / "model_comparison.csv")
holdout = pd.DataFrame(artifact["holdout_predictions"])
history = pd.DataFrame(artifact["history"])
history["date"] = pd.to_datetime(history["date"])
holdout["date"] = pd.to_datetime(holdout["date"])

page = st.sidebar.radio(
    "Dashboard",
    ["Overview", "Sales Analysis", "Model Performance", "Prediction"],
)

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transaction rows", f"{summary['rows_after_overlap_reconciliation']:,.0f}")
    c2.metric("Unique customers", f"{summary['unique_customers']:,.0f}")
    c3.metric("Countries", f"{summary['countries']:,}")
    c4.metric("Net sales", f"{summary['net_sales_revenue']:,.0f}")

    st.subheader("Dataset coverage")
    coverage = pd.DataFrame(
        [
            ["Date start", summary["date_start"][:10]],
            ["Date end", summary["date_end"][:10]],
            ["Calendar days", f"{summary['calendar_days']:,}"],
            ["Unique products", f"{summary['unique_products']:,}"],
            ["Missing Customer ID", f"{summary['missing_customer_id_rows']:,}"],
            ["Missing Description", f"{summary['missing_description_rows']:,}"],
            ["Cancellation rows", f"{summary['cancellation_rows']:,}"],
            ["Adjustment rows excluded", f"{summary['adjustment_rows']:,}"],
        ],
        columns=["Item", "Value"],
    )
    st.dataframe(coverage, width="stretch", hide_index=True)

    st.subheader("Top countries by row count")
    top_countries = pd.DataFrame(summary["top_countries_by_rows"])
    st.plotly_chart(
        px.bar(top_countries, x="country", y="rows", color="country", title="Top Countries"),
        width="stretch",
    )

elif page == "Sales Analysis":
    st.subheader("Daily net sales")
    country = st.selectbox(
        "Country",
        ["All countries", *[row["country"] for row in summary["top_countries_by_rows"]]],
    )
    try:
        if country == "All countries":
            records = api_get("/daily-sales")
        else:
            records = api_get("/daily-sales?country=" + country.replace(" ", "%20"))
        daily = pd.DataFrame(records)
        daily["date"] = pd.to_datetime(daily["date"])
    except Exception as exc:
        st.warning(f"API is unavailable; showing the locally loaded all-country history. ({exc})")
        daily = history
    fig = px.line(daily, x="date", y="sales", title=f"Daily Net Sales - {country}")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Monthly aggregation")
    monthly = daily.set_index("date")["sales"].resample("MS").sum().reset_index()
    monthly.rename(columns={"sales": "net_sales"}, inplace=True)
    st.plotly_chart(
        px.bar(monthly, x="date", y="net_sales", title="Monthly Net Sales"),
        width="stretch",
    )

elif page == "Model Performance":
    st.subheader("Model comparison")
    st.dataframe(comparison.round(3), width="stretch", hide_index=True)
    fig = px.bar(comparison, x="model", y="validation_mae", title="Rolling Validation MAE")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Actual vs predicted - chronological holdout")
    line = go.Figure()
    line.add_trace(go.Scatter(x=holdout["date"], y=holdout["actual_sales"], name="Actual"))
    line.add_trace(go.Scatter(x=holdout["date"], y=holdout["predicted_sales"], name="Predicted"))
    line.update_layout(title=f"Selected Model: {artifact['model_name']}", hovermode="x unified")
    st.plotly_chart(line, width="stretch")

    m = artifact["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Holdout MAE", f"{m['mae']:,.2f}")
    c2.metric("Holdout RMSE", f"{m['rmse']:,.2f}")
    c3.metric("Holdout R²", f"{m['r2']:.4f}")
    c4.metric("Holdout WAPE", f"{m['wape_percent']:.2f}%")

else:
    st.subheader("Generate a short-horizon forecast")
    horizon = st.slider("Forecast horizon (days)", 1, 30, 7)
    if st.button("Generate forecast", type="primary"):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"horizon": horizon},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            forecast = pd.DataFrame(result["forecasts"])
            forecast["date"] = pd.to_datetime(forecast["date"])
            st.success(f"Forecast from selected model: {result['model_name']}")
            st.dataframe(forecast, width="stretch", hide_index=True)
            st.plotly_chart(
                px.line(forecast, x="date", y="predicted_sales", markers=True, title="Forecast"),
                width="stretch",
            )
        except requests.RequestException as exc:
            st.error(
                f"Could not reach the API. Start it with `python -m uvicorn api.main:app --reload`. ({exc})"
            )
