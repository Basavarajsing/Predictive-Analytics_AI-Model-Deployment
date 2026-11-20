from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predictive_analytics.config import ARTIFACT_PATH, DATA_PATH
from src.predictive_analytics.data import build_daily_sales, load_transactions, prepare_transactions
from src.predictive_analytics.forecasting import forecast_values

APP_STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        APP_STATE["artifact"] = joblib.load(ARTIFACT_PATH)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Trained artifact not found at {ARTIFACT_PATH}. Run python -m src.predictive_analytics.train."
        ) from exc
    try:
        raw, _ = load_transactions(DATA_PATH)
        transactions = prepare_transactions(raw)
        APP_STATE["daily_sales"] = {
            "All countries": build_daily_sales(transactions),
        }
        for country in sorted(transactions["Country"].dropna().unique()):
            APP_STATE["daily_sales"][str(country)] = build_daily_sales(transactions, country=str(country))
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"Unable to prepare API daily-sales data: {exc}") from exc
    yield
    APP_STATE.clear()


app = FastAPI(
    title="Retail Net-Sales Forecast API",
    version="1.0.0",
    description="One-day and short-horizon daily net-sales forecasting for Online Retail II.",
    lifespan=lifespan,
)


class ForecastRequest(BaseModel):
    horizon: int = Field(default=7, ge=1, le=30)


class ForecastResponse(BaseModel):
    model_name: str
    horizon: int
    forecasts: list[dict[str, str | float]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": str(APP_STATE["artifact"]["model_name"])}


@app.get("/data-summary")
def data_summary() -> dict[str, Any]:
    return APP_STATE["artifact"]["data_summary"]


@app.get("/model-comparison")
def model_comparison() -> list[dict[str, Any]]:
    artifact = APP_STATE["artifact"]
    metrics = artifact["metrics"]
    comparison_path = ARTIFACT_PATH.parent.parent / "outputs" / "model_comparison.csv"
    if comparison_path.exists():
        records = pd.read_csv(comparison_path).fillna("").to_dict(orient="records")
    else:
        records = [metrics]
    return records


@app.get("/holdout")
def holdout() -> dict[str, Any]:
    artifact = APP_STATE["artifact"]
    return {
        "model_name": artifact["model_name"],
        "metrics": artifact["metrics"],
        "predictions": artifact["holdout_predictions"],
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest) -> ForecastResponse:
    if "artifact" not in APP_STATE:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
    artifact = APP_STATE["artifact"]
    history = pd.Series(
        {pd.Timestamp(row["date"]): row["sales"] for row in artifact["history"]},
        name="sales",
        dtype=float,
    )
    forecasts = forecast_values(artifact, history, request.horizon)
    return ForecastResponse(
        model_name=str(artifact["model_name"]),
        horizon=request.horizon,
        forecasts=forecasts,
    )


@app.get("/daily-sales")
def daily_sales(country: str | None = None) -> list[dict[str, str | float]]:
    country_key = country if country else "All countries"
    series = APP_STATE["daily_sales"].get(country_key)
    if series is None:
        raise HTTPException(status_code=404, detail=f"Unknown country: {country}")
    return [
        {"date": timestamp.date().isoformat(), "sales": float(value)}
        for timestamp, value in series.items()
    ]
