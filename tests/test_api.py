import joblib

from fastapi.testclient import TestClient

import api.main as api_main


def test_health_and_forecast_endpoints(monkeypatch, tmp_path):
    artifact = {
        "model_name": "Seasonal Naive (7-day)",
        "estimator": None,
        "feature_columns": [],
        "history": [
            {"date": f"2024-01-{day:02d}", "sales": float(day)}
            for day in range(1, 29)
        ],
        "metrics": {"mae": 1.0},
        "holdout_predictions": [],
        "data_summary": {"calendar_days": 28},
    }
    artifact_path = tmp_path / "forecast_model.joblib"
    joblib.dump(artifact, artifact_path)
    monkeypatch.setattr(api_main, "ARTIFACT_PATH", artifact_path)

    with TestClient(api_main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        forecast = client.post("/predict", json={"horizon": 3})
        assert forecast.status_code == 200
        assert len(forecast.json()["forecasts"]) == 3
