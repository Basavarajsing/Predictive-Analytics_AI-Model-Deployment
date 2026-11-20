import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.getenv("PREDICTIVE_DATA_PATH", PROJECT_ROOT / "Data" / "raw" / "online_retail_II.xlsx"))
ARTIFACT_PATH = Path(os.getenv("PREDICTIVE_ARTIFACT_PATH", PROJECT_ROOT / "artifacts" / "forecast_model.joblib"))
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "docs" / "reports"
RANDOM_STATE = 42
