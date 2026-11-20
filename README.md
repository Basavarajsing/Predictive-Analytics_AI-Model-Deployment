# Predictive Analytics & AI Model Deployment

An end-to-end retail analytics project using the **Online Retail II** transaction workbook. It trains and evaluates daily net-sales forecasting models, serves forecasts through FastAPI, and presents the data and model results in a Streamlit dashboard.

## What the project does

- Loads and validates both workbook sheets without changing the source file.
- Reconciles exact rows duplicated across the two overlapping sheets.
- Creates daily net sales as `Quantity × Price`. Cancellation rows remain negative sales; `A`-prefixed adjustment invoices are identified and excluded from the target.
- Creates calendar, lag, and rolling-average features using only prior sales.
- Compares a 7-day seasonal-naive baseline, Ridge Regression, Random Forest, and HistGradientBoosting.
- Selects the model by rolling time-series validation MAE and reports chronological holdout MAE, RMSE, R², and WAPE.
- Serves the trained model and sales data through FastAPI.
- Displays dataset statistics, sales charts, model comparisons, holdout predictions, and forecasts in Streamlit.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.11 or newer (the project has been tested with Python 3.13)
- The Online Retail II workbook at `Data/raw/online_retail_II.xlsx`
- Internet access for the initial Python package installation

Check your Python version:

```powershell
python --version
```

If `python` is not recognized on Windows, install Python and enable **Add Python to PATH**, or use the Python launcher:

```powershell
py --list
```

## Setup on Windows

Open PowerShell in this project folder:

```powershell
cd "C:\Users\basav\OneDrive\Pictures\Desktop\7th Sem\Cognevance\Predictive_Analytics_AI"
```

Create and activate a virtual environment using the Python version available on your computer:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If your computer uses the Python launcher instead, replace the first command with a version that `py --list` shows, for example:

```powershell
py -3.13 -m venv .venv
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The virtual environment is local to your computer and is excluded from Git.

## Dataset setup

Put the original workbook here:

```text
Data/raw/online_retail_II.xlsx
```

The workbook is not included in this repository because it is a large raw dataset. Obtain the Online Retail II dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place the workbook at the path above. Do not edit or overwrite the raw workbook.

## Run the project

### 1. Train and evaluate the models

From the project folder, with `.venv` activated:

```powershell
python run_pipeline.py
```

This reads the real workbook and creates:

- `artifacts/forecast_model.joblib` — selected model and inference data
- `outputs/model_comparison.csv` — validation and holdout comparison for all candidate models
- `outputs/evaluation_summary.json` — selected-model metrics
- `outputs/eda_summary.json` and `docs/reports/eda_report.md` — dataset quality and EDA summary
- `outputs/monthly_sales.csv` — monthly net sales
- `docs/reports/model_evaluation_report.md` — generated evaluation report

These are generated locally and excluded from Git.

### 2. Start the API

Open a **second** PowerShell terminal in the project folder. Activate the virtual environment there too:

```powershell
cd "C:\Users\basav\OneDrive\Pictures\Desktop\7th Sem\Cognevance\Predictive_Analytics_AI"
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload
```

The API reads the model artifact and prepares daily sales when it starts. On first startup, loading the workbook can take a little time.

- API: <http://127.0.0.1:8000>
- Interactive API documentation: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

### 3. Start the dashboard

Open a **third** PowerShell terminal in the project folder, activate `.venv`, and run:

```powershell
cd "C:\Users\basav\OneDrive\Pictures\Desktop\7th Sem\Cognevance\Predictive_Analytics_AI"
.\.venv\Scripts\Activate.ps1
python -m streamlit run dashboard/app.py
```

Open the dashboard at <http://127.0.0.1:8501>. Keep the API terminal and dashboard terminal open while using the app.

### Run tests

```powershell
python -m pytest -q
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | API and selected model status |
| `GET` | `/data-summary` | Dataset size, date range, and quality summary |
| `GET` | `/model-comparison` | Candidate model results |
| `GET` | `/holdout` | Actual and predicted holdout sales with metrics |
| `GET` | `/daily-sales` | Daily net sales for all countries |
| `GET` | `/daily-sales?country=United%20Kingdom` | Daily net sales filtered by country |
| `POST` | `/predict` | Recursive forecast for 1–30 days |

Example request body for `POST /predict`:

```json
{
  "horizon": 7
}
```

See [`docs/api.md`](docs/api.md) for details.

## Docker

Docker deployment requires Docker Desktop (Windows/macOS) or Docker Engine (Linux). First train the model locally so `artifacts/forecast_model.joblib` exists. Ensure the workbook is in `Data/raw/`, then run:

```powershell
docker compose up --build
```

The API is available at port `8000`; the dashboard is available at port `8501`. Stop the services with `Ctrl+C`, or run `docker compose down` in another terminal.

## Repository structure

```text
Data/raw/                    Local raw workbook (not committed)
src/predictive_analytics/    Data loading, features, forecasting, training
api/                         FastAPI application
dashboard/                   Streamlit dashboard
tests/                       Data, feature, and API tests
docs/                        Architecture, API guide, generated reports
artifacts/                   Generated model artifact (not committed)
outputs/                     Generated data and evaluation files (not committed)
requirements.txt             Python dependencies
run_pipeline.py              Training entry point
Dockerfile                   Container image
docker-compose.yml           API and dashboard services
```

## Modeling and limitations

The training pipeline uses a chronological 80/20 holdout and four expanding-window time-series validation folds. Model selection uses validation MAE; the final holdout is reported separately. MAPE is not used because daily actual sales may be zero or close to zero. A fixed performance score is not assumed: reports are generated from the workbook on each training run.

The dataset covers about two years and ends with a partial December. Recursive multi-day forecasts are less reliable than one-day-ahead holdout metrics. Customer identifiers are missing for a substantial portion of transactions, so complete customer-level conclusions are not supported. See [`docs/architecture.md`](docs/architecture.md) and the generated reports under `docs/reports/`.

## Troubleshooting

- **`Activate.ps1` not found:** Create the environment first with `python -m venv .venv` from the project root.
- **PowerShell blocks activation:** Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.
- **Python version is unavailable:** Use `python -m venv .venv` for the installed default, or select an installed version shown by `py --list`.
- **Dataset not found:** Confirm the workbook is named exactly `online_retail_II.xlsx` and is in `Data/raw/`.
- **Model artifact not found:** Run `python run_pipeline.py` before starting the API.
- **Port 8000 or 8501 is already in use:** Stop the other server using that port, or start this service with another port and update the dashboard's `PREDICTIVE_API_URL` if the API port changes.
- **Dashboard forecast says API is unavailable:** Confirm the API is running at `http://127.0.0.1:8000` and that the model artifact was generated.
