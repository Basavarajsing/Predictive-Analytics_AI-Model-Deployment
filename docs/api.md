# API

Start the API after training:

```powershell
python -m uvicorn api.main:app --reload
```

Interactive documentation is available at `http://127.0.0.1:8000/docs`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | API and selected-model status |
| GET | `/data-summary` | Dataset quality and coverage summary |
| GET | `/model-comparison` | Candidate model metrics |
| GET | `/holdout` | Holdout actual/predicted values and metrics |
| GET | `/daily-sales?country=United Kingdom` | Complete calendar-day sales; omit country for all countries |
| POST | `/predict` | Recursive forecast for 1–30 days |

Example forecast request:

```json
{"horizon": 7}
```
