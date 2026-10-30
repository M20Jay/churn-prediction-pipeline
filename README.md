# Churn Prediction Pipeline
[![CI](https://github.com/M20Jay/churn-prediction-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/M20Jay/churn-prediction-pipeline/actions/workflows/ci.yml)
**Martin James Ng'ang'a — ML / MLOps Engineer | GitHub: M20Jay | Nairobi, Kenya**

---

## Overview
An end-to-end churn prediction system for a telecom company (7,043 customers, 26.5% churn). The model is evaluated without leakage, the decision threshold comes from a cost rule instead of the default 0.5, and the saved pipeline is served through a validated FastAPI service in Docker. Predictions are stored in PostgreSQL and monitored in Grafana.

---

## Key results (honest evaluation)

| Metric | Result |
|---|---|
| Cross-validated ROC-AUC (5-fold, training data only) | 0.848 ± 0.011 |
| Held-out test ROC-AUC | 0.8458 |
| Test PR-AUC (no-skill baseline = 0.265) | 0.6576 |
| Test Brier score (baseline ≈ 0.195) | 0.1358 |
| Decision threshold | 0.198 (from the cost rule) |
| Test recall / precision at that threshold | 0.858 / 0.471 |
| Test cost (1,409 customers) | 946, versus 1,870 offering no one and 1,409 offering everyone |

With an offer costing 1 and a lost customer costing 5, the model's decisions cost about 49% less than doing nothing and about 33% less than offering everyone.

---

## What went wrong first, and how it was fixed
The first version resampled (SMOTE) **before** cross-validation, which leaked information across folds. It reported a cross-validated ROC-AUC of 0.9283, but only 0.8159 on held-out data. The notebook `notebooks/04_rigour_pass.ipynb` rebuilds the evaluation:

1. Split first and leave the test set untouched.
2. Compare models with cross-validation **inside the training data**, using the full pipeline.
3. Get out-of-fold probabilities for every training row.
4. Choose the threshold from those out-of-fold predictions using the cost ratio.
5. Check calibration.
6. Fit the final model on all training data and look at the test set once.

### Model comparison (5-fold stratified CV, training data)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic regression + SMOTE | 0.847 ± 0.012 | 0.666 ± 0.017 |
| Logistic regression + class weights | 0.848 ± 0.011 | 0.668 ± 0.014 |
| Logistic regression, no resampling | 0.848 ± 0.011 | 0.669 ± 0.015 |
| XGBoost + SMOTE | 0.823 ± 0.008 | 0.629 ± 0.022 |
| XGBoost + class weights | 0.822 ± 0.005 | 0.620 ± 0.010 |
| XGBoost, no resampling | 0.824 ± 0.007 | 0.629 ± 0.016 |

Logistic regression beat XGBoost, and resampling made no measurable difference, so the simplest option (no resampling) was kept.

### Calibration
The out-of-fold Brier score of 0.1337 (baseline 0.1949) and the calibration curve show the probabilities are already well calibrated. Isotonic calibration did not help (0.1342), so it is not used.

---

## Customer segments
Every customer gets an out-of-fold churn probability (no customer is scored by a model that saw them). Customers are then split by risk (probability ≥ 0.198) and value (monthly charge above the median, about 70).

| Segment | Customers | Actual churn | Avg monthly charge | Revenue at risk |
|---|---|---|---|---|
| Save now (high risk, high value) | 2,217 | 50.9% | 89.45 | 100,751 |
| Cheap offer (high risk, low value) | 1,233 | 40.1% | 45.16 | 23,926 |
| Protect (low risk, high value) | 1,307 | 8.4% | 92.47 | 9,346 |
| Monitor (low risk, low value) | 2,286 | 6.0% | 35.55 | 5,121 |

Revenue at risk is the sum of churn probability × monthly charge. "Save now" is 31% of customers but about 72% of the revenue at risk.

---

## API
A FastAPI service loads the saved pipeline (`src/churn_pipeline.pkl`) and threshold (`src/threshold.json`) once at startup. It takes raw customer fields, validates them against the allowed categories, builds the engineered features itself (so training and serving cannot drift apart), and logs predictions to PostgreSQL.

- `GET /health` returns the model name, version and threshold.
- `POST /predict` returns the churn probability, the decision, the threshold used and a risk level.
- Invalid values (for example `"fiber optic"` in lowercase) are rejected with a 422 error that names the field.

Example request:

```
curl -sS -X POST http://127.0.0.1:5001/predict -H "Content-Type: application/json" -d '{"customer_id":"TEST-1","gender":"Female","senior_citizen":0,"partner":"No","dependents":"No","tenure":2,"phone_service":"Yes","multiple_lines":"No","internet_service":"Fiber optic","online_security":"No","online_backup":"No","device_protection":"No","tech_support":"No","streaming_tv":"No","streaming_movies":"No","contract":"Month-to-month","paperless_billing":"Yes","payment_method":"Electronic check","monthly_charges":70.7,"total_charges":151.65}'
```

Example response:

```
{"customer_id":"TEST-1","churn_probability":0.7054,"churn_predicted":true,"threshold_used":0.1984,"risk_level":"HIGH"}
```

---

## How to run
With Docker:

```
docker build -t churn-api .
docker compose up -d
curl -sS http://127.0.0.1:5001/health
docker compose down
```

Without Docker (port 8000, because macOS uses port 5000 for AirPlay):

```
pip install -r src/requirements.txt
uvicorn src.app:app --port 8000
```

Prediction logging needs PostgreSQL. If it is unreachable, the API logs a warning and still returns the prediction. Library versions in `src/requirements.txt` are pinned to the versions used to train the saved pipeline.

---

## Tests and CI

The API has six automated tests in `tests/test_api.py`: a health check, the response fields, a high-risk and a low-risk customer, and two invalid-input cases that must return 422.

Run them locally:

    python3 -m venv .venv && source .venv/bin/activate
    python -m pip install -r src/requirements.txt -r requirements-dev.txt
    python -m pytest -q

GitHub Actions (`.github/workflows/ci.yml`) runs the same tests on every pull request and every push to `main`. A ruleset on `main` requires the `test` check to pass before anything can be merged.

## Tech stack

| Tool | Purpose |
|---|---|
| scikit-learn, imbalanced-learn | Pipeline and logistic regression model |
| FastAPI, Pydantic | Serving and input validation |
| Docker, Docker Compose | Containerised deployment |
| PostgreSQL | Storing predictions |
| Grafana | Monitoring dashboard |
| joblib | Saving the fitted pipeline |

---

## Where This Fits

A churn probability score is not a retention outcome. It's an input that only creates value once it reaches the right team, with the right context:

- **Retention/CRM campaign targeting** — the "Save now" segment isn't just a label, it's a priority queue: these are the customers worth a retention agent's time or a targeted discount, not a generic mass email.
- **Customer service prioritization** — a churn score can route high-risk, high-value customers to a different support tier — faster response times, a human agent instead of a bot, before frustration becomes cancellation.
- **Revenue forecasting** — ARPU-weighted churn risk feeds directly into revenue-at-risk projections finance teams use for forecasting, not just a churn percentage in isolation.
- **Product/pricing decisions** — patterns in why high-value customers churn can inform pricing tier or product changes, not just reactive retention offers.
- **Feedback loop** — whether a targeted retention action actually worked is the real label that should feed back into the next model retrain, otherwise the model never learns whether its "at risk" calls were acted on successfully.

This pipeline currently ends at prediction and segmentation. The natural next integration is a CRM or retention-campaign system that consumes the segment output directly.

---

## Limitations
- The costs (1 per offer, 5 per lost customer) are assumptions. The threshold should be recomputed with the business's real figures.
- The dataset is a single snapshot with no timestamps, so there is no time-based split.
- Retention outcomes are not fed back into training.
- The Grafana dashboard was built on version 1 predictions. Version 2 logs `model_version = v2.0`.

---

## Project history
Version 1 (week 1) used XGBoost with SMOTE, a Flask API and KMeans segments. Version 2 (the rigour pass) replaced it with the leakage-free evaluation, logistic regression pipeline, cost-based threshold and FastAPI service described above.

## Key learnings
- Resampling must happen inside cross-validation, never before it.
- Choose thresholds from out-of-fold predictions and look at the test set once.
- Training and serving must match exactly: keep preprocessing inside the saved pipeline, and build engineered features in one place.
- Pin library versions so a saved model loads the same way everywhere.
- Docker needs host.docker.internal to reach host PostgreSQL.

---

## Next
Credit card fraud detection with Kafka and Redis for real-time streaming under 200 ms, then the same evaluation method on credit risk.
