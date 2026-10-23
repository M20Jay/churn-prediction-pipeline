# Week 1: Customer Churn Prediction Pipeline

**Martin James Ng'ang'a · ML / MLOps Engineer · Nairobi, Kenya · github.com/M20Jay**

> **How to read these notes.** The first section (v2) is current and correct. Everything after it is the original Week 1 log (v1), kept as a learning record. Where the two disagree, v2 is right.

---

# v2: Rigour pass (current)

The Week 1 model was re-evaluated honestly in `notebooks/04_rigour_pass.ipynb`, then served through a rebuilt API.

## Results

| Metric | Result |
|---|---|
| Dataset | 7,043 telecom customers, 26.5% churn |
| Cross-validated ROC-AUC (training data only) | 0.848 ± 0.011 (logistic regression) |
| Held-out test ROC-AUC | 0.8458 |
| Test PR-AUC (no-skill baseline 0.265) | 0.6576 |
| Test Brier score | 0.1358 (out-of-fold 0.1337, baseline 0.1949) |
| Threshold | 0.198, from the cost rule (offer = 1, lost customer = 5) |
| Test recall / precision | 0.858 / 0.471 |
| Test cost (1,409 customers) | 946, versus 1,870 (offer no one) and 1,409 (offer everyone) |

## What went wrong in v1
Resampling (SMOTE) was applied **before** cross-validation. That leaks information into the validation folds and inflated the cross-validated ROC-AUC to 0.9283, against 0.8159 on held-out data.

## The honest evaluation recipe
1. Split first and leave the test set alone.
2. Compare models with `cross_val_score` inside the training data, using the full pipeline (resampling, scaling and encoding happen inside each fold).
3. Get one out-of-fold probability per training row with `cross_val_predict(..., method="predict_proba")`.
4. Choose the threshold from those out-of-fold probabilities using the cost of each mistake.
5. Check calibration (Brier score, calibration curve) before trusting probabilities.
6. Fit the final model on all the training data and look at the test set once.

`cross_val_score` returns one score per fold and is used to compare models. `cross_val_predict` returns one probability per row and is used for thresholds, calibration and segments.

## Decision rule
Sending an offer costs 1. Losing a customer costs 5. Offer when probability × 5 > 1, so the best threshold is about 1/5 = 0.2 when probabilities are calibrated. The sweep on out-of-fold predictions found 0.198.

## Confusion matrix order
`tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()`. The order is fixed: rows are actual classes, columns are predicted, class 0 first, and `ravel()` reads row by row.

## Model choice
Logistic regression (0.848 ROC-AUC) beat XGBoost (0.824). SMOTE, class weights and no resampling were indistinguishable (within ±0.01), so the simplest option, no resampling, was kept. The model was already well calibrated, and isotonic calibration did not help (Brier 0.1342 vs 0.1337).

## Segments (replaces KMeans)
Segments come from the out-of-fold churn probability and monthly charge. High risk means probability ≥ 0.198. High value means monthly charge above the median (about 70).

| Segment | Customers | Actual churn | Avg monthly charge | Revenue at risk |
|---|---|---|---|---|
| Save now | 2,217 | 50.9% | 89.45 | 100,751 |
| Cheap offer | 1,233 | 40.1% | 45.16 | 23,926 |
| Protect | 1,307 | 8.4% | 92.47 | 9,346 |
| Monitor | 2,286 | 6.0% | 35.55 | 5,121 |

Revenue at risk is the sum of churn probability × monthly charge. "Save now" is 31% of customers and about 72% of revenue at risk.

## API (FastAPI)
- The saved pipeline (`src/churn_pipeline.pkl`) and threshold (`src/threshold.json`) are loaded once at startup.
- Pydantic validates every field against the allowed categories, so a typo returns a 422 error naming the field.
- The three engineered columns (`tenure_group`, `charges_per_month`, `is_high_value`) are built inside the API in one place, so training and serving cannot drift apart.
- Library versions are pinned in `src/requirements.txt` because a pickle saved with one scikit-learn version may not load on another.
- The old setup label-encoded columns with `LabelEncoder` and never saved the encoders. The new pipeline keeps all encoding inside the saved pipeline.

## Commands (v2)
```
# Build and run
docker build -t churn-api .
docker compose up -d
curl -sS http://127.0.0.1:5001/health
docker compose down

# Run locally without Docker (port 8000, because macOS uses 5000 for AirPlay)
uvicorn src.app:app --port 8000

# Example prediction
curl -sS -X POST http://127.0.0.1:5001/predict -H "Content-Type: application/json" -d '{"customer_id":"TEST-1","gender":"Female","senior_citizen":0,"partner":"No","dependents":"No","tenure":2,"phone_service":"Yes","multiple_lines":"No","internet_service":"Fiber optic","online_security":"No","online_backup":"No","device_protection":"No","tech_support":"No","streaming_tv":"No","streaming_movies":"No","contract":"Month-to-month","paperless_billing":"Yes","payment_method":"Electronic check","monthly_charges":70.7,"total_charges":151.65}'

# Check that a change made it into a commit
git show HEAD:path/to/file | grep -c "distinctive text"
```

## Gotchas learned
- macOS uses port 5000 for AirPlay Receiver. Run local servers on another port and call 127.0.0.1, not localhost.
- A notebook must be saved (Cmd+S) before committing, or the commit misses the changes.
- Typing `git` correctly matters: a misspelt command silently skips the step.
- `grep -c` counts matching lines, `-n` shows line numbers, `-r` searches folders.
- `sed -i` needs an empty string after it on a Mac (`sed -i '' 's/a/b/' file`).

## Git workflow (branch and pull request)
```
git checkout -b my-branch
git add <files>
git commit -m "message"
git push -u origin my-branch
# open the pull request on GitHub, merge it, then:
git checkout main
git pull
```

---

# v1: Original Week 1 log (kept for history)

## Overview
End-to-end MLOps pipeline that predicts customer churn for a telecom company using real customer data. Scores customers in real time, stores predictions in PostgreSQL, segments customers by value and risk, and monitors everything through a live Grafana dashboard.

This was the foundation week. Every pattern learned here repeats across the later projects.

## v1 results

| Metric | Result |
|---|---|
| Dataset | 7,043 telecom customers |
| Model | XGBoost classifier (v1) |
| Segments | 4 KMeans clusters (v1) |
| Endpoints | GET /health, POST /predict |
| Grafana panels | 4 panels: segments, churn probability, churn rate, total predictions |
| Deployment | AWS EC2 (currently stopped) |

### v1 customer segments (KMeans)

| Segment | Customers | Avg ARPU | Priority |
|---|---|---|---|
| High Value At Risk | 1,693 | $98.4 | ACT NOW |
| High Value Loyal | 1,851 | $73.8 | Maintain |
| Low Value At Risk | 1,524 | $47.7 | Monitor |
| Low Value Loyal | 1,975 | $20.1 | Standard |

ARPU = Average Revenue Per User. High ARPU + High churn risk = call them today.

## 7-day build plan

| Day | Task | Status |
|---|---|---|
| Day 1 | EDA + Feature Engineering | done |
| Day 2 | XGBoost model training | done |
| Day 3 | Flask API: /health + /predict | done |
| Day 4 | Docker container | done |
| Day 5 | Grafana dashboard | done |
| Day 6 | ARPU analysis + KMeans segmentation | done |
| Day 7 | README + Deploy to AWS EC2 | done |

## Original project structure plan
The repo as built is flatter (for example `src/app.py`, `src/load_data.py`, `src/requirements.txt`).

```
churn-prediction-pipeline/
├── data/                       Raw and processed data
├── src/
│   ├── data/preprocessing.py   Load, clean, encode features
│   ├── features/feature_engineering.py   ARPU, tenure groups, KMeans
│   ├── models/train.py         XGBoost training + SMOTE
│   ├── models/evaluate.py      Metrics, confusion matrix
│   └── api/app.py, schemas.py  Flask app and validation
├── grafana/dashboards/         Grafana JSON dashboard configs
├── sql/init.sql                PostgreSQL table creation
├── notebooks/                  EDA, modelling, rigour pass
├── reports/                    Model evaluation reports
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## v1 pipeline architecture
```
Raw telecom data (CSV)
    -> preprocessing        -> cleaned df
    -> feature engineering  -> ARPU + KMeans segments
    -> train                -> SMOTE -> XGBoost -> save model.pkl
    -> API loads model at startup
    -> POST /predict arrives -> features -> model.predict_proba()
    -> PostgreSQL           -> save prediction + segment
    -> Grafana              -> reads PostgreSQL -> live dashboard
```

## Key concepts (v1)

### XGBoost
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)
```
XGBoost handles mixed feature types, missing values, class imbalance (`scale_pos_weight`) and non-linear relationships. In the rigour pass, logistic regression scored higher on this dataset.

### SMOTE for class imbalance
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```
SMOTE generates synthetic minority samples. **Correction:** applying SMOTE to the whole training set before cross-validation is itself a leak. It must run inside each cross-validation fold (use an imblearn Pipeline), and never on validation or test data. In this project it also made no measurable difference.

### KMeans customer segmentation (v1)
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['monthly_charges', 'tenure', 'total_charges']])

kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)
```
Always scale before KMeans, because it uses Euclidean distance. Find K with the elbow method. The v2 segments use risk and value rules instead.

### ARPU calculation
```python
df['arpu'] = df['total_charges'] / df['tenure']
df['arpu'] = df['arpu'].fillna(df['monthly_charges'])  # new customers
```
Combine churn probability and ARPU to prioritise interventions: high ARPU and high churn probability means call today; low ARPU and high churn probability means an automated email.

### Flask API pattern (v1; v2 uses FastAPI, see above)
```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model ONCE at startup, not per request
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'loaded'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    features = engineer_features(df)
    features_scaled = scaler.transform(features)
    proba = model.predict_proba(features_scaled)[0][1]
    decision = 'CHURN_RISK' if proba >= 0.5 else 'RETAIN'
    return jsonify({'churn_probability': round(float(proba), 4), 'decision': decision})
```

### PostgreSQL predictions storage
```python
import psycopg2
import os

def save_prediction(customer_id, churn_probability, decision, segment):
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        database=os.getenv('DB_NAME', 'churn'),
        user=os.getenv('DB_USER', 'churn'),
        password=os.getenv('DB_PASSWORD', 'churn')
    )
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions
        (customer_id, churn_probability, decision, segment, created_at)
        VALUES (%s, %s, %s, %s, NOW())
    """, (customer_id, churn_probability, decision, segment))
    conn.commit()
    cursor.close()
    conn.close()
```
Inside Docker, the database host must be the service name (`postgres`) or `host.docker.internal` (for a database on the Mac), never `localhost`. The credentials above are development defaults only. Use secrets or environment variables in any real deployment.

### Grafana dashboard
Four panels: Customer Segments (ARPU and risk, bar chart), Churn Probability Over Time (time series), Churn Rate % (gauge), Total Predictions Made (stat). The dashboard was built on v1 predictions.

## Docker patterns (v1 example)
```yaml
services:
  api:
    build: .
    ports:
      - "8002:5001"       # host:container
    depends_on:
      - postgres
    environment:
      - DB_HOST=postgres  # service name, not localhost
  postgres:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana
volumes:
  postgres_data:
  grafana_data:
```
The current repo maps `5001:5000` for the API. Use named volumes so data survives container restarts.

## CLI reference

### Docker commands
```
docker compose up -d            # start all services
docker compose ps               # check status
docker compose logs api --tail=20
docker compose restart api
docker compose down
docker compose up --build -d    # rebuild after code changes
```

### Database inspection
```
docker compose exec postgres psql -U churn -d churn
\dt
SELECT COUNT(*) FROM predictions;
SELECT segment, COUNT(*), AVG(churn_probability) FROM predictions GROUP BY segment;
\q
```

## Debugging reference

| Error | Fix |
|---|---|
| ModuleNotFoundError | Run as `python -m src.models.train`, not `python src/models/train.py` |
| Connection refused to PostgreSQL | Use the service name (or host.docker.internal) inside Docker, not localhost |
| Address already in use | `lsof -i :<port>`, then stop the process (on macOS, port 5000 is AirPlay) |
| Feature names mismatch | Make sure the pipeline was fitted on the same features the API sends |
| KMeans poor clusters | Check StandardScaler is applied before fitting |
| Grafana no data | Check the PostgreSQL data source URL in Grafana settings |

Debugging order: read the last line of the traceback first; check which container is failing (`docker compose ps`); read the logs (`docker compose logs api --tail=50`); test the database connection; check the port binding.

## AWS EC2 deployment (v1)
The v1 API ran on an EC2 instance behind Docker Compose, reached over SSH with a key pair (kept outside the repository, never committed). Server address and key path are intentionally not recorded here. The instance is currently stopped. A redeployment is planned as an infrastructure-as-code exercise (Terraform with ECS Fargate).

## Key learnings from Week 1 (corrected)
1. Training and serving code must match exactly. Keep preprocessing inside the saved pipeline and build engineered features in one place.
2. Resample inside cross-validation folds only, never before cross-validation, and never on validation or test data. Doing it before inflated the score from 0.82 to 0.93.
3. Docker uses service names, not localhost, to reach other containers (and host.docker.internal to reach the Mac).
4. KMeans needs StandardScaler before clustering.
5. Persistent Docker volumes prevent data loss on restart. Always declare named volumes.
6. Load the model once at startup, not per request.
7. Pin library versions so a saved model loads the same everywhere.
