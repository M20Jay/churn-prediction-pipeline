# Week 1 — Customer Churn Prediction Pipeline
**Martin James Ng'ang'a · MLOps Engineer · Nairobi, Kenya 🇰🇪**
`github.com/M20Jay` · Week 1 of 15

---

## Overview

End-to-end MLOps pipeline that predicts customer churn for a telecom company using real customer data. Scores customers in real time, stores predictions in PostgreSQL, segments customers by value and risk, and monitors everything through a live Grafana dashboard.

**This is Week 1 — the foundation. Every pattern learned here repeats across all 15 weeks.**

---

## Final Results

| Metric | Result |
|--------|--------|
| Dataset | 7,043 telecom customers |
| Model | XGBoost classifier |
| Segments | 4 KMeans clusters |
| Live API | http://18.184.3.203:8002/docs |
| Endpoints | GET /health · POST /predict |
| Grafana panels | 4 panels · segments, churn probability, churn rate, total predictions |

---

## Customer Segments

| Segment | Customers | Avg ARPU | Priority |
|---------|-----------|----------|----------|
| High Value At Risk | 1,693 | $98.4 | ACT NOW |
| High Value Loyal | 1,851 | $73.8 | Maintain |
| Low Value At Risk | 1,524 | $47.7 | Monitor |
| Low Value Loyal | 1,975 | $20.1 | Standard |

**ARPU** = Average Revenue Per User. High ARPU + High churn risk = call them today.

---

## 7-Day Build Plan

| Day | Task | Status |
|-----|------|--------|
| Day 1 | EDA + Feature Engineering | ✅ |
| Day 2 | XGBoost model training | ✅ |
| Day 3 | Flask API — /health + /predict | ✅ |
| Day 4 | Docker container | ✅ |
| Day 5 | Grafana dashboard | ✅ |
| Day 6 | ARPU analysis + KMeans segmentation | ✅ |
| Day 7 | README + Deploy to AWS EC2 | ✅ |

---

## Project Structure

```
churn-prediction-pipeline/
├── data/                       Raw and processed data
├── src/
│   ├── data/
│   │   └── preprocessing.py    Load, clean, encode features
│   ├── features/
│   │   └── feature_engineering.py  ARPU, tenure groups, KMeans
│   ├── models/
│   │   ├── train.py            XGBoost training + SMOTE
│   │   └── evaluate.py         Metrics, confusion matrix
│   └── api/
│       ├── app.py              Flask app — /health + /predict
│       └── schemas.py          Request/response validation
├── grafana/
│   └── dashboards/             Grafana JSON dashboard configs
├── sql/
│   └── init.sql                PostgreSQL table creation
├── notebooks/                  EDA — exploratory analysis
├── reports/                    Model evaluation reports
├── Dockerfile
├── docker-compose.yml          API + PostgreSQL + Grafana
└── requirements.txt
```

---

## Pipeline Architecture

```
Raw telecom data (CSV)
    ↓
src/data/preprocessing.py       → cleaned df
    ↓
src/features/feature_engineering.py  → ARPU + KMeans segments
    ↓
src/models/train.py             → SMOTE → XGBoost → save model.pkl
    ↓
src/api/app.py                  → Flask API loads model at startup
    ↓
POST /predict request arrives
    ↓
Feature engineering → model.predict_proba()
    ↓
PostgreSQL                      → save prediction + segment
    ↓
Grafana                         → reads PostgreSQL → live dashboard
```

---

## Key Concepts

### Why XGBoost for Churn

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)
```

XGBoost handles:
- Mixed feature types (categorical + numerical)
- Missing values natively
- Class imbalance with `scale_pos_weight`
- Non-linear relationships between features and churn

---

### SMOTE for Class Imbalance

```python
from imblearn.over_sampling import SMOTE

# Churn datasets are always imbalanced — churners are minority
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# SMOTE generates synthetic minority samples
# Apply ONLY on training set — never on test set
```

---

### KMeans Customer Segmentation

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# CRITICAL: Always scale before KMeans
# KMeans uses Euclidean distance — unscaled features distort clusters
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['monthly_charges', 'tenure', 'total_charges']])

kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Find optimal K — elbow method
inertias = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
# Plot inertias — find the elbow (point where curve flattens)
```

---

### ARPU Calculation

```python
# Average Revenue Per User — key business metric
# Segments customers by financial value to the business

df['arpu'] = df['total_charges'] / df['tenure']
df['arpu'] = df['arpu'].fillna(df['monthly_charges'])  # new customers

# Combine churn probability + ARPU to prioritise interventions:
# High ARPU + High churn probability → call today
# Low ARPU + High churn probability → automated email
```

---

### Flask API Pattern

```python
# src/api/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model ONCE at startup — not per request
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'loaded'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        # Feature engineering
        features = engineer_features(df)
        features_scaled = scaler.transform(features)
        proba = model.predict_proba(features_scaled)[0][1]
        decision = 'CHURN_RISK' if proba >= 0.5 else 'RETAIN'
        return jsonify({
            'churn_probability': round(float(proba), 4),
            'decision': decision
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

### PostgreSQL Predictions Storage

```python
# src/data/database.py
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

**Critical:** Inside Docker, `DB_HOST` must be the service name (`postgres`) not `localhost`.

---

### Grafana Dashboard Setup

```yaml
# docker-compose.yml
services:
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
```

**4 panels:**
1. Customer Segments — ARPU & Risk (bar chart)
2. Churn Probability Over Time (time series)
3. Churn Rate % (gauge)
4. Total Predictions Made (stat panel)

---

## Docker Patterns

### docker-compose.yml Key Points

```yaml
services:
  api:
    build: .
    ports:
      - "8002:5001"       # host:container
    depends_on:
      - postgres
    environment:
      - DB_HOST=postgres  # service name not localhost
      - DB_NAME=churn
      - DB_USER=churn
      - DB_PASSWORD=churn

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=churn
      - POSTGRES_USER=churn
      - POSTGRES_PASSWORD=churn
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

---

## CLI Reference

### Training Pipeline

```bash
# Run from project root
PYTHONPATH=. python src/data/preprocessing.py
PYTHONPATH=. python src/features/feature_engineering.py
PYTHONPATH=. python src/models/train.py
PYTHONPATH=. python src/models/evaluate.py

# Check saved model
ls -lh models/
```

### API Testing

```bash
# Health check
curl -s http://localhost:8002/health | python3 -m json.tool

# Churn prediction
curl -s -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "monthly_charges": 65.5,
    "total_charges": 786.0,
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "tech_support": "No",
    "online_security": "No"
  }' | python3 -m json.tool
```

### Docker Commands

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View API logs
docker compose logs api --tail=20

# Restart API only
docker compose restart api

# Stop everything
docker compose down

# Rebuild after code changes
docker compose up --build -d

# Check port on server
sudo ss -tlnp | grep 8002
```

### Database Inspection

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U churn -d churn

# Inside psql:
\dt                                    -- list tables
SELECT COUNT(*) FROM predictions;
SELECT decision, COUNT(*) FROM predictions GROUP BY decision;
SELECT segment, COUNT(*), AVG(churn_probability)
  FROM predictions GROUP BY segment;
\q                                     -- quit
```

### Git Workflow

```bash
git status
git add src/models/train.py
git commit -m "feat: add SMOTE balancing and XGBoost training"
git push origin main
git log --oneline -5
```

---

## Debugging Reference

### Common Errors and Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run as `python -m src.models.train` not `python src/models/train.py` |
| `Connection refused PostgreSQL` | Check `DB_HOST=postgres` not `localhost` inside Docker |
| `Address already in use port 8002` | `lsof -i :8002` then `kill -9 <PID>` |
| `Feature names mismatch` | Ensure scaler fitted on same features as model |
| `KMeans poor clusters` | Check StandardScaler applied before fitting |
| `Grafana no data` | Check PostgreSQL data source URL in Grafana settings |

### Debugging Order

```
1. Read the LAST line of the traceback — that is the actual error
2. Check which container is failing: docker compose ps
3. Check logs: docker compose logs api --tail=50
4. Check database connection: docker compose exec postgres psql -U churn -d churn
5. Check port binding: sudo ss -tlnp | grep 8002
```

---

## AWS EC2 Deployment

```bash
# SSH to server
ssh -i ~/Documents/GitHub/mlops-key.pem ubuntu@18.184.3.203

# Start churn API
cd ~/churn-prediction-pipeline
docker compose up -d churn-api

# Verify running
docker ps | grep churn
curl -s http://localhost:8002/health

# Check logs
docker compose logs churn-api --tail=20
```

---

## Key Learnings from Week 1

- **Training and serving code must match exactly** — if you engineer features during training, you must engineer the exact same features during prediction
- **Docker uses service names not localhost** — `DB_HOST=postgres` not `DB_HOST=localhost` inside a Docker network
- **KMeans requires StandardScaler** — always scale before clustering, never after
- **Persistent Docker volumes** — prevent data loss on container restart — always declare named volumes
- **Load model once at startup** — not per request — loading per request is too slow in production
- **SMOTE only on training set** — never apply oversampling to test set — this is data leakage

---

## Interview Q&A

**Q: What is customer churn and why does it matter?**
A: Churn is when a customer stops using a service. Acquiring a new customer costs 5-7x more than retaining an existing one. A churn prediction model allows the business to intervene — targeted offers, personal calls — before the customer leaves. The KMeans segmentation means interventions are prioritised: High Value At Risk customers ($98.4 ARPU) get immediate personal attention.

**Q: Why XGBoost and not Logistic Regression?**
A: Telecom churn data has non-linear relationships between features. Contract type, internet service, and payment method interact in complex ways that logistic regression cannot capture. XGBoost handles feature interactions, missing values, and class imbalance natively.

**Q: What is SMOTE and when should you use it?**
A: Synthetic Minority Oversampling Technique — generates synthetic samples of the minority class (churners) to balance the training dataset. Use when churners are less than 20-30% of your dataset. Critical rule: apply SMOTE only to the training set, never to the test set — applying to test set is data leakage and gives falsely optimistic metrics.

**Q: How does Grafana connect to your data?**
A: Grafana reads directly from PostgreSQL via a configured data source. Every prediction saved by the Flask API is written to PostgreSQL. Grafana queries that table and renders the panels. The dashboard refreshes automatically — when a new prediction arrives, Grafana shows it within seconds.

---

*Week 1 of 15 · Churn Prediction Pipeline · Built in Nairobi, Kenya 🇰🇪*
*Live API: http://18.184.3.203:8002/docs · Repository: https://github.com/M20Jay/churn-prediction-pipeline*

---

## Deep Dives — Critical Concepts

### Why StandardScaler Before KMeans — Not After

KMeans uses Euclidean distance. Without scaling, total_charges (range $18-$8,684) dominates over tenure (range 1-72). The cluster boundaries are drawn by total_charges alone — not actual behaviour.

With StandardScaler (mean=0, std=1) all features contribute equally to distance calculations.

Rule: fit scaler on training data only — transform new data with same fitted scaler.

### Data Leakage — The Most Dangerous Mistake in ML

Data leakage = test set information influences training = falsely optimistic metrics.

```python
# WRONG — SMOTE before split
X_res, y_res = smote.fit_resample(X, y)      # sees test data
X_train, X_test = train_test_split(X_res, y_res)

# CORRECT — split first, SMOTE on train only
X_train, X_test = train_test_split(X, y)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# WRONG — scaler fitted on full dataset
scaler.fit(X)

# CORRECT — scaler fitted on train only
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Class Imbalance — Why Accuracy is Misleading

73% stay, 27% churn. Predicting "stay" for everyone: Accuracy 73%, Recall on churners 0%. Use Precision, Recall, F1 instead. Business context drives threshold — retention call costs $5, lost customer costs $500/year — lower threshold catches more churners even with false positives.

### Docker Networking — Why localhost Fails Inside Containers

Inside a Docker container, localhost refers to the container itself — not the host machine. PostgreSQL runs in a separate container. Docker Compose creates a shared network — services reach each other by service name. DB_HOST=postgres resolves to the postgres container automatically.
