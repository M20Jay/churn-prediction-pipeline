cat > ~/Documents/GitHub/churn-prediction-pipeline/README.md << 'EOF'
# Churn Prediction Pipeline
**Martin James — MLOps Engineer | GitHub: M20Jay | Nairobi, Kenya**

---

## Overview
An end-to-end MLOps pipeline that predicts customer churn for a telecom company using real customer data. The system scores customers in real time, stores predictions in PostgreSQL, segments customers by value and risk, and monitors everything through a live Grafana dashboard.

---

## Tech Stack
| Tool | Purpose |
|------|---------|
| XGBoost | Churn prediction model |
| Flask | REST API serving predictions |
| Docker | Containerised deployment |
| PostgreSQL | Storing predictions and segments |
| Grafana | Real time monitoring dashboard |
| KMeans | Customer segmentation |
| pandas / sklearn | Data processing |

---

## Week 1 — Day by Day
| Day | Task | Status |
|-----|------|--------|
| Day 1 | EDA + Feature Engineering | ✅ |
| Day 2 | XGBoost model training | ✅ |
| Day 3 | Flask API — /health + /predict | ✅ |
| Day 4 | Docker container | ✅ |
| Day 5 | Grafana dashboard | ✅ |
| Day 6 | ARPU analysis + KMeans segmentation | ✅ |
| Day 7 | README + Deploy to Render | ✅ |

---

## Customer Segmentation
4 segments identified using KMeans clustering:

| Segment | Customers | Avg ARPU | Priority |
|---------|-----------|----------|----------|
| High Value At Risk | 1693 | $98.4 | ACT NOW |
| High Value Loyal | 1851 | $73.8 | Maintain |
| Low Value At Risk | 1524 | $47.7 | Monitor |
| Low Value Loyal | 1975 | $20.1 | Standard |

---

## How to Run
Start everything: docker compose up -d
Test the API: curl http://localhost:5001/health
Stop everything: docker compose down

## Grafana Dashboard

4 panels monitoring the pipeline in real time:
- Customer Segments — ARPU & Risk
- Churn Probability Over Time
- Churn Rate %
- Total Predictions Made

![Grafana Dashboard](https://github.com/user-attachments/assets/8ec9795e-a484-4d15-9205-207bf37eaaac)

---

## Live API
https://churn-prediction-pipeline-1zue.onrender.com/docs
---

## Key Learnings
- Training and serving code must match exactly
- Docker needs host.docker.internal to reach host PostgreSQL
- KMeans requires StandardScaler before clustering
- Persistent Docker volumes prevent data loss on restart

---

## Next — Week 2
Credit Card Fraud Detection using Kafka and Redis for real time streaming under 200ms
