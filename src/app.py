import os
import joblib
import pandas as pd
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel

# ── App init ──────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability using XGBoost. Built by Martin James — MLOps Engineer.",
    version="1.0.0"
)

# ── Load model once at startup ────────────────────────────────
model = joblib.load("src/churn_model.pkl")

# ── Pydantic input schema — all 20 features ───────────────────
class Customer(BaseModel):
    customer_id: str
    gender: int
    senior_citizen: int
    partner: int
    dependents: int
    tenure: float
    phone_service: int
    multiple_lines: int
    internet_service: int
    online_security: int
    online_backup: int
    device_protection: int
    tech_support: int
    streaming_tv: int
    streaming_movies: int
    contract: int
    paperless_billing: int
    payment_method: int
    monthly_charges: float
    total_charges: float
    tenure_group: int
    charges_per_month: float
    is_high_value: int

# ── Health check ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "churn_model.pkl", "version": "1.0.0"}

# ── Predict endpoint ──────────────────────────────────────────
@app.post("/predict")
def predict(customer: Customer):

    features = pd.DataFrame([{
        "gender"           : customer.gender,
        "senior_citizen"   : customer.senior_citizen,
        "partner"          : customer.partner,
        "dependents"       : customer.dependents,
        "tenure"           : customer.tenure,
        "phone_service"    : customer.phone_service,
        "multiple_lines"   : customer.multiple_lines,
        "internet_service" : customer.internet_service,
        "online_security"  : customer.online_security,
        "online_backup"    : customer.online_backup,
        "device_protection": customer.device_protection,
        "tech_support"     : customer.tech_support,
        "streaming_tv"     : customer.streaming_tv,
        "streaming_movies" : customer.streaming_movies,
        "contract"         : customer.contract,
        "paperless_billing": customer.paperless_billing,
        "payment_method"   : customer.payment_method,
        "monthly_charges"  : customer.monthly_charges,
        "total_charges"    : customer.total_charges,
        "tenure_group"     : customer.tenure_group,
        "charges_per_month": customer.charges_per_month,
        "is_high_value"    : customer.is_high_value,
    }])

    churn_probability = float(model.predict_proba(features)[0][1])
    churn_predicted   = bool(churn_probability >= 0.5)

    result = {
        "customer_id"      : customer.customer_id,
        "churn_probability": round(churn_probability, 4),
        "churn_predicted"  : churn_predicted,
        "risk_level"       : (
            "HIGH"   if churn_probability > 0.7 else
            "MEDIUM" if churn_probability > 0.4 else
            "LOW"
        )
    }

    database_url = os.environ.get("DATABASE_URL")
    try:
        if database_url:
            conn = psycopg2.connect(database_url)
        else:
            conn = psycopg2.connect(
                host="host.docker.internal",
                database="churn_db",
                user="postgres",
                password=""
            )
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO churn_predictions
                (customer_id, churn_probability, churn_predicted, model_version)
            VALUES (%s, %s, %s, %s)
        """, (
            result["customer_id"],
            result["churn_probability"],
            result["churn_predicted"],
            "v1.0"
        ))
        conn.commit()
        cur.close()
        conn.close()
        print(f"Prediction saved for {result['customer_id']}")

    except Exception as e:
        print(f"DB save failed (non-fatal): {e}")

    return result