import json
import os
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_VERSION = "v2.0"

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability with a leakage-free logistic regression pipeline. Built by Martin James — MLOps Engineer.",
    version="2.0.0",
)

pipeline = joblib.load(BASE_DIR / "churn_pipeline.pkl")
with open(BASE_DIR / "threshold.json") as f:
    THRESHOLD = json.load(f)["threshold"]

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    customer_id: str
    gender: Literal["Male", "Female"]
    senior_citizen: int = Field(ge=0, le=1)
    partner: YesNo
    dependents: YesNo
    tenure: int = Field(ge=0, le=72)
    phone_service: YesNo
    multiple_lines: Literal["No phone service", "No", "Yes"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: YesNoNoInternet
    online_backup: YesNoNoInternet
    device_protection: YesNoNoInternet
    tech_support: YesNoNoInternet
    streaming_tv: YesNoNoInternet
    streaming_movies: YesNoNoInternet
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Credit card (automatic)",
        "Bank transfer (automatic)",
    ]
    monthly_charges: float
    total_charges: Optional[float] = None


def tenure_group(tenure: int) -> str:
    if tenure <= 12:
        return "New"
    if tenure <= 24:
        return "Mid"
    return "Loyal"


def build_features(c: Customer) -> pd.DataFrame:
    total = c.total_charges or 0.0
    return pd.DataFrame([{
        "gender": c.gender,
        "SeniorCitizen": c.senior_citizen,
        "Partner": c.partner,
        "Dependents": c.dependents,
        "tenure": c.tenure,
        "PhoneService": c.phone_service,
        "MultipleLines": c.multiple_lines,
        "InternetService": c.internet_service,
        "OnlineSecurity": c.online_security,
        "OnlineBackup": c.online_backup,
        "DeviceProtection": c.device_protection,
        "TechSupport": c.tech_support,
        "StreamingTV": c.streaming_tv,
        "StreamingMovies": c.streaming_movies,
        "Contract": c.contract,
        "PaperlessBilling": c.paperless_billing,
        "PaymentMethod": c.payment_method,
        "MonthlyCharges": c.monthly_charges,
        "TotalCharges": total,
        "tenure_group": tenure_group(c.tenure),
        "charges_per_month": total / (c.tenure + 1),
        "is_high_value": int(c.monthly_charges > 70),
    }])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "churn_pipeline.pkl",
        "version": "2.0.0",
        "threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(customer: Customer):
    features = build_features(customer)
    churn_probability = float(pipeline.predict_proba(features)[0][1])
    churn_predicted = bool(churn_probability >= THRESHOLD)

    result = {
        "customer_id": customer.customer_id,
        "churn_probability": round(churn_probability, 4),
        "churn_predicted": churn_predicted,
        "threshold_used": round(THRESHOLD, 4),
        "risk_level": (
            "HIGH" if churn_probability >= 0.5 else
            "MEDIUM" if churn_probability >= THRESHOLD else
            "LOW"
        ),
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
                password="",
            )
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO churn_predictions
                (customer_id, churn_probability, churn_predicted, model_version)
            VALUES (%s, %s, %s, %s)
            """,
            (
                result["customer_id"],
                result["churn_probability"],
                result["churn_predicted"],
                MODEL_VERSION,
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"Prediction saved for {result['customer_id']}")
    except Exception as e:
        print(f"DB save failed (non-fatal): {e}")

    return result
