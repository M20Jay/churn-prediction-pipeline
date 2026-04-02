# ============================================================
# app.py
# Flask API for Churn Prediction
# ============================================================
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime 
from sklearn.preprocessing import LabelEncoder

# Initialise Flask app
app = Flask(__name__)

#load model
model = joblib.load('src/churn_model.pkl')
print(" Model successfully loaded")

# ============================================================
# Route 1 — Health Check
# ============================================================
@app.route('/health', methods = ['GET'])
def health ():
    return jsonify ({
        "status" :"ok",
        "model" : "churn_model.pkl",
        "message": "churn prediction API running"

    }), 200

# ============================================================
# Route 2 — Predict Churn
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    # Rename columns to match training data
    df = df.rename(columns={
        'SeniorCitizen': 'senior_citizen',
        'Partner': 'partner',
        'Dependents': 'dependents',
        'PhoneService': 'phone_service',
        'MultipleLines': 'multiple_lines',
        'InternetService': 'internet_service',
        'OnlineSecurity': 'online_security',
        'OnlineBackup': 'online_backup',
        'DeviceProtection': 'device_protection',
        'TechSupport': 'tech_support',
        'StreamingTV': 'streaming_tv',
        'StreamingMovies': 'streaming_movies',
        'Contract': 'contract',
        'PaperlessBilling': 'paperless_billing',
        'PaymentMethod': 'payment_method',
        'MonthlyCharges': 'monthly_charges',
        'TotalCharges': 'total_charges'
    })

    # Engineer the 3 missing features
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 72],
        labels=[0, 1, 2])
    
    df['tenure_group'] = df['tenure_group'].fillna(0).astype(int)
    
    df['charges_per_month'] = df['total_charges'] / (df['tenure'] + 1)
    
    df['is_high_value'] = (df['monthly_charges'] > 70).astype(int)

    # Drop customerID
    df = df.drop(columns=['customerID'], errors='ignore')

    # Encode categorical columns
    le = LabelEncoder()
    cat_cols = [
        'gender', 'partner', 'dependents',
        'phone_service', 'multiple_lines',
        'internet_service', 'online_security',
        'online_backup', 'device_protection',
        'tech_support', 'streaming_tv',
        'streaming_movies', 'contract',
        'paperless_billing', 'payment_method',
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Reorder columns to match training order
    expected_cols = [
        'gender', 'senior_citizen', 'partner', 'dependents', 'tenure',
        'phone_service', 'multiple_lines', 'internet_service',
        'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies',
        'contract', 'paperless_billing', 'payment_method',
        'monthly_charges', 'total_charges', 
        'tenure_group', 'charges_per_month', 'is_high_value'
    ]
    df = df[expected_cols]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    result = {
        "customer_id": data.get("customerID", "unknown"),
        "churn_predicted": bool(prediction),
        "churn_probability": round(float(probability), 4)
    }
    # Save prediction to database
    try:
        conn = psycopg2.connect(
            dbname="churn_db",
            user="martinjames",
            host="localhost"
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
        print(f"DB save failed: {e}")

    return jsonify(result), 200

# ============================================================
# Run the app
# ============================================================
if __name__ =='__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
