# ============================================================
# Dockerfile — Churn Prediction API
# ============================================================

# Base image
FROM python:3.12-slim

WORKDIR /app

COPY src/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src

# ✅ ADD THIS LINE — copies the model into the image
COPY src/churn_model.pkl ./src/churn_model.pkl

EXPOSE 5001

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5000"]