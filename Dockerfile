# ============================================================
# Dockerfile — Churn Prediction API
# ============================================================

# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first
COPY src/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY src/ ./src

# Expose port
EXPOSE 5000

# Run the app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5000"]