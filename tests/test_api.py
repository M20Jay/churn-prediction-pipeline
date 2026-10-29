import os

# Point the API's database call at a port where nothing listens, so tests
# never touch a real database. The API treats a failed insert as non-fatal.
os.environ["DATABASE_URL"] = "postgresql://test:test@127.0.0.1:1/test"

from fastapi.testclient import TestClient  # noqa: E402

from src.app import app  # noqa: E402

client = TestClient(app)

BASE = {
    "customer_id": "TEST-001",
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 50.0,
    "total_charges": 600.0,
}


def customer(**changes):
    return {**BASE, **changes}


HIGH_RISK = customer(
    tenure=2,
    internet_service="Fiber optic",
    monthly_charges=95.0,
    total_charges=190.0,
)

LOW_RISK = customer(
    tenure=60,
    contract="Two year",
    monthly_charges=40.0,
    total_charges=2400.0,
    payment_method="Bank transfer (automatic)",
    online_security="Yes",
    tech_support="Yes",
    partner="Yes",
    dependents="Yes",
    paperless_billing="No",
)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_expected_fields():
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    expected = {"customer_id", "churn_probability", "churn_predicted",
                "threshold_used", "risk_level"}
    assert expected <= r.json().keys()


def test_high_risk_customer_scores_high():
    data = client.post("/predict", json=HIGH_RISK).json()
    assert data["churn_probability"] > 0.5
    assert data["risk_level"] == "HIGH"


def test_low_risk_customer_scores_low():
    data = client.post("/predict", json=LOW_RISK).json()
    assert data["churn_probability"] < 0.3
    assert data["churn_predicted"] is False


def test_invalid_category_is_rejected():
    r = client.post("/predict", json=customer(contract="Weekly"))
    assert r.status_code == 422


def test_tenure_out_of_range_is_rejected():
    r = client.post("/predict", json=customer(tenure=999))
    assert r.status_code == 422


def test_risk_ordering():
    high = client.post("/predict", json=HIGH_RISK).json()["churn_probability"]
    base = client.post("/predict", json=BASE).json()["churn_probability"]
    low = client.post("/predict", json=LOW_RISK).json()["churn_probability"]
    assert high > base > low