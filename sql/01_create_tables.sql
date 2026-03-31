-- ============================================================
-- Churn Prediction Pipeline — Table Definitions
-- Database: churn_db
-- ============================================================

-- 1. RAW DATA
CREATE TABLE IF NOT EXISTS churn_raw (
    id                     SERIAL PRIMARY KEY,
    customer_id            VARCHAR(50),
    gender                 VARCHAR(10),
    senior_citizen         INTEGER,
    partner                VARCHAR(5),
    dependents             VARCHAR(5),
    tenure                 INTEGER,
    phone_service          VARCHAR(5),
    multiple_lines         VARCHAR(20),
    internet_service       VARCHAR(20),
    online_security        VARCHAR(20),
    online_backup          VARCHAR(20),
    device_protection      VARCHAR(20),
    tech_support           VARCHAR(20),
    streaming_tv           VARCHAR(20),
    streaming_movies       VARCHAR(20),
    contract               VARCHAR(20),
    paperless_billing      VARCHAR(5),
    payment_method         VARCHAR(30),
    monthly_charges        NUMERIC(10, 2),
    total_charges          NUMERIC(10, 2),
    churn                  VARCHAR(5),
    loaded_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. PREDICTIONS
CREATE TABLE IF NOT EXISTS churn_predictions (
    id                     SERIAL PRIMARY KEY,
    customer_id            VARCHAR(50) NOT NULL,
    churn_probability      NUMERIC(6, 4) NOT NULL,
    churn_predicted        BOOLEAN NOT NULL,
    model_version          VARCHAR(20) DEFAULT 'v1.0',
    predicted_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. METRICS
CREATE TABLE IF NOT EXISTS churn_metrics (
    id                     SERIAL PRIMARY KEY,
    model_version          VARCHAR(20) NOT NULL,
    run_date               DATE NOT NULL DEFAULT CURRENT_DATE,
    accuracy               NUMERIC(6, 4),
    precision              NUMERIC(6, 4),
    recall                 NUMERIC(6, 4),
    f1_score               NUMERIC(6, 4),
    roc_auc                NUMERIC(6, 4),
    total_predictions      INTEGER,
    notes                  TEXT,
    logged_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);