# ============================================================
# load_data.py
# Loads Telco CSV into churn_raw table in churn_db
# ============================================================
import pandas as pd
import psycopg2
# Load CSV
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# Clean before loading
df.columns = df.columns.str.lower()
df = df.rename(columns={'customerid': 'customer_id'})
df['totalcharges']=pd.to_numeric(df['totalcharges'],errors= 'coerce')
df= df.fillna(0)

# Connect to database
conn = psycopg2.connect(
    dbname="churn_db",
    user="martinjames",
    host="localhost"
)
cur = conn.cursor()

# Insert rows
for _, row in df.iterrows():
    cur.execute(""" INSERT INTO churn_raw (
            customer_id, gender, senior_citizen,
            partner, dependents, tenure,
            phone_service, multiple_lines,
            internet_service, online_security,
            online_backup, device_protection,
            tech_support, streaming_tv,
            streaming_movies, contract,
            paperless_billing, payment_method,
            monthly_charges, total_charges, churn)
            VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s
)""",
(
        row['customer_id'], row['gender'],
        row['seniorcitizen'], row['partner'],
        row['dependents'], row['tenure'],
        row['phoneservice'], row['multiplelines'],
        row['internetservice'], row['onlinesecurity'],
        row['onlinebackup'], row['deviceprotection'],
        row['techsupport'], row['streamingtv'],
        row['streamingmovies'], row['contract'],
        row['paperlessbilling'], row['paymentmethod'],
        row['monthlycharges'], row['totalcharges'],
        row['churn']
    ))

# Commit and close
conn.commit()
cur.close()
conn.close()
print(f" ✅ Loaded {len(df)} rows into churn")