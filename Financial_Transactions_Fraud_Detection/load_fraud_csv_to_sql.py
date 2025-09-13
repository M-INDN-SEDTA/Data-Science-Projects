import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import os

# 1. Load .env credentials
load_dotenv(override=True)

username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
host = "localhost"
port = 3306
database = "financial_transactions_fraud_db"

# 2. Load CSV file
sql_df = pd.read_csv("./datasets/fraud_transactions_for_sql")
print("CSV loaded with shape:", sql_df.shape)

# 3. Connect to MySQL
conn = mysql.connector.connect(
    host=host,
    user=username,
    password=password,
    database=database,
    port=port
)
cursor = conn.cursor()

# 4. Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    timestamp DATETIME,
    amount DOUBLE,
    customer_id VARCHAR(50),
    customer_age INT,
    previous_transactions INT,
    hour INT,
    is_weekend BOOLEAN,
    merchant_category VARCHAR(50),
    customer_location VARCHAR(50),
    day_of_week VARCHAR(50),
    device_type VARCHAR(50),
    is_fraud INT
)
""")

# 5. Prepare insert statement
sql = """
INSERT IGNORE INTO transactions (
    transaction_id, timestamp, amount, customer_id, customer_age,
    previous_transactions, hour, is_weekend, merchant_category,
    customer_location, day_of_week, device_type, is_fraud
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# 6. Convert DataFrame rows to list of tuples
values = [tuple(x) for x in sql_df.to_numpy()]

# 7. Insert all rows
cursor.executemany(sql, values)

# 8. Commit changes
conn.commit()
print(cursor.rowcount, "records inserted (duplicates ignored).")

# 9. Close cursor and connection
cursor.close()
conn.close()
