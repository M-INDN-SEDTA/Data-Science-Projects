-- 1. Percentage of Fraud vs Non-Fraud
SELECT 
    is_fraud,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 2) AS percent
FROM transactions
GROUP BY is_fraud;

                  --

-- 2. Monthly Fraud Counts (Aggregated Across Years)
SELECT 
    DATE_FORMAT(timestamp, '%Y-%m') AS month,
    COUNT(*) AS fraud_count
FROM transactions
WHERE is_fraud = 1
GROUP BY month
ORDER BY month;

                  --

-- 3. Yearly Fraud Counts
SELECT 
    YEAR(timestamp) AS year,
    COUNT(*) AS fraud_count
FROM transactions
WHERE is_fraud = 1
GROUP BY year
ORDER BY year;

                  --

-- 4. Fraud Count by Date
SELECT 
    DATE(timestamp) AS date_only,
    COUNT(*) AS fraud_count
FROM transactions
WHERE is_fraud = 1
GROUP BY date_only
ORDER BY date_only;

                  --

-- 5. Fraud vs Merchant Category
SELECT 
    merchant_category,
    COUNT(*) AS fraud_count
FROM transactions
WHERE is_fraud = 1
GROUP BY merchant_category
ORDER BY fraud_count DESC;

                  --

-- 6. Fraud vs Customer Age
SELECT 
    customer_age,
    COUNT(*) AS fraud_count
FROM transactions
WHERE is_fraud = 1
GROUP BY customer_age
ORDER BY customer_age;

                  --

-- 7. Fraud vs Customer Location
SELECT 
    customer_location,
    COUNT(*) AS fraud_count
FROM transactions
WHERE is_fraud = 1
GROUP BY customer_location
ORDER BY fraud_count DESC;

                  --

-- 8. Fraud vs Device Type
SELECT 
    device_type,
    COUNT(*) AS fraud_count
FROM transactions
WHERE is_fraud = 1
GROUP BY device_type
ORDER BY fraud_count DESC;

                  --

-- 9. Fraud vs Previous Transactions
SELECT 
    previous_transactions,
    COUNT(*) AS fraud_count
FROM transactions
WHERE is_fraud = 1
GROUP BY previous_transactions
ORDER BY previous_transactions;

                  --

-- 10. Fraud vs Hour of Transaction
SELECT 
    hour,
    COUNT(*) AS fraud_count
FROM transactions
WHERE is_fraud = 1
GROUP BY hour
ORDER BY hour;

                  --

-- 11. Fraud vs Day of Week
SELECT 
    day_of_week,
    COUNT(*) AS fraud_count
FROM transactions
WHERE is_fraud = 1
GROUP BY day_of_week
ORDER BY FIELD(day_of_week, 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday');

                  --

-- 12. Fraud vs Weekend/Weekday
SELECT 
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    COUNT(*) AS fraud_count
FROM transactions
WHERE is_fraud = 1
GROUP BY day_type;

                  --

-- 13. Top 5 Non-Fraudulent Transactions by Amount
SELECT *
FROM transactions
WHERE is_fraud = 0
ORDER BY amount DESC
LIMIT 5;

                  --

-- 14. Fraud Heatmap: Hour vs Day of Week
SELECT 
    hour,
    day_of_week AS day_of_week,
    COUNT(*) AS fraud_count
FROM transactions
WHERE is_fraud = 1
GROUP BY hour, day_of_week
ORDER BY hour, FIELD(day_of_week, 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday');
