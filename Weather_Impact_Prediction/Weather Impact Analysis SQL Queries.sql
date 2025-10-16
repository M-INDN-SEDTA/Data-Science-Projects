-- 1. Average values per hour
SELECT 
    hour,
    AVG(temperature) AS avg_temp,
    AVG(humidity) AS avg_humidity,
    AVG(wind_speed) AS avg_wind_speed,
    AVG(pressure) AS avg_pressure
FROM weather_data
GROUP BY hour
ORDER BY hour;

-- 2. Average values per day
SELECT 
    date,
    AVG(temperature) AS avg_temp,
    AVG(humidity) AS avg_humidity,
    AVG(wind_speed) AS avg_wind_speed,
    AVG(pressure) AS avg_pressure
FROM weather_data
GROUP BY date
ORDER BY date;

-- 3. Average values per month
SELECT 
    month,
    AVG(temperature) AS avg_temp,
    AVG(humidity) AS avg_humidity,
    AVG(wind_speed) AS avg_wind_speed,
    AVG(pressure) AS avg_pressure
FROM weather_data
GROUP BY month
ORDER BY month;

-- 4. Average values per quarter
SELECT 
    QUARTER(formatted_date) AS quarter,
    AVG(temperature) AS avg_temp,
    AVG(humidity) AS avg_humidity,
    AVG(wind_speed) AS avg_wind_speed
FROM weather_data
GROUP BY quarter;

-- 5. Temperature vs Apparent Temperature ratio & difference
SELECT 
    formatted_date,
    temperature,
    apparent_temperature,
    temperature / NULLIF(apparent_temperature, 0) AS temp_apparent_ratio,
    temperature - apparent_temperature AS temp_diff
FROM weather_data
LIMIT 100;

-- 6. Wind and humidity interaction
SELECT 
    formatted_date,
    wind_speed * humidity AS wind_humidity_impact
FROM weather_data
LIMIT 100;

-- 7. Statistics for numeric columns
SELECT 
    MIN(temperature) AS min_temp,
    MAX(temperature) AS max_temp,
    AVG(temperature) AS avg_temp,
    STDDEV(temperature) AS std_temp,
    MIN(humidity) AS min_humidity,
    MAX(humidity) AS max_humidity,
    AVG(humidity) AS avg_humidity,
    STDDEV(humidity) AS std_humidity
FROM weather_data;

-- 8. Hourly standard deviation & coefficient of variation
SELECT 
    hour,
    STDDEV(temperature) AS temp_std,
    AVG(temperature) AS temp_mean,
    STDDEV(temperature)/AVG(temperature) AS temp_cv,
    STDDEV(humidity) AS humidity_std,
    AVG(humidity) AS humidity_mean,
    STDDEV(humidity)/AVG(humidity) AS humidity_cv
FROM weather_data
GROUP BY hour
ORDER BY hour;

-- 9. Count of unique values for precipitation
SELECT precip_type, COUNT(*) AS count
FROM weather_data
GROUP BY precip_type;

-- 10. Count of unique values for wind category
SELECT wind_category, COUNT(*) AS count
FROM weather_data
GROUP BY wind_category;

-- 11. Top correlated feature (Temperature vs Humidity)
SELECT 
    (SUM((temperature - temp_avg)*(humidity - hum_avg)) / (SQRT(SUM(POW(temperature - temp_avg, 2)))*SQRT(SUM(POW(humidity - hum_avg, 2)))) ) AS temp_humidity_corr
FROM (
    SELECT temperature, humidity,
           AVG(temperature) OVER () AS temp_avg,
           AVG(humidity) OVER () AS hum_avg
    FROM weather_data
) AS sub;

-- 12. Export numeric dataset to CSV (optional, adjust path)
SELECT *
INTO OUTFILE '/tmp/weather_numeric_features.csv'
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n'
FROM weather_data;
