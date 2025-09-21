-- Total and Average Traffic per Area
SELECT 
    a.text AS Area_Name,
    SUM(t.Traffic_Volume) AS Total_Traffic,
    AVG(t.Average_Speed) AS AvgSpeed_per_Area
FROM bengaluru_traffic t
JOIN area_names a ON t.Area_Name = a.id
GROUP BY a.text
ORDER BY Total_Traffic DESC;

-- Average Traffic per Weather Condition
SELECT 
    w.text AS Weather_Condition,
    AVG(t.Traffic_Volume) AS Avg_Traffic_per_Weather
FROM bengaluru_traffic t
JOIN weather_condition w ON t.Weather_Conditions = w.id
GROUP BY w.text
ORDER BY Avg_Traffic_per_Weather DESC;

-- Average Traffic per Month
SELECT 
    MONTH(t.Date) AS Month,
    AVG(t.Traffic_Volume) AS Avg_Traffic_per_Month
FROM bengaluru_traffic t
GROUP BY Month
ORDER BY Month;

-- Average Traffic per Day of Week
SELECT 
    DAYNAME(t.Date) AS Day_Name,
    AVG(t.Traffic_Volume) AS Avg_Traffic_per_Day
FROM bengaluru_traffic t
GROUP BY Day_Name
ORDER BY Avg_Traffic_per_Day DESC;

-- Week of Year & Quarter for each record
SELECT 
    t.Date,
    WEEK(t.Date) AS WeekOfYear,
    QUARTER(t.Date) AS Quarter
FROM bengaluru_traffic t;

-- Traffic per Road Capacity & Congestion Impact
SELECT 
    t.Traffic_Volume / NULLIF(t.Road_Capacity_Utilization,0) AS Traffic_per_Capacity,
    t.Traffic_Volume * (t.Congestion_Level/100) AS Congestion_Impact
FROM bengaluru_traffic t;

-- Incident Rate per Traffic Volume
SELECT 
    t.Incident_Reports / NULLIF(t.Traffic_Volume,0) AS Incident_Rate
FROM bengaluru_traffic t;

-- Pedestrian, Public Transport & Parking Ratios
SELECT
    t.Pedestrian_and_Cyclist_Count / NULLIF(t.Traffic_Volume,0) AS Pedestrian_Ratio,
    t.Traffic_Volume / NULLIF(t.Public_Transport_Usage,0) AS PublicTransport_Effect,
    t.Parking_Usage / NULLIF(t.Traffic_Volume,0) AS Parking_Load
FROM bengaluru_traffic t;

-- Environmental Impact per Traffic Volume
SELECT
    t.Environmental_Impact / NULLIF(t.Traffic_Volume,0) AS Env_Impact_per_Traffic
FROM bengaluru_traffic t;

-- Traffic Variability per Area
SELECT 
    a.text AS Area_Name,
    STDDEV(t.Traffic_Volume) AS Traffic_std_per_Area,
    STDDEV(t.Traffic_Volume) / NULLIF(SUM(t.Traffic_Volume),0) AS Traffic_cv_per_Area
FROM bengaluru_traffic t
JOIN area_names a ON t.Area_Name = a.id
GROUP BY a.text;

-- Top 5 Areas by Total Traffic
SELECT 
    a.text AS Area_Name,
    SUM(t.Traffic_Volume) AS Total_Traffic
FROM bengaluru_traffic t
JOIN area_names a ON t.Area_Name = a.id
GROUP BY a.text
ORDER BY Total_Traffic DESC
LIMIT 5;

-- Top 5 Days by Average Traffic
SELECT 
    DAYNAME(t.Date) AS Day_Name,
    AVG(t.Traffic_Volume) AS Avg_Traffic
FROM bengaluru_traffic t
GROUP BY Day_Name
ORDER BY Avg_Traffic DESC
LIMIT 5;

-- Top 5 Months by Average Traffic
SELECT 
    MONTH(t.Date) AS Month,
    AVG(t.Traffic_Volume) AS Avg_Traffic
FROM bengaluru_traffic t
GROUP BY Month
ORDER BY Avg_Traffic DESC
LIMIT 5;

-- Traffic vs Incident Reports Correlation
SELECT
    (SUM(t.Traffic_Volume*t.Incident_Reports) - SUM(t.Traffic_Volume)*SUM(t.Incident_Reports)/COUNT(*)) /
    (SQRT(SUM(POW(t.Traffic_Volume,2)) - POW(SUM(t.Traffic_Volume),2)/COUNT(*)) *
     SQRT(SUM(POW(t.Incident_Reports,2)) - POW(SUM(t.Incident_Reports),2)/COUNT(*))) AS corr_Traffic_Incident
FROM bengaluru_traffic t;

-- Traffic vs Congestion Level Correlation
SELECT
    (SUM(t.Traffic_Volume*t.Congestion_Level) - SUM(t.Traffic_Volume)*SUM(t.Congestion_Level)/COUNT(*)) /
    (SQRT(SUM(POW(t.Traffic_Volume,2)) - POW(SUM(t.Traffic_Volume),2)/COUNT(*)) *
     SQRT(SUM(POW(t.Congestion_Level,2)) - POW(SUM(t.Congestion_Level),2)/COUNT(*))) AS corr_Traffic_Congestion
FROM bengaluru_traffic t;

-- Average Speed vs Congestion Level Correlation
SELECT
    (SUM(t.Average_Speed*t.Congestion_Level) - SUM(t.Average_Speed)*SUM(t.Congestion_Level)/COUNT(*)) /
    (SQRT(SUM(POW(t.Average_Speed,2)) - POW(SUM(t.Average_Speed),2)/COUNT(*)) *
     SQRT(SUM(POW(t.Congestion_Level,2)) - POW(SUM(t.Congestion_Level),2)/COUNT(*))) AS corr_Speed_Congestion
FROM bengaluru_traffic t;
