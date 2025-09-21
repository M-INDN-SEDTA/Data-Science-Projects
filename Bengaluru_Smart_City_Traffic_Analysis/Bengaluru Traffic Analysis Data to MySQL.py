import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import os

# Credentials
load_dotenv(override=True)
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
host = "localhost"
port = 3306
database = "traffic_analysis_db"

# Load Dataset
df = pd.read_csv("./datasets/Banglore_traffic_to_numeric_Dataset.csv")

# Map Roadwork_Construction to 0 or 1 to yes or no
df['Roadwork and Construction Activity'] = df['Roadwork and Construction Activity'].map({0:'No', 1:'Yes'})

# Rename columns to match SQL-friendly names
df.rename(columns={
    'Area Name': 'Area_Name',
    'Road/Intersection Name': 'Road_Intersection_Name',
    'Weather Conditions': 'Weather_Conditions',
    'Roadwork and Construction Activity': 'Roadwork_Construction_Activity'
}, inplace=True)

# Connect to MySQL 
conn = mysql.connector.connect(
    host=host,
    user=username,
    password=password,
    database=database,
    port=port
)
cursor = conn.cursor()

# Area table
cursor.execute("""
CREATE TABLE IF NOT EXISTS area_names (
    id INT PRIMARY KEY,
    text VARCHAR(50)
)
""")
area_values = [
    (0, 'Electronic City'), 
    (1, 'Hebbal'), 
    (2, 'Indiranagar'), 
    (3, 'Jayanagar'),
    (4, 'Koramangala'), 
    (5, 'M.G. Road'), 
    (6, 'Whitefield'), 
    (7, 'Yeshwanthpur')
]
cursor.executemany("INSERT IGNORE INTO area_names VALUES (%s,%s)", area_values)

print("values inserted in to Area table")

# Road names
cursor.execute("""
CREATE TABLE IF NOT EXISTS road_names (
    id INT PRIMARY KEY,
    text VARCHAR(50)
)
""")
road_values = [
    (0, '100 Feet Road'), 
    (1, 'Anil Kumble Circle'),
    (2, 'Ballari Road'), 
    (3, 'CMH Road'),
    (4, 'Hebbal Flyover'), 
    (5, 'Hosur Road'), 
    (6, 'ITPL Main Road'),
    (7, 'Jayanagar 4th Block'),
    (8, 'Marathahalli Bridge'),
    (9, 'Sarjapur Road'), 
    (10, 'Silk Board Junction'),
    (11, 'Sony World Junction'),
    (12, 'South End Circle'), 
    (13, 'Trinity Circle'), 
    (14, 'Tumkur Road'), 
    (15, 'Yeshwanthpur Circle')
]
cursor.executemany("INSERT IGNORE INTO road_names VALUES (%s,%s)", road_values)

print("values inserted in to Road table")
# Weather conditions
cursor.execute("""
CREATE TABLE IF NOT EXISTS weather_condition (
    id INT PRIMARY KEY,
    text VARCHAR(20)
)
""")
weather_values = [(0,'Clear'), (1,'Fog'), (2,'Overcast'), (3,'Rain'), (4,'Windy')]
cursor.executemany("INSERT IGNORE INTO weather_condition VALUES (%s,%s)", weather_values)

print("values inserted in to weather table")

# Create table traffic
cursor.execute("""
CREATE TABLE IF NOT EXISTS bengaluru_traffic (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Date DATE,
    Area_Name INT,
    Road_Intersection_Name INT,
    Traffic_Volume INT,
    Average_Speed DOUBLE,
    Travel_Time_Index DOUBLE,
    Congestion_Level DOUBLE,
    Road_Capacity_Utilization DOUBLE,
    Incident_Reports INT,
    Environmental_Impact DOUBLE,
    Public_Transport_Usage DOUBLE,
    Traffic_Signal_Compliance DOUBLE,
    Parking_Usage DOUBLE,
    Pedestrian_and_Cyclist_Count INT,
    Weather_Conditions INT,
    Roadwork_Construction_Activity VARCHAR(3),
    FOREIGN KEY (Area_Name) REFERENCES area_names(id),
    FOREIGN KEY (Road_Intersection_Name) REFERENCES road_names(id),
    FOREIGN KEY (Weather_Conditions) REFERENCES weather_condition(id)
)
""")

# Insert all 16 columns
sql_numeric = """
INSERT INTO bengaluru_traffic (
    Date, Area_Name, Road_Intersection_Name, Traffic_Volume,
    Average_Speed, Travel_Time_Index, Congestion_Level,
    Road_Capacity_Utilization, Incident_Reports, Environmental_Impact,
    Public_Transport_Usage, Traffic_Signal_Compliance, Parking_Usage,
    Pedestrian_and_Cyclist_Count, Weather_Conditions, Roadwork_Construction_Activity
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# Convert DataFrame to list of tuples
values_numeric = [tuple(x) for x in df.to_numpy()]

cursor.executemany(sql_numeric, values_numeric)
conn.commit()
print(f"{cursor.rowcount} records inserted")

# Close connection 
cursor.close()
conn.close()
print("Database connection closed.")
