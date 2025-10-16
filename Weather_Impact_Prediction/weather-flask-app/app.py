from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv
import joblib
import pandas as pd
from datetime import datetime, timezone

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("OPENWEATHER_API_KEY")
if not api_key:
    raise Exception("OPENWEATHER_API_KEY missing in .env")

model = joblib.load("./models/weather_temp_model.pkl")
print("Model loaded successfully.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reverse-geocode", methods=["POST"])
def reverse_geocode():
    data = request.json
    lat = data.get("lat")
    lon = data.get("lon")
    if lat is None or lon is None:
        return jsonify({"error": "Missing coordinates"}), 400

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "json",
        "lat": lat,
        "lon": lon,
        "zoom": 10,
        "addressdetails": 1
    }
    headers = {"User-Agent": "WeatherMapApp/1.0 (example@example.com)"}
    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        return jsonify({"error": "Reverse geocoding failed", "details": response.text}), 500

    data = response.json()
    address = data.get("address", {})
    city = address.get("city") or address.get("town") or address.get("village") or address.get("municipality") or address.get("county") or "Unknown"

    return jsonify({
        "latitude": lat,
        "longitude": lon,
        "city": city,
        "state": address.get("state", "Unknown"),
        "country": address.get("country", "Unknown")
    })


@app.route("/api/weather", methods=["POST"])
def get_weather():
    data = request.json
    lat = data.get("lat")
    lon = data.get("lon")
    if lat is None or lon is None:
        return jsonify({"error": "Missing coordinates"}), 400

    CURRENT_URL = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(CURRENT_URL)
    if response.status_code != 200:
        return jsonify({"error": "Weather API failed", "details": response.text}), 500

    data = response.json()
    if "main" not in data:
        return jsonify({"error": "Invalid weather data"}), 500

    actual_temp = data["main"]["temp"]
    apparent_temp = data["main"]["feels_like"]
    humidity = data["main"]["humidity"] / 100.0
    wind_speed_kmh = round(data["wind"]["speed"] * 3.6, 2)
    wind_bearing = data["wind"].get("deg", 0)
    visibility_km = round(data.get("visibility", 0) / 1000, 2)
    pressure = data["main"]["pressure"]
    hour = datetime.fromtimestamp(data["dt"], tz=timezone.utc).hour
    temp_diff = round(data["main"]["temp_max"] - data["main"]["temp_min"], 2)

    if wind_speed_kmh < 1:
        wind_category = 0
    elif wind_speed_kmh < 20:
        wind_category = 1
    elif wind_speed_kmh < 40:
        wind_category = 2
    else:
        wind_category = 3

    precip_type = 1 if "rain" in data else 0

    features = pd.DataFrame([{
        "Apparent Temperature (C)": apparent_temp,
        "Humidity": humidity,
        "Wind Speed (km/h)": wind_speed_kmh,
        "Wind Bearing (degrees)": wind_bearing,
        "Visibility (km)": visibility_km,
        "Pressure (millibars)": pressure,
        "Hour": hour,
        "Temp_Diff": temp_diff,
        "Wind_Category": wind_category,
        "Precip Type": precip_type
    }])

    predicted_temp = model.predict(features)[0]

    return jsonify({
        "actual_temp": actual_temp,
        "predicted_temp": predicted_temp,
        "apparent_temp": apparent_temp,
        "humidity": humidity * 100,
        "wind_speed": wind_speed_kmh,
        "wind_bearing": wind_bearing,
        "visibility": visibility_km,
        "pressure": pressure,
        "hour": hour,
        "temp_diff": temp_diff,
        "wind_category": wind_category,
        "precip_type": precip_type
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
