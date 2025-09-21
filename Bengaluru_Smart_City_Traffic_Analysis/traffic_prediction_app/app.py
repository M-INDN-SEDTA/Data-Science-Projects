# app.py
import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Root folder of project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Model paths
MODEL_PATHS = {
    "model1": os.path.join(BASE_DIR, "models", "model1_traffic_volume_independent.pkl"),
    "model2": os.path.join(BASE_DIR, "models", "model2_congestion_level_independent.pkl"),
    "model3": os.path.join(BASE_DIR, "models", "model3_average_speed_independent.pkl"),
    "model4": os.path.join(BASE_DIR, "models", "model4_traffic_volume_with_features.pkl")
}

# Load models
model1 = joblib.load(MODEL_PATHS["model1"])
model2 = joblib.load(MODEL_PATHS["model2"])
model3 = joblib.load(MODEL_PATHS["model3"])
model4 = joblib.load(MODEL_PATHS["model4"])

#  Feature Columns 
FEATURES_BASE = [
    "Area Name", "Road/Intersection Name", "Travel Time Index",
    "Road Capacity Utilization", "Incident Reports", "Environmental Impact",
    "Public Transport Usage", "Traffic Signal Compliance", "Parking Usage",
    "Pedestrian and Cyclist Count", "Weather Conditions",
    "Roadwork and Construction Activity", "Year", "Month", "Day", "DayOfWeek"
]

# Model 1, 2, 3 use base features
FEATURES_M123 = FEATURES_BASE

# Model 4 additionally needs Congestion Level + Average Speed
FEATURES_M4 = FEATURES_BASE + ["Congestion Level", "Average Speed"]

#  Utility: Predict Function 
def make_prediction(model, form_data, feature_list):
    values = [float(form_data.get(f, 0)) for f in feature_list]
    arr = np.array(values).reshape(1, -1)
    return round(float(model.predict(arr)[0]), 2)

#  Routes 
@app.route("/")
def index():
    # Use base features for form rendering
    return render_template("index.html", features=FEATURES_BASE)

@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    form_data = request.form
    if model_name == "model1":
        pred = make_prediction(model1, form_data, FEATURES_M123)
        label = "Predicted Traffic Volume (Independent)"
    elif model_name == "model2":
        pred = make_prediction(model2, form_data, FEATURES_M123)
        label = "Predicted Congestion Level (Independent)"
    elif model_name == "model3":
        pred = make_prediction(model3, form_data, FEATURES_M123)
        label = "Predicted Average Speed (Independent)"
    elif model_name == "model4":
        pred = make_prediction(model4, form_data, FEATURES_M4)
        label = "Predicted Traffic Volume (With Features)"
    else:
        return jsonify({"error": "Invalid model"}), 400

    return jsonify({"prediction": pred, "label": label})

if __name__ == "__main__":
    app.run(debug=True)
