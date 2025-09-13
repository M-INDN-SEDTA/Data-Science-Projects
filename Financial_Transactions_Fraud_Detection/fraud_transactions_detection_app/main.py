from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model

#============If want to load model direcly from models folder -----
# Load model from models folder (one level up from main.py)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fraud_model.pkl")
# model = joblib.load(MODEL_PATH)

#====================== OR ========================

# copy in current directory and use this below line -------
model = joblib.load("fraud_model.pkl")

# FastAPI app
app = FastAPI(title="Fraud Detection API")

templates = Jinja2Templates(directory="templates")

# Required features
required_features = [
    "amount","customer_age","merchant_category","customer_location",
    "device_type","previous_transactions","hour","month","year",
    "day_of_week","is_weekend"
]

# Input schema
class Transaction(BaseModel):
    amount: float
    customer_age: int
    merchant_category: int
    customer_location: int
    device_type: int
    previous_transactions: int
    hour: int
    month: int
    year: int
    day_of_week: int
    is_weekend: bool

# Serve frontend
@app.get("/", response_class=HTMLResponse)
def get_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoint
@app.post("/predict_fraud/")
def predict_fraud(transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])
    
    # Check missing features
    missing = [f for f in required_features if f not in data.columns]
    if missing:
        return JSONResponse({"error": "Missing features", "missing_features": missing})
    
    prediction = model.predict(data[required_features])[0]
    probability = model.predict_proba(data[required_features])[0][1]
    
    return JSONResponse({
        "is_fraud": int(prediction),
        "fraud_probability": round(probability, 3)
    })
