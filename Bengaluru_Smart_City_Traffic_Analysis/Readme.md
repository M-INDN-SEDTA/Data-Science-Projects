# Bengaluru Smart City Traffic Analysis

### 1. **Project Title & Description**

* **Name:** `Bengaluru Smart City Traffic Analysis`

* **Type:** Data Science / Machine Learning Project

* **Description:**

  > This project focuses on analyzing traffic patterns in Bengaluru, predicting traffic volume, congestion, and average speed using machine learning. The project implements feature engineering, exploratory data analysis, and predictive modeling. A Flask-based web application provides a user interface for real-time predictions. Dashboards, reports (`.pdf` / `.pptx`), and SQL queries help visualize trends and support decision-making for city traffic management.

* **Key Highlights:**

  * Integrates **data preprocessing, feature engineering, and EDA** for traffic insights.
  * Implements multiple ML models:

    * Traffic Volume (Independent)
    * Congestion Level (Independent)
    * Average Speed (Independent)
    * Traffic Volume with Features
  * Provides a **Flask web app** for real-time predictions.
  * Interactive dashboards and reports for **stakeholder insights**.
  * SQL scripts for integrating traffic datasets into databases.

---

### 2. Project Structure (Tree View)

```markdown
📂 Bengaluru_Smart_City_Traffic_Analysis
├── datasets/                              # Raw and processed datasets
│   ├── Banglore_feature_engg_numeric_traffic_dataset.csv
│   ├── Banglore_traffic_Dataset.csv
│   └── Banglore_traffic_to_numeric_Dataset.csv
│
├── models/                                 # Trained ML models 
│   ├── model1_traffic_volume_independent.pkl
│   ├── model2_congestion_level_independent.pkl
│   ├── model3_average_speed_independent.pkl
│   └── model4_traffic_volume_with_features.pkl
│
├── traffic_prediction_app/                 # Flask web app
│   ├── templates/                          # Frontend HTML (Jinja2)
│   │   └── index.html
│   ├── app.py                              # Flask app entry point
│   └── requirements.txt                    # App dependencies
│
├── Bengaluru_Smart_City_Traffic_Analysis.ipynb    # Jupyter Notebook (EDA + ML)
├── Bengaluru_Smart_City_Traffic_Analysis.py       # Python script version
├── Bengaluru_Smart_City_Traffic_Analysis.pbix     # Power BI dashboard
├── Bengaluru_Smart_City_Traffic_Analysis.pdf      # PDF Power BI dashboard
├── Bengaluru_Smart_City_Traffic_Analysis.pptx     # Complete report
├── Bengaluru_Traffic_Analysis_SQL_Queries_.sql    # SQL scripts
├── Bengaluru_traffic_data_to_mysql.py             # CSV → MySQL loader
└── requirements.txt                               # Root dependencies
```

---

### 3. **Objective / Problem / Goal**

Bengaluru suffers from heavy traffic congestion, accidents, and slow mobility.
The goal of this project is to **predict traffic patterns** (volume, congestion, speed), explore trends using **EDA**, and provide a **Flask web app** for real-time traffic predictions. This supports **better urban planning, traffic management, and decision-making**.

---

### 4. **Data Source**

* Bengaluru traffic datasets in CSV format (`datasets/`) including:

  * Traffic volume
  * Road/intersection features
  * Weather conditions
  * Roadworks/construction activity
  * Temporal features (year, month, day, day of week)

---

### 5. **Data Cleaning & Preprocessing**

* Converted categorical features (`Area Name`, `Road/Intersection Name`, `Weather Conditions`) to numeric values.
* Handled missing values and duplicates.
* Feature scaling for numeric columns like `Travel Time Index`, `Road Capacity Utilization`.
* Generated additional temporal features for modeling (DayOfWeek, Month, Year).

---

### 6. **Exploratory Data Analysis (EDA)**

* Python libraries (`pandas`, `matplotlib`, `seaborn`) used for analysis.
* Key insights:

  * Traffic peaks during morning (8–10 AM) and evening (5–8 PM) rush hours.
  * Congestion higher in industrial and IT zones.
  * Roadworks and accidents significantly impact average speed.
  * Weather conditions (rain, fog) correlate with slower traffic and congestion.

---

### 7. **Modeling**

* ML algorithms tested for traffic prediction:

  * **Linear Regression** – baseline regression model.
  * **Random Forest Regressor** – best performance across models, robust to non-linear relationships.
  * **Gradient Boosting Regressor** – used for comparison, handles complex feature interactions.

* **Models saved in `models/` folder:**

| Model File                                | Purpose / Prediction                          | Features Used                                                                                                                                                                                                                                                                                                                                |
| ----------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model1_traffic_volume_independent.pkl`   | Predict **Traffic Volume** independently      | `Area Name`, `Road/Intersection Name`, `Travel Time Index`, `Road Capacity Utilization`, `Incident Reports`, `Environmental Impact`, `Public Transport Usage`, `Traffic Signal Compliance`, `Parking Usage`, `Pedestrian and Cyclist Count`, `Weather Conditions`, `Roadwork and Construction Activity`, `Year`, `Month`, `Day`, `DayOfWeek` |
| `model2_congestion_level_independent.pkl` | Predict **Congestion Level** independently    | Same as Model 1                                                                                                                                                                                                                                                                                                                              |
| `model3_average_speed_independent.pkl`    | Predict **Average Speed** independently       | Same as Model 1                                                                                                                                                                                                                                                                                                                              |
| `model4_traffic_volume_with_features.pkl` | Predict **Traffic Volume** using all features | All features from Model 1 + `Congestion Level`, `Average Speed`                                                                                                                                                                                                                                                                              |


* **Target Variables / Predictions:**

  * Model 1 → Traffic Volume (vehicles/hour)
  * Model 2 → Congestion Level (percentage, 0–100%)
  * Model 3 → Average Speed (km/h)
  * Model 4 → Traffic Volume with enhanced features (vehicles/hour, factoring congestion and speed)

* **Training & Evaluation:**

  * Dataset split into **train (80%) / test (20%)**.
  * Metrics calculated: **RMSE**, **R²**, **MAE** for regression evaluation.
  * **Random Forest Regressor** selected as final model for all predictions due to superior performance on test data.

* **Notes on Feature Engineering:**

  * Categorical features (`Area Name`, `Road/Intersection Name`, `Weather Conditions`, `Roadwork and Construction Activity`) encoded numerically.
  * Temporal features (`Year`, `Month`, `Day`, `DayOfWeek`) used to capture patterns across time.
  * Additional derived features for Model 4:

    * `Congestion Level` (input for traffic volume prediction)
    * `Average Speed` (input for traffic volume prediction)


---

### 8. Flask App (Traffic Prediction Dashboard)

* **Run the app**

```bash
cd traffic_prediction_app
pip install -r requirements.txt
python app.py
```

* **Access in browser**

  * Homepage: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
* **Features**

  * Form for real-time traffic prediction
  * Predict traffic volume, congestion, average speed
  * Clear inputs and pre-loaded example datasets
  * Results shown with units (e.g., vehicles/hour, km/h)

---

### 9. **Installation**

* Clone the repository:

```bash
git clone https://github.com/your-username/Bengaluru_Smart_City_Traffic_Analysis.git
```

* Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 10. **Visualization**

* **Power BI Dashboard:** `Bengaluru_Smart_City_Traffic_Analysis.pbix` for interactive traffic trends.
* **PDF Report:** `Bengaluru_Smart_City_Traffic_Analysis.pdf` summarizing analysis and insights.
* **Presentation Slides:** `Bengaluru_Smart_City_Traffic_Analysis.pptx` for stakeholders.
* **Python plots:** generated using `matplotlib` & `seaborn` (heatmaps, congestion distribution, traffic trends by time/day/area).

---

### 11. **Future Scope**

* Deploy Flask app to **cloud or internal city server**.
* Integrate **real-time traffic sensors** for live predictions.
* Incorporate **deep learning models** for more accurate predictions.
* Build **mobile app or GIS dashboard** for public traffic updates.

---
