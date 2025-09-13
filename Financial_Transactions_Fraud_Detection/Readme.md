### 1. **Project Title & Description**

* **Name:** `Financial_Transaction_Fraud_Detection`
* **Type:** Data Science / Machine Learning Project
* **Description:**

  > This project focuses on analyzing, detecting, and predicting fraudulent financial transactions. Using Python for data cleaning, exploratory analysis, and modeling, it implements machine learning algorithms to identify fraud patterns. FastAPI serves a lightweight API for real-time predictions, while interactive dashboards (Power BI) and reports (`.pdf` / `.pptx`) visualize insights, trends, and key findings for decision-making.

* **Key Highlights:**

  * Integrates **data preprocessing, feature engineering, and EDA** for robust analysis.
  * Implements **Random Forest Classifier** to detect fraudulent transactions with high accuracy.
  * Provides **API endpoints** for real-time fraud detection via FastAPI.
  * Interactive **dashboards and visualizations** for dynamic exploration of transaction trends.
  * Summarizes analysis and insights in **PDF and presentation formats** for stakeholders.

---

### 2. Project Structure (Tree View)

```markdown

📂 Financial_Transaction_Fraud_Detection
├── datasets/                               # Raw and processed datasets
│   ├── financial_fraud_dataset.csv         # Dataset for analysis
│   └── fraud_transactions_for_sql.csv      # CSV for SQL import
│
├── fraud_transactions_detection_app/       # FastAPI web app
│   ├── templates/                          # Frontend HTML (Jinja2)
│   │   └── index.html
│   ├── __pycache__/                        # Python cache
│   ├── fraud_model.pkl                     # Local trained model
│   ├── main.py                             # FastAPI app (run with uvicorn)
│   └── requirements.txt                    # App dependencies
│
├── models/                                 # Trained ML models
│   └── fraud_model.pkl
│
├── Financial Fraud Detection.pbix          # Power BI dashboard
├── Financial Fraud Detection.pdf           # Project report (PDF)
├── Financial Fraud Detection.pptx          # Presentation slides
│
├── Financial Fraud Transaction Query.sql   # SQL queries
├── fraud_analysis_and_model.ipynb          # Jupyter Notebook (EDA + ML)
├── fraud_analysis_and_model.py             # Python script version
├── load_fraud_csv_to_sql.py                # Script to load CSV into SQL
└── requirements.txt                        # Root dependencies
```


---

### 3. **Objective / Problem / Goal**

Financial fraud in online transactions is often rare but highly damaging.
The goal of this project is to **detect fraudulent transactions** using ML models, explore fraud patterns with **EDA + visualization**, and deploy a **FastAPI service** for real-time fraud prediction.

---

### 4. **Data Source**

Dataset: [Kaggle – Financial Transactions Fraud Dataset](https://www.kaggle.com/datasets/ziya07/financial-transaction-for-fraud-detection-research/data?select=financial_fraud_dataset.csv)
Contains transaction details (amount, timestamp, merchant, device type, customer demographics, etc.) along with fraud labels.

---

### 5. **Data Cleaning & Preprocessing**

* Converted `timestamp` to datetime (extracted hour, day, month, year, weekend).
* Encoded categorical features (merchant category, device type, customer location).
* Dropped missing values & duplicates, ensured numeric scaling for features.

---

### 6. **Exploratory Data Analysis (EDA)**

* Python libraries (`pandas`, `matplotlib`, `seaborn`) used for charts.
* **Power BI dashboard** created for interactive visual insights.
* Key findings:

  * Fraud peaks in Electronics (\~33% of fraud losses).
  * Highest fraud by **hour** = 13:00 (1 PM).
  * Fraud more common on **Fridays**; lowest on weekends.
  * Tablet users showed slightly higher fraud rates than mobile/desktop.
  * Age group **52–68 years** showed most fraud cases.

---

### 7. **Modeling**

* ML algorithms tried:

  * **Logistic Regression** (baseline)
  * **Decision Trees**
  * **Random Forest Classifier** (best performance)
* Final Model: **Random Forest (200 trees, max depth 10, balanced class weights)**
* Saved at: `./models/fraud_model.pkl`
* Achieved strong performance with balanced fraud detection and generalization.

---

### 8. FastAPI App (Fraud Detection API)

* **Run the app**

  ```bash
  cd fraud_transactions_detection_app
  pip install -r requirements.txt
  uvicorn main:app --reload
  ```

* **Access in browser**

  * Homepage (frontend form): [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
  * API docs (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

* **Test with Postman**

  * Endpoint: `POST http://127.0.0.1:8000/predict_fraud/`
  * Example request body (JSON):

    ```json
    {
      "amount": 2500.75,
      "customer_age": 34,
      "merchant_category": 2,
      "customer_location": 1,
      "device_type": 0,
      "previous_transactions": 5,
      "hour": 14,
      "month": 9,
      "year": 2025,
      "day_of_week": 2,
      "is_weekend": false
    }
    ```
  * Example response:

    ```json
    {
      "is_fraud": 1,
      "fraud_probability": 0.876
    }
    ```


* Run FastAPI,
* Use **browser UI**,
* Use **Swagger**, or
* Test directly in **Postman** with JSON payloads.

---

9. **Installation**

   * Clone repo.
    ```bash
    git clone https://github.com/M-INDN-SEDTA/Data-Science-Projects/tree/main/Financial_Transactions_Fraud_Detection
    ```

   * Install dependencies:

     ```bash
     pip install -r requirements.txt
     ```

---

### 10. **Visualization**

All insights and analysis are presented using multiple formats for clarity and reporting:

* **Interactive Dashboards:** `Financial Fraud Detection.pbix` (Power BI) for exploring trends, patterns, and fraud distribution dynamically.
* **Static Reports:** `.pdf` provides a complete analysis summary of transactions, fraud trends, and key insights.
* **Presentation:** `.pptx` highlights the full analysis, including charts, trends, and summary points for easy sharing and discussion.
* **Python Plots:** Generated using `matplotlib` and `seaborn` for exploratory data analysis (monthly/yearly trends, fraud by merchant, customer age distribution, heatmaps, KDE plots, etc.).

---

11. **Future Scope**

    * Deploy API on cloud.
    * Use real-time transaction streaming.
    * Improve model with deep learning.

