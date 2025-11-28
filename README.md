# **Sales Forecasting & ML Model Deployment Pipeline**

## **Project Overview**

This project demonstrates a complete end-to-end machine learning workflow, from data analysis to model deployment. The aim is to predict key metrics (like sales/revenue) using historical data, automate the pipeline, and expose the trained model via an API for real-time predictions.

It bridges the gap between experimentation and production by combining **data preprocessing**, **machine learning**, **MLOps**, and **interactive UI deployment**.

---

## **1. Data Exploration & Analysis**

**Steps Performed:**

1. **Load & Inspect Data**

   * Imported the dataset and explored its structure.
   * Checked for missing values, data types, duplicates, and outliers.

2. **Clean the Data**

   * Handled missing values using appropriate imputation.
   * Removed duplicates and corrected inconsistent formatting.

3. **Analyze & Visualize**

   * Identified top-selling products, key cities, and customer demographics.
   * Created visualizations using **Matplotlib** and **Seaborn** (histograms, box plots, heatmaps).

4. **Feature Engineering & Preprocessing**

   * Created time-based features (month, day, promo flags).
   * Encoded categorical variables and handled remaining missing values.

---

## **2. Machine Learning Model Development**

**Steps Performed:**

1. **Model Training**

   * Trained multiple models: **Linear Regression**, **Random Forest**, **XGBoost/LightGBM**.

2. **Model Evaluation**

   * Used **RMSE**, **MAE**, and **R²** metrics.
   * Performed time-based cross-validation for robust evaluation.

3. **Explainability**

   * Plotted **feature importance** for tree-based models.
   * Conducted **residual analysis** to assess model performance.

**Tools Used:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `seaborn`

**Outcome:** A validated machine learning model ready for production, with clear insights into influential features.

---

## **3. Automated Data Pipeline**

**Objective:** Automate data ingestion, preprocessing, retraining, and artifact versioning.

**Steps:**

1. Connected to simulated data sources (CSV folder).
2. Automated preprocessing and feature generation scripts.
3. Retrained models periodically and saved best models with versioning using `joblib`.
4. Scheduled automated runs using Python `schedule` library (or alternatives like `cron`/GitHub Actions).
5. Logged all runs and stored retraining artifacts systematically.

**Tools Used:** `pandas`, `sqlalchemy`, `joblib`, `schedule`

**Outcome:** A reproducible pipeline demonstrating reliable periodic model updates.

---

## **4. MLOps - Model Deployment & API Serving**

**Objective:** Expose the trained ML model as an API for real-time predictions.

**Steps:**

1. **Build Inference API**

   * Used **FastAPI** to create a `/predict` endpoint accepting JSON input features.
   * Validated input using **Pydantic schemas**.

2. **Containerization**

   * Wrote a **Dockerfile** to package Python, dependencies, model file, and API code into a lightweight image.

3. **Cloud Deployment**

   * Docker container can be deployed to cloud platforms such as **Hugging Face Spaces**, **Render**, or **AWS Free Tier**.

4. **Interactive Demo (Optional)**

   * Built a **Streamlit** interface consuming the API.
   * Users can adjust sliders/inputs and see predictions in real-time.

**Tools Used:** `FastAPI`, `Uvicorn`, `Docker`, `Streamlit`

**Outcome:** A live public API endpoint accessible for predictions and a user-friendly interactive UI.

---

## **5. How to Run**

### **Run API locally**

```bash
# Build Docker image
docker build -t ml-api .

# Run container
docker run -p 8000:8000 ml-api

# Test API in browser or Postman
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
```

### **Run Streamlit UI**

```bash
streamlit run streamlit_app.py
```

---

## **6. Project Highlights**

* End-to-end ML workflow: **Data Analysis → Preprocessing → Training → Deployment**
* Automated pipeline with **versioning** and scheduled runs
* Real-time API predictions with **FastAPI**
* Interactive Streamlit demo for non-technical users
* Dockerized deployment ensuring **reproducibility** across environments

---

## **7. Tools & Libraries**

* **Data Handling & ML:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`
* **Visualization:** `matplotlib`, `seaborn`
* **API & Deployment:** `FastAPI`, `Uvicorn`, `Docker`
* **UI Demo:** `Streamlit`
* **Automation:** `joblib`, `schedule`


