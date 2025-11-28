# import joblib

# bundle = joblib.load("best_model.pkl")

# print("\n=== FEATURE NAMES INSIDE MODEL ===")
# for f in bundle["features"]:
#     print(f)






from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# ----------------------------
# Pydantic schema for input
# ----------------------------
class PredictRequest(BaseModel):
    Order_ID: int
    Order_Date: str
    Units_Sold: int
    Revenue: float
    Deal_Size: str
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float

# ----------------------------
# Load model bundle
# ----------------------------
bundle = joblib.load("best_model.pkl")
model = bundle["model"]
feature_names = bundle["features"]

# ----------------------------
# Initialize FastAPI
# ----------------------------
app = FastAPI(title="ML Model API")

@app.get("/")
def root():
    return {"message": "ML Model API is running"}

@app.post("/predict")
def predict(data: PredictRequest):
    # Convert input JSON to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Convert object columns (like dates) to numeric ordinal
    for col in input_df.columns:
        if input_df[col].dtype == object:
            try:
                input_df[col] = pd.to_datetime(input_df[col], errors="coerce").map(pd.Timestamp.toordinal)
            except:
                input_df = input_df.drop(columns=[col])

    # Align features with training features
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Predict
    prediction = model.predict(input_df)
    return {"prediction": prediction.tolist()}
