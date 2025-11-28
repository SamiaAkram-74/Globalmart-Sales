import streamlit as st
import requests

# ----------------------------
# App title
# ----------------------------
st.title("ML Model Prediction App")
st.write("Enter the features of the order to get the prediction from the ML model API.")

# ----------------------------
# User inputs
# ----------------------------
Order_ID = st.number_input("Order ID", min_value=1, step=1)
Order_Date = st.date_input("Order Date")
Units_Sold = st.number_input("Units Sold", min_value=0, step=1)
Revenue = st.number_input("Revenue", min_value=0.0, step=0.01, format="%.2f")
Deal_Size = st.selectbox("Deal Size", ["Small", "Medium", "Large"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
tenure = st.number_input("Tenure (months)", min_value=0, step=1)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.01, format="%.2f")

# ----------------------------
# Prediction button
# ----------------------------
if st.button("Predict"):

    # Convert date to string
    Order_Date_str = Order_Date.strftime("%Y-%m-%d")

    # Prepare JSON payload
    payload = {
        "Order_ID": int(Order_ID),
        "Order_Date": Order_Date_str,
        "Units_Sold": int(Units_Sold),
        "Revenue": float(Revenue),
        "Deal_Size": Deal_Size,
        "SeniorCitizen": int(SeniorCitizen),
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges)
    }

    try:
        # Send POST request to FastAPI
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            prediction = response.json()["prediction"][0]
            st.success(f"Predicted Target: {prediction:.2f}")
        else:
            st.error(f"API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
