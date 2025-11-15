import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

# =========================
# Step 1: Title
# =========================
st.title("Supermarket Revenue Prediction Dashboard")

# =========================
# Step 2: Load the trained model
# =========================
model = load("lr_demand_model.pkl")  # Load your saved linear regression model

# =========================
# Step 3: Load CSV data
# =========================
df = pd.read_csv("supermarket_sales.csv")

# =========================
# Step 4: Feature Engineering
# =========================
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Year'] = df['Order_Date'].dt.year
df['Month'] = df['Order_Date'].dt.month
df['DayOfWeek'] = df['Order_Date'].dt.dayofweek
df['WeekOfYear'] = df['Order_Date'].dt.isocalendar().week
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# =========================
# Step 5: Streamlit Filters
# =========================
st.sidebar.header("Filters")

# Date range filter
start_date = st.sidebar.date_input("Start Date", df['Order_Date'].min())
end_date = st.sidebar.date_input("End Date", df['Order_Date'].max())

# Product Category filter
product_categories = st.sidebar.multiselect(
    "Product Category",
    options=df['Product_Category'].unique(),
    default=df['Product_Category'].unique()
)

# Product Name filter
product_names = st.sidebar.multiselect(
    "Product Name",
    options=df['Product_Name'].unique(),
    default=df['Product_Name'].unique()
)

# Apply filters
filtered_df = df[
    (df['Order_Date'] >= pd.to_datetime(start_date)) &
    (df['Order_Date'] <= pd.to_datetime(end_date)) &
    (df['Product_Category'].isin(product_categories)) &
    (df['Product_Name'].isin(product_names))
]

# =========================
# Step 6: Features and Target
# =========================
features = ['Units_Sold', 'Target', 'Deal_Size', 'Month', 'DayOfWeek', 'WeekOfYear', 'IsWeekend']
target = 'Revenue'

X = filtered_df[features]
y = filtered_df[target]

# =========================
# Step 7: Make Predictions
# =========================
predictions = model.predict(X)
filtered_df["Predicted_Revenue"] = predictions

# =========================
# Step 8: Show Model Performance
# =========================
mae = mean_absolute_error(y, predictions)
rmse = rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

st.subheader("Model Performance")
st.write(f"**MAE:** {mae:.2f}  |  **RMSE:** {rmse:.2f}  |  **R² Score:** {r2:.2f}")

# =========================
# Step 9: Display DataFrame
# =========================
st.subheader("Filtered Data with Predictions")
st.dataframe(filtered_df)

# =========================
# Step 10: Interactive Plots
# =========================
st.subheader("Revenue Trend")
fig = px.line(
    filtered_df.groupby('Order_Date').agg({'Revenue':'sum', 'Predicted_Revenue':'sum'}).reset_index(),
    x='Order_Date',
    y=['Revenue', 'Predicted_Revenue'],
    labels={'value':'Revenue', 'Order_Date':'Date'},
    title="Actual vs Predicted Revenue Over Time"
)
st.plotly_chart(fig, use_container_width=True)

# Top features (coefficients)
st.subheader("Top Contributing Features")
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)
st.bar_chart(coef_df.set_index("Feature"))

# =========================
# Step 11: Download Filtered Data
# =========================
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Filtered Predictions as CSV",
    data=csv,
    file_name='filtered_predicted_revenue.csv',
    mime='text/csv'
)
