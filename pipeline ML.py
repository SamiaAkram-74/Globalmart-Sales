import os
import glob
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime

# --------------------------
# Config
# --------------------------
DATA_FOLDER = "./"          # Current folder
MODEL_PATH = "best_model.pkl"
LOG_FILE = "pipeline_log.txt"

# --------------------------
# Auto-clean old model file
# --------------------------
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    print("Old best_model.pkl deleted. Fresh training will create a new one.")

# --------------------------
# Load CSVs
# --------------------------
def load_data():
    csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    df_list = [pd.read_csv(file) for file in csv_files if file.endswith(".csv")]

    if not df_list:
        print("⚠ No CSV files found.")
        return None, []

    data = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(csv_files)} CSV files. Total rows: {len(data)}")
    return data, csv_files

# --------------------------
# Preprocessing
# --------------------------
def preprocess(df):
    df = df.dropna(how='all')
    df = df.ffill().bfill()

    target = df.columns[-1]  # assume last column is target
    print(f"Target column: {target}")

    if df[target].dtype == object:
        unique_vals = df[target].unique()
        if set(unique_vals) <= {"Yes", "No"}:
            df[target] = df[target].map({"Yes": 1, "No": 0})
            print(f"Converted target '{target}' from Yes/No to 1/0")
        else:
            print(f"Target '{target}' not numeric and not Yes/No. Skipping.")
            return None

    df[target] = pd.to_numeric(df[target], errors='coerce')
    df = df.dropna(subset=[target])

    if df.shape[0] == 0:
        print("After cleaning, dataframe is empty.")
        return None

    return df

# --------------------------
# Train Model
# --------------------------
def train_model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Convert non-numeric features
    X_numeric = X.copy()
    for col in X_numeric.columns:
        if X_numeric[col].dtype == object:
            try:
                X_numeric[col] = pd.to_datetime(X_numeric[col], errors="coerce").map(pd.Timestamp.toordinal)
            except:
                X_numeric = X_numeric.drop(columns=[col])

    X = X_numeric.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        print("No numeric features available.")
        return None, None, None

    if X.shape[0] < 5:
        print("Not enough data to train (need ≥5 rows).")
        return None, None, None

    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Model MSE: {mse:.4f}")

    return model, mse, X.columns.tolist()

# --------------------------
# Save Best Model
# --------------------------
def save_best_model(model, mse, feature_names):
    if model is None:
        return False

    bundle = {"model": model, "mse": mse, "features": feature_names}
    joblib.dump(bundle, MODEL_PATH)
    print("Model saved with features.")

    # Log run
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as log:
        log.write(f"{timestamp} | MSE: {mse:.4f} | Features: {len(feature_names)}\n")

    return True

# --------------------------
# Prediction Function
# --------------------------
def predict(file):
    print(f"\nPredicting for {file}...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_names = bundle["features"]

    new_data = pd.read_csv(file)

    # Preprocess new data
    X = new_data.copy()
    for col in X.columns:
        if X[col].dtype == object:
            try:
                X[col] = pd.to_datetime(X[col], errors="coerce").map(pd.Timestamp.toordinal)
            except:
                X = X.drop(columns=[col])

    # Align features
    X = X.reindex(columns=feature_names, fill_value=0)

    preds = model.predict(X)

    # Save predictions with actual target if available
    out_file = f"predictions_{os.path.basename(file)}"
    result_df = pd.DataFrame({"Prediction": preds})

    # If the file has a target column, include it
    target_col = new_data.columns[-1]
    if target_col not in feature_names:  # likely the target column
        result_df["Actual"] = new_data[target_col]

    result_df.to_csv(out_file, index=False)
    print(f"Predictions (with Actuals if available) saved to {out_file}")
    return preds

# --------------------------
# Pipeline Runner (Train + Predict)
# --------------------------
def pipeline():
    print("\nRunning pipeline...")
    df, files = load_data()
    if df is None: return

    df = preprocess(df)
    if df is None: return

    model, mse, features = train_model(df)
    if model is None: return

    save_best_model(model, mse, features)
    print("✔ Training completed.")

    # Run predictions on all CSVs immediately
    for file in files:
        predict(file)

    print("✔ Pipeline completed with predictions.\n")

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    pipeline()
