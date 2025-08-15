import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import numpy as np

# ===== CONFIG =====
MODEL_PATH = "nb_model.pkl"
SCALER_PATH = "scaler.pkl"

# ===== APP TITLE =====
st.title("📈 Naive Bayes Model Extender & Evaluator")

# ===== Load model and scaler =====
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    nb_model, scaler = load_model_and_scaler()
    st.success("✅ Model and scaler loaded successfully.")
except Exception as e:
    st.error(f"❌ Could not load model or scaler: {e}")
    st.stop()

# ===== File uploader =====

uploaded_file = st.file_uploader("📂 Upload new Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    # Load data
    try:
        if uploaded_file.name.endswith(".csv"):
            new_df = pd.read_csv(uploaded_file)
        else:
            new_df = pd.read_excel(uploaded_file)
        st.write("📊 Preview of Uploaded Data:", new_df.head())
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

    # Check 'Prediction' column and create rule-based features
    if "Prediction" not in new_df.columns:
        st.error("❌ 'Prediction' column missing. Cannot generate rule-based features.")
        st.stop()

    new_df["Rule_Draw"] = new_df["Prediction"].astype(str).str.contains("Draw", na=False).astype(int)
    new_df["Rule_Home"] = new_df["Prediction"].astype(str).str.contains("Home", na=False).astype(int)
    new_df["Rule_Away"] = new_df["Prediction"].astype(str).str.contains("Away", na=False).astype(int)

    # Prepare features & target
    required_features = ["H%change", "D%change", "A%change", "Rule_Draw", "Rule_Home", "Rule_Away"]
    target_col = "Result"

    missing_features = [f for f in required_features if f not in new_df.columns]
    if missing_features:
        st.error(f"❌ Missing required features: {missing_features}")
        st.stop()

    if target_col not in new_df.columns:
        st.error("❌ 'Result' column missing.")
        st.stop()

    # Drop NaN values
    new_df = new_df.dropna(subset=required_features + [target_col])
    if new_df.empty:
        st.error("❌ All rows have NaN values in required columns.")
        st.stop()

    X_new = new_df[required_features]
    y_new = new_df[target_col]

    # Scale features
    try:
        X_new_scaled = scaler.transform(X_new)
    except Exception as e:
        st.error(f"❌ Error scaling features: {e}")
        st.stop()

    # ===== Extend and evaluate =====
    if st.button("🚀 Extend Model with New Data"):
        try:
            # Accuracy before
            y_pred_before = nb_model.predict(X_new_scaled)
            acc_before = accuracy_score(y_new, y_pred_before)

            # Extend model
            nb_model.partial_fit(X_new_scaled, y_new, classes=["Home", "Draw", "Away"])

            # Accuracy after
            y_pred_after = nb_model.predict(X_new_scaled)
            acc_after = accuracy_score(y_new, y_pred_after)

            # Save updated model
            joblib.dump(nb_model, MODEL_PATH)

            st.success("✅ Model successfully extended and saved!")
            st.write(f"**Accuracy before:** {acc_before:.2%}")
            st.write(f"**Accuracy after:** {acc_after:.2%}")
            st.write(f"**Change in accuracy:** {acc_after - acc_before:+.2%}")

        except Exception as e:
            st.error(f"❌ Error extending model: {e}")
