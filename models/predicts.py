import streamlit as st
import pandas as pd
import os
import joblib
from io import BytesIO
from sklearn.naive_bayes import GaussianNB


# === Load model & scaler ===
@st.cache_resource
def load_model_and_scaler():
    base_path = os.path.dirname(__file__)  # directory of predicts.py
    nb_model = joblib.load(os.path.join(base_path, "nb_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    return nb_model, scaler


nb_model, scaler = load_model_and_scaler()


# === Rule-based prediction function ===
def predict_result(row):
    h = row["H%change"]
    d = row["D%change"]
    a = row["A%change"]

    prediction_flags = []

    if abs(d) > abs(h) and abs(d) > abs(a):
        prediction_flags.append("Draw")

    if h < avg_h_pct_change_away:
        prediction_flags.append("Away")
    elif (
        avg_h_pct_change_home > avg_h_pct_change_away
        and avg_h_pct_change_home > avg_h_pct_change_draw
    ):
        prediction_flags.append("Home")

    if h > 0 and d > 0 and a > 0:
        if (
            avg_h_pct_change_home < avg_h_pct_change_away
            and avg_h_pct_change_draw < avg_h_pct_change_away
        ):
            prediction_flags.append("Away")
        elif (
            avg_h_pct_change_home > avg_h_pct_change_away
            and avg_h_pct_change_draw > avg_h_pct_change_away
        ):
            prediction_flags.append("Home")

    if not prediction_flags:
        return "No Prediction"
    elif len(prediction_flags) == 1:
        return prediction_flags[0]
    else:
        return "/".join(sorted(set(prediction_flags)))


# === Streamlit UI ===
st.title("⚽ Match Outcome Predictor")
st.write("Upload your Excel file to get predictions or retrain the model.")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    # Load uploaded file into DataFrame
    df = pd.read_excel(uploaded_file)

    # Calculate averages for prediction function
    if "Result" in df.columns:
        avg_h_pct_change_home = df[df["Results"] == "Home"]["H%change"].mean()
        avg_h_pct_change_draw = df[df["Results"] == "Draw"]["H%change"].mean()
        avg_h_pct_change_away = df[df["Results"] == "Away"]["H%change"].mean()
    else:
        avg_h_pct_change_home = avg_h_pct_change_draw = avg_h_pct_change_away = 0

    # Apply predictions
    df["Rule_Prediction"] = df.apply(predict_result, axis=1)

    # Binary rule columns
    df["Rule_Draw"] = (df["Rule_Prediction"] == "Draw").astype(int)
    df["Rule_Home"] = (df["Rule_Prediction"] == "Home").astype(int)
    df["Rule_Away"] = (df["Rule_Prediction"] == "Away").astype(int)

    # Prepare features
    features = [
        "H%change",
        "D%change",
        "A%change",
        "Rule_Draw",
        "Rule_Home",
        "Rule_Away",
    ]
    X_new = df[features]

    # Scale features
    X_new_scaled = scaler.transform(X_new)

    # Model prediction
    df["Model_Prediction"] = nb_model.predict(X_new_scaled)

    # Show results
    st.subheader("📊 Predictions")
    st.dataframe(df)

    # Download predictions
    output = BytesIO()
    df.to_excel(output, index=False)
    st.download_button(
        label="📥 Download Predictions as Excel",
        data=output.getvalue(),
        file_name="Predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # === Retrain Model Button ===
    if "Result" in df.columns:
        if st.button("🔄 Retrain Model with This Data"):
            y_new = df["Result"]
            classes = ["Home", "Draw", "Away"]
            if hasattr(nb_model, "partial_fit"):
                nb_model.partial_fit(X_new_scaled, y_new, classes=classes)
                joblib.dump(nb_model, r"C:\Users\komen\Desktop\Models\nb_model.pkl")
                st.success("✅ Model retrained and saved successfully!")
            else:
                st.error("❌ This model type does not support incremental training.")
    else:
        st.warning("⚠️ Cannot retrain without 'Result' column in the uploaded file.")
