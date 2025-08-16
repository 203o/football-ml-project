import os

import joblib


def load_model_and_scaler():
    base_path = os.path.dirname(__file__)  # directory of predicts.py
    nb_model = joblib.load(os.path.join(base_path, "nb_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    return nb_model, scaler


os.system("streamlit run models/predicts.py")
