# pipeline/model_loader.py

import joblib
import os

MODEL_PATH = os.path.join("Model", "model (1).pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    print(f"[INFO] Model loaded from {MODEL_PATH}")

    return model