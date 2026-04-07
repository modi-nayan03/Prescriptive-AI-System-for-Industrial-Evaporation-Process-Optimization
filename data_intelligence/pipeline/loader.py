# pipeline/loader.py

import pandas as pd
from config import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.dropna()

    print(f"[INFO] Data loaded: {df.shape}")

    return df