"""
model_explainer.py
------------------
Loads the trained XGBoost model (pkl) and uses SHAP TreeExplainer to:
  1. Rank features by their absolute mean SHAP value (global importance)
  2. Derive safe/optimal operating ranges from training data
  3. Return a structured list of recommendations for the query engine

This is model-driven intelligence — unlike the static vector-DB patterns,
these insights are directly driven by what the model actually learned.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
import shap

# ─────────────────────────────────────────────────────────────
# Config (relative to CWD = d:/knowledgr base)
# ─────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("Model", "model (1).pkl")
DATA_PATH    = os.path.join("Data", "Train 1 All clean data.csv")
TARGET_COL   = "Steam_Economy_cond"

# Which output index = Steam Economy (MultiOutputRegressor has 2 outputs)
# We'll auto-detect by checking output 0 vs 1 correlation with Steam Economy
STEAM_OUTPUT_IDX = 0

# How many rows to use for SHAP background (full data is 150k+ rows, slow)
# 2000 rows is enough for stable SHAP values
SHAP_SAMPLE_SIZE = 2000


# ─────────────────────────────────────────────────────────────
# Model + Data Loader
# ─────────────────────────────────────────────────────────────

def load_model():
    """Load the cloudpickle sklearn MultiOutputRegressor model."""
    print(f"[MODEL] Loading model from: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"[MODEL] Loaded: {type(model).__name__} wrapping {type(model.estimator).__name__}")
    return model


def load_training_data(model):
    """Load training CSV, keep only model input features, drop NaN rows."""
    print(f"[DATA]  Loading training data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    feature_cols = list(model.feature_names_in_)

    # Check which model features exist in CSV
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[WARN]  Features missing from CSV (filling with 0s): {missing}")
        for c in missing:
            df[c] = 0.0

    # Keep features + target if available
    cols_to_keep = feature_cols + ([TARGET_COL] if TARGET_COL in df.columns else [])
    df = df[cols_to_keep].dropna()
    print(f"[DATA]  Shape after cleaning: {df.shape}")
    return df, feature_cols


# ─────────────────────────────────────────────────────────────
# SHAP Explainer
# ─────────────────────────────────────────────────────────────

def compute_shap_importance(model, df, feature_cols, target_steam=None):
    """
    Compute mean |SHAP| values for the Steam Economy output.

    Returns a DataFrame:
        feature | mean_abs_shap | direction | best_range_low | best_range_high
    """
    X = df[feature_cols].copy()

    # Sample for speed
    if len(X) > SHAP_SAMPLE_SIZE:
        X_sample = X.sample(n=SHAP_SAMPLE_SIZE, random_state=42)
    else:
        X_sample = X

    print(f"[SHAP]  Computing SHAP values on {len(X_sample)} samples ...")

    # Access the XGBoost estimator for Steam Economy output
    xgb_model = model.estimators_[STEAM_OUTPUT_IDX]

    # TreeExplainer is fast and exact for XGBoost
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)  # shape: (n_samples, n_features)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)  # (n_features,)
    mean_shap     = shap_values.mean(axis=0)           # direction: + means increases SE

    # Build result rows
    rows = []
    for i, feat in enumerate(feature_cols):
        if feat not in df.columns:
            continue

        feat_vals  = df[feat]
        direction  = "increase" if mean_shap[i] > 0 else "decrease"

        # Safe/optimal range: percentile band of the top-25% Steam Economy records
        if TARGET_COL in df.columns:
            top_q = df[TARGET_COL].quantile(0.75)
            high_se_df = df[df[TARGET_COL] >= top_q]
        else:
            # Fallback: use overall distribution
            high_se_df = df

        feat_high = high_se_df[feat] if feat in high_se_df.columns else feat_vals
        safe_low  = round(float(feat_high.quantile(0.10)), 4)
        safe_high = round(float(feat_high.quantile(0.90)), 4)
        current_median = round(float(feat_vals.median()), 4)

        rows.append({
            "feature":          feat,
            "mean_abs_shap":    round(float(mean_abs_shap[i]), 5),
            "direction":        direction,           # to improve Steam Economy
            "safe_range_low":   safe_low,
            "safe_range_high":  safe_high,
            "current_median":   current_median,
        })

    result_df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    result_df["rank"] = result_df.index + 1
    return result_df


# ─────────────────────────────────────────────────────────────
# Main entry: get SHAP recommendations
# ─────────────────────────────────────────────────────────────

def get_shap_recommendations(top_n: int = 15, target_steam: float = None):
    """
    Full pipeline: load model → load data → compute SHAP → return DataFrame.

    Parameters
    ----------
    top_n        : Return top N most important features
    target_steam : Not used for filtering here, passed for context

    Returns
    -------
    pd.DataFrame with columns:
        rank, feature, mean_abs_shap, direction,
        safe_range_low, safe_range_high, current_median
    """
    model = load_model()
    df, feature_cols = load_training_data(model)
    importance_df = compute_shap_importance(model, df, feature_cols, target_steam)
    return importance_df.head(top_n)


# ─────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = get_shap_recommendations(top_n=15)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n[SHAP] Top Features Driving Steam Economy:\n")
    print(df.to_string(index=False))
