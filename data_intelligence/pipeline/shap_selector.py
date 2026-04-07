# pipeline/shap_selector.py

import shap
import pandas as pd
from config import TARGET_COLUMN


class ShapFeatureSelector:

    def __init__(self, model):
        self.model = model
        # If model is a MultiOutputRegressor, use the first inner estimator
        if hasattr(model, 'estimators_'):
            self.inner_model = model.estimators_[0]
        else:
            self.inner_model = model

        try:
            self.explainer = shap.TreeExplainer(self.inner_model)
        except:
            # fallback
            self.explainer = shap.Explainer(self.inner_model)

    def get_top_features(self, df, top_k=6):
        sample_df = df.sample(min(5000, len(df)), random_state=42)

        if hasattr(self.model, 'feature_names_in_'):
            feature_cols = list(self.model.feature_names_in_)
            missing = [c for c in feature_cols if c not in sample_df.columns]
            for c in missing:
                sample_df[c] = 0.0
            X = sample_df[feature_cols]
        else:
            X = sample_df.drop(columns=[TARGET_COLUMN], errors='ignore')

        shap_values = self.explainer(X)

        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": abs(shap_values.values).mean(axis=0)
        })

        importance = importance.sort_values(by="importance", ascending=False)

        top_features = importance["feature"].head(top_k).tolist()

        print(f"[INFO] Top Features: {top_features}")

        return top_features