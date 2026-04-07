# pipeline/binning.py

import pandas as pd
from config import NUM_BINS, TARGET_COLUMN

def apply_binning(df):
    binned_df = df.copy()

    for col in df.columns:
        if col == TARGET_COLUMN:
            continue

        try:
            binned_df[col] = pd.qcut(
                df[col],
                NUM_BINS,
                duplicates='drop'
            )
        except Exception:
            continue

    return binned_df