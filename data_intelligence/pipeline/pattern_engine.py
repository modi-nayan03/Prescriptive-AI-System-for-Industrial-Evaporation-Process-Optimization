# pipeline/pattern_engine.py

from config import TARGET_COLUMN, MIN_RECORDS_THRESHOLD

def extract_patterns(df, combinations):
    patterns = []

    for combo in combinations:
        grouped = df.groupby(list(combo))[TARGET_COLUMN].agg(
            ["mean", "count"]
        ).reset_index()

        for _, row in grouped.iterrows():
            if row["count"] < MIN_RECORDS_THRESHOLD:
                continue

            patterns.append({
                "features": combo,
                "ranges": [str(row[f]) for f in combo],
                "avg_steam": round(row["mean"], 3),
                "count": int(row["count"])
            })

    print(f"[INFO] Patterns extracted: {len(patterns)}")

    return patterns