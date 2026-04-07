# pipeline/text_converter.py

def patterns_to_text(patterns):
    texts = []

    for p in patterns:
        condition = " AND ".join([
            f"{f} is {r}" for f, r in zip(p["features"], p["ranges"])
        ])

        text = (
            f"When {condition}, "
            f"Steam Economy is {p['avg_steam']} "
            f"based on {p['count']} records."
        )

        texts.append(text)

    print(f"[INFO] Texts created: {len(texts)}")

    return texts