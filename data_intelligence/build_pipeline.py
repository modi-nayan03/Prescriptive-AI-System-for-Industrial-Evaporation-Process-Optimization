# build_pipeline.py

from pipeline.loader import load_data
from pipeline.binning import apply_binning
from pipeline.pattern_engine import extract_patterns
from pipeline.text_converter import patterns_to_text
from pipeline.vector_store import build_vector_store

def run_pipeline():
    print("\n[STEP 1] Loading Data...")
    df = load_data()

    print("\n[STEP 2] Applying Binning...")
    binned_df = apply_binning(df)

    print("\n[STEP 3] Extracting Patterns...")
    patterns = extract_patterns(binned_df)

    print("\n[STEP 4] Converting to Text...")
    texts = patterns_to_text(patterns)

    print("\n[STEP 5] Building Vector DB...")
    build_vector_store(texts)

    print("\n[SUCCESS] Data Intelligence Pipeline Completed!")

if __name__ == "__main__":
    run_pipeline()
