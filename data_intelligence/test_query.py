import argparse
import warnings

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")

def test_query(query: str, k: int = 5, db_path: str = "vector_db_multi"):
    print(f"[INFO] Loading vector database from '{db_path}'...")
    embeddings = HuggingFaceEmbeddings()
    
    try:
        # allow_dangerous_deserialization=True is required for loading local FAISS dbs
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"[ERROR] Failed to load Vector DB. Did you run the pipeline to create '{db_path}'?")
        print(f"Error details: {e}")
        return

    print(f"\nQuerying: '{query}'")
    print("-" * 60)
    
    # Perform similarity search
    docs = db.similarity_search(query, k=k)
    
    if not docs:
        print("No documents found.")
        return

    # Print results
    for i, doc in enumerate(docs, 1):
        print(f"--- Document {i} ---")
        print(doc.page_content)
        print("-" * 60)
        
    print(f"\n[SUCCESS] Retrieved {len(docs)} documents based on feature combinations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test querying the SHAP-driven multi-feature Vector DB.")
    parser.add_argument(
        "--query", "-q", 
        type=str, 
        default="improve the steam economy 3.2 to 3.5 which parameter should i change", 
        help="The query to search the vector database."
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=5, 
        help="Number of documents to retrieve."
    )
    parser.add_argument(
        "--db", 
        type=str, 
        default="vector_db_multi", 
        help="Path to the Vector DB to load."
    )
    
    args = parser.parse_args()
    test_query(query=args.query, k=args.k, db_path=args.db)
