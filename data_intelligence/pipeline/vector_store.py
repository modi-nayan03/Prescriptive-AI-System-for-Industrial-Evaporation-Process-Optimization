# pipeline/vector_store.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_vector_store(texts, save_path="vector_db_multi"):
    embeddings = HuggingFaceEmbeddings()

    db = FAISS.from_texts(texts, embeddings)

    db.save_local(save_path)

    print(f"[INFO] Vector DB saved at: {save_path}")

    return db