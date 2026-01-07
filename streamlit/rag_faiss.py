# rag_faiss.py
import os
import pickle
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from utils import chunk_text
from llm_df import chat_with_llm

# -------------------------------------------------
# Environment
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Path handling (FIXED)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
OUT_PATH = os.path.join(OUTPUT_DIR, "faiss_index.pkl")

# -------------------------------------------------
# Config
# -------------------------------------------------
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "300"))
FAISS_K = int(os.getenv("FAISS_K", "6"))

# -------------------------------------------------
# Globals
# -------------------------------------------------
_db = None
_emb = None


def get_emb():
    global _emb
    if _emb is None:
        _emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return _emb


# -------------------------------------------------
# Build FAISS index
# -------------------------------------------------
def build_faiss_index():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    texts = []

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.lower().endswith(".txt"):
            continue

        fpath = os.path.join(DATA_DIR, fname)
        with open(fpath, encoding="utf-8", errors="ignore") as fp:
            raw = fp.read()
            for ch in chunk_text(raw, size=CHUNK_WORDS):
                texts.append(ch)

    if not texts:
        raise RuntimeError("No text chunks found for FAISS indexing.")

    db = FAISS.from_texts(texts, get_emb())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUT_PATH, "wb") as fp:
        pickle.dump(db, fp)

    print(f"FAISS index built with {len(texts)} chunks")
    print(f"Saved to: {OUT_PATH}")


# -------------------------------------------------
# Load FAISS index
# -------------------------------------------------
def load_index():
    global _db
    if _db is None:
        if not os.path.exists(OUT_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {OUT_PATH}. "
                f"Run build_faiss_index() first."
            )
        with open(OUT_PATH, "rb") as fp:
            _db = pickle.load(fp)
    return _db


# -------------------------------------------------
# FAISS-based RAG retrieval
# -------------------------------------------------
def retrieve_with_faiss(question, opts, model=None):
    db = load_index()

    docs = db.as_retriever(
        search_kwargs={"k": FAISS_K}
    ).get_relevant_documents(question)

    ctx = "\n\n".join(d.page_content for d in docs[:3])

    prompt = f"""
Use ONLY the context below to answer the MCQ.

CONTEXT:
{ctx}

QUESTION:
{question}

OPTIONS:
A) {opts['A']}
B) {opts['B']}
C) {opts['C']}
D) {opts['D']}

Return ONLY one letter: A, B, C, or D.
"""

    return chat_with_llm(prompt)


# -------------------------------------------------
# CLI helper
# -------------------------------------------------
if __name__ == "__main__":
    build_faiss_index()
