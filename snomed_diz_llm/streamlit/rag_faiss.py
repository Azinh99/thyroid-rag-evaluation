import os
import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from utils import chunk_text
from llm_df import chat_with_llm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_PATH = PROJECT_ROOT / "output" / "faiss_index.pkl"

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_emb = None
_db = None

def get_emb():
    global _emb
    if _emb is None:
        _emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return _emb

def build_faiss_index():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    chunk_words = int(os.getenv("CHUNK_WORDS", "300"))

    texts = []
    metas = []

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        fname = file_path.name
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            chunks = chunk_text(f.read(), max_words=chunk_words)

        for i, ch in enumerate(chunks):
            chunk_id = f"{fname}::chunk_{i:04d}"
            texts.append(ch)
            metas.append({"chunk_id": chunk_id, "source": fname})

    db = FAISS.from_texts(texts, get_emb(), metadatas=metas)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(db, f)

def load_index():
    global _db
    if _db is not None:
        return _db
    with open(OUT_PATH, "rb") as f:
        _db = pickle.load(f)
    return _db

def build_prompt(context, question, opts):
    return f"""
You are a senior medical board examiner.

Use ONLY the context below.
Choose the option BEST supported by the evidence.

CONTEXT:
{context}

QUESTION:
{question}

OPTIONS:
A) {opts['A']}
B) {opts['B']}
C) {opts['C']}
D) {opts['D']}

Rules:
- Return ONLY ONE LETTER (A, B, C, or D)
- No explanation

ANSWER:
"""

def retrieve_with_faiss(question, opts):
    db = load_index()
    k = int(os.getenv("FAISS_K", "8"))
    top = int(os.getenv("HYBRID_TEXT_TOP", "4"))

    docs = db.as_retriever(search_kwargs={"k": k}).get_relevant_documents(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs[:top])

    prompt = build_prompt(context, question, opts)
    return chat_with_llm(prompt).strip().upper()
