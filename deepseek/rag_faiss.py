import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_df import chat_mcq
from utils import chunk_text, normalize_ws, get_driver, upsert_chunks_and_link_concepts

DATA_DIR = "data"
OUT_PATH = "output/faiss_index.pkl"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_db = None
_emb = None


def _embeddings():
    global _emb
    if _emb is None:
        _emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return _emb


def build_faiss_index(max_words=260):
    texts, metas = [], []
    driver = get_driver()

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".txt"):
            continue
        raw = open(os.path.join(DATA_DIR, fname), encoding="utf-8", errors="ignore").read()
        chunks = chunk_text(raw, max_words)

        upsert_chunks_and_link_concepts(driver, fname, chunks)

        for i, c in enumerate(chunks):
            texts.append(c)
            metas.append({"chunk_id": f"{fname}::{i}"})

    os.makedirs("output", exist_ok=True)
    db = FAISS.from_texts(texts, _embeddings(), metadatas=metas)
    pickle.dump(db, open(OUT_PATH, "wb"))
    print("FAISS index built")


def load_index():
    global _db
    if _db is None:
        _db = pickle.load(open(OUT_PATH, "rb"))
    return _db


def retrieve_with_faiss(question, opts):
    docs = load_index().as_retriever(search_kwargs={"k": 6}).invoke(question)
    ctx = "\n".join(normalize_ws(d.page_content) for d in docs[:4])

    prompt = f"""
Use ONLY the TEXT.
Final answer: A/B/C/D

TEXT:
{ctx}

QUESTION:
{question}

A) {opts['A']}
B) {opts['B']}
C) {opts['C']}
D) {opts['D']}
"""
    return chat_mcq(prompt)
