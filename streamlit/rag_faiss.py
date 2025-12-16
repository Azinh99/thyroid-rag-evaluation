import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils import chunk_text
from llm_df import chat_with_llm

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = "data"
OUT_PATH = "output/faiss_index.pkl"

_emb = None
_db = None


def get_emb():
    global _emb
    if _emb is None:
        _emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return _emb


def build_faiss_index():
    global _db
    texts = []

    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(".txt"):
            with open(os.path.join(DATA_DIR, f), "r", encoding="utf-8") as fp:
                raw = fp.read()
                for ch in chunk_text(raw, size=300):
                    texts.append(ch)

    emb = get_emb()
    _db = FAISS.from_texts(texts, emb)

    os.makedirs("output", exist_ok=True)
    with open(OUT_PATH, "wb") as fp:
        pickle.dump(_db, fp)

    print("FAISS index built:", OUT_PATH)


def load_index():
    global _db
    if _db is not None:
        return _db

    with open(OUT_PATH, "rb") as fp:
        _db = pickle.load(fp)
    return _db


def build_prompt(question, opts, ctx):
    return f"""
Use the following context to answer the MCQ.

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


def retrieve_with_faiss(question, opts, model):
    db = load_index()
    retriever = db.as_retriever(search_kwargs={"k": 6})
    docs = retriever.get_relevant_documents(question)

    ctx = "\n\n".join(d.page_content for d in docs[:3])
    prompt = build_prompt(question, opts, ctx)
    return chat_with_llm(prompt, model)


# ---------------------------------------
# ✅ MAIN (Run this to build FAISS index)
# ---------------------------------------
if __name__ == "__main__":
    print("📌 Building FAISS index...")
    build_faiss_index()
    print("✅ Done.")
