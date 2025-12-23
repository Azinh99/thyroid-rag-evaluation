import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from llm_df import chat_with_llm
from utils import chunk_text

DATA_DIR = "data"
OUT_PATH = "output/faiss_index.pkl"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_emb = None
_db = None


def get_emb():
    global _emb
    if _emb is None:
        _emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return _emb


def build_faiss_index():
    os.makedirs("output", exist_ok=True)
    texts = []

    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8", errors="ignore") as f:
                chunks = chunk_text(f.read(), max_words=400)
                texts.extend(chunks)

    db = FAISS.from_texts(texts, get_emb())

    with open(OUT_PATH, "wb") as f:
        pickle.dump(db, f)

    print("✅ FAISS index built.")


def load_index():
    global _db
    if _db:
        return _db
    with open(OUT_PATH, "rb") as f:
        _db = pickle.load(f)
    return _db


def build_prompt(context, question, opts):
    return f"""
You are a senior medical board examiner.

Use ONLY the context below.
Choose the option that is BEST supported by the evidence.
If multiple options seem plausible, choose the strongest one.

=== CONTEXT ===
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
- Be strict and evidence-based

ANSWER:
"""


def retrieve_with_faiss(question, opts):
    db = load_index()
    docs = db.as_retriever(search_kwargs={"k": 6}).get_relevant_documents(question)
    context = "\n".join(d.page_content for d in docs[:3])

    prompt = build_prompt(context, question, opts)
    return chat_with_llm(prompt).strip()