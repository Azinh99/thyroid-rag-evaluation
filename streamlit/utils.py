# utils.py
import os
import re
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# -----------------------
# Neo4j helpers
# -----------------------
_DRIVER = None

def get_driver():
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
        )
    return _DRIVER


def neo4j_scalar(query: str) -> int:
    driver = get_driver()
    with driver.session(database=os.getenv("NEO4J_DB")) as s:
        rec = s.run(query).single()
        return int(list(rec.values())[0]) if rec else 0


# -----------------------
# Text utilities
# -----------------------
def chunk_text(text: str, size: int = 300):
    words = text.split()
    out, buf = [], []
    for w in words:
        buf.append(w)
        if len(buf) >= size:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out


def extract_keywords(text: str, max_k: int = 20):
    tokens = re.findall(r"[A-Za-z]{4,}", text.lower())
    stop = {
        "this","that","with","from","into","about",
        "when","what","which","where","patient","patients"
    }
    out, seen = [], set()
    for t in tokens:
        if t not in stop and t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= max_k:
            break
    return out


# -----------------------
# MCQ loader
# -----------------------
Q_PATTERN = re.compile(
    r"\s*\d+[\.\)]\s*(.*?)\n"
    r"\s*A\)\s*(.*?)\n"
    r"\s*B\)\s*(.*?)\n"
    r"\s*C\)\s*(.*?)\n"
    r"\s*D\)\s*(.*?)\n"
    r"\s*(?:Answer|Correct Answer)[: ]+\s*([A-D])",
    re.MULTILINE
)

def load_questions(path):
    with open(path, encoding="utf-8") as f:
        data = f.read()

    qs = []
    for m in Q_PATTERN.finditer(data):
        qs.append({
            "q": m.group(1).strip(),
            "opts": {
                "A": m.group(2).strip(),
                "B": m.group(3).strip(),
                "C": m.group(4).strip(),
                "D": m.group(5).strip(),
            },
            "ans": m.group(6).upper(),
        })
    return qs
