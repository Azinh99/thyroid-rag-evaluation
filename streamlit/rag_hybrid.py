# rag_hybrid.py
import os, re
from neo4j import GraphDatabase
from rag_faiss import load_index
from utils import extract_keywords
from llm_df import chat_with_llm

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

def retrieve_with_hybrid(q, opts, model=None):
    kws = extract_keywords(q)
    db = load_index()

    docs = db.as_retriever(search_kwargs={"k": 8}).get_relevant_documents(
        q + " " + " ".join(kws)
    )
    chunks = [d.page_content for d in docs[:4]]

    terms = set()
    for c in chunks:
        terms.update(re.findall(r"[A-Za-z]{4,}", c.lower()))

    with driver.session(database=os.getenv("NEO4J_DB")) as s:
        rows = s.run(
            """
            MATCH (a)-[r]->(b)
            WHERE any(k IN $t WHERE toLower(a.name) CONTAINS k)
            RETURN a.name,r.type,b.name LIMIT 30
            """,
            t=list(terms),
        ).values()

    graph = "\n".join(f"{h} {r} {t}" for h, r, t in rows)
    text = "\n---\n".join(chunks)

    prompt = f"""
TEXT CONTEXT:
{text}

GRAPH CONTEXT:
{graph}

QUESTION:
{q}

OPTIONS:
A) {opts['A']}
B) {opts['B']}
C) {opts['C']}
D) {opts['D']}
"""

    return chat_with_llm(prompt)
