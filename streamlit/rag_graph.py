# rag_graph.py
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

from llm_df import chat_with_llm
from utils import extract_keywords

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

def retrieve_with_graph(question, opts, model=None):
    kws = extract_keywords(question, max_k=12)

    with driver.session(database=os.getenv("NEO4J_DB")) as s:
        rows = s.run(
            """
            MATCH (a)-[r]->(b)
            WHERE any(k IN $kws WHERE
                toLower(a.name) CONTAINS k OR
                toLower(b.name) CONTAINS k OR
                toLower(r.type) CONTAINS k
            )
            RETURN a.name, r.type, b.name
            LIMIT 25
            """,
            kws=kws,
        ).values()

    graph_ctx = "\n".join(f"{h} {r} {t}" for h, r, t in rows)

    prompt = f"""
GRAPH CONTEXT:
{graph_ctx}

QUESTION:
{question}

OPTIONS:
A) {opts['A']}
B) {opts['B']}
C) {opts['C']}
D) {opts['D']}
"""
    return chat_with_llm(prompt)
