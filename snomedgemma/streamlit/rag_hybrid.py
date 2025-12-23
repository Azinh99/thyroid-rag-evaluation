# rag_hybrid.py
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from rag_faiss import load_index
from llm_df import chat_with_llm

load_dotenv(".env")

# --------------------------------------------------
# Neo4j driver
# --------------------------------------------------
def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )

# --------------------------------------------------
# Step 1: Retrieve graph context based on chunks
# --------------------------------------------------
def retrieve_graph_context_from_chunks(chunks, max_triples=25):
    """
    Traverses the graph starting from Chunk.text matches
    and returns a small, question-aware subgraph.
    """
    driver = get_driver()
    triples = []

    with driver.session() as s:
        for chunk in chunks:
            rows = s.run(
                """
                MATCH (c:Chunk)
                WHERE c.text CONTAINS $chunk
                MATCH (e)-[r]->(f)
                WHERE (e)-->(c) OR (f)-->(c)
                RETURN e.name + " " + type(r) + " " + f.name AS triple
                LIMIT 5
                """,
                chunk=chunk[:300]  # safety
            ).values()

            for r in rows:
                triples.append(r[0])

            if len(triples) >= max_triples:
                break

    return list(dict.fromkeys(triples))[:max_triples]

# --------------------------------------------------
# Step 2: Hybrid retrieval
# --------------------------------------------------
def retrieve_with_hybrid(question, opts):
    # ---------- TEXT RETRIEVAL ----------
    db = load_index()
    docs = db.as_retriever(search_kwargs={"k": 6}).get_relevant_documents(question)
    text_chunks = [d.page_content for d in docs[:4]]
    text_ctx = "\n\n---\n\n".join(text_chunks)

    # ---------- GRAPH RETRIEVAL ----------
    graph_triples = retrieve_graph_context_from_chunks(text_chunks)

    graph_ctx = "\n".join(graph_triples) if graph_triples else "No graph evidence found."

    # ---------- FINAL PROMPT ----------
    prompt = f"""
You are a senior medical board examiner.

You must answer the multiple-choice question using ONLY the evidence provided below.
You are NOT allowed to use external knowledge.

======================
TEXTUAL EVIDENCE
======================
{text_ctx}

======================
GRAPH EVIDENCE
(Subject – Relation – Object)
======================
{graph_ctx}

======================
QUESTION
======================
{question}

OPTIONS:
A) {opts['A']}
B) {opts['B']}
C) {opts['C']}
D) {opts['D']}

INSTRUCTIONS:
- Choose the option BEST supported by the evidence.
- If two options are partially correct, choose the stronger one.
- Do NOT explain your reasoning.
- Return ONLY ONE LETTER: A, B, C, or D.

ANSWER:
"""

    return chat_with_llm(prompt).strip().upper()