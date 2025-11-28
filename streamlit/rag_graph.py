from neo4j import GraphDatabase
from llm_df import chat_with_llm
import os

# Connect Neo4j
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def retrieve_with_graph(question, opts, model):

    with driver.session() as session:
        rows = session.run(
            """
            MATCH (a)-[r]->(b)
            RETURN a.name + " " + r.type + " " + b.name AS ctx
            LIMIT 10
            """
        ).values()

    # clean None
    ctx_list = []
    for r in rows:
        if r[0] is not None:
            ctx_list.append(r[0])

    if not ctx_list:
        ctx = ""
    else:
        ctx = "\n".join(ctx_list)

    prompt = f"""
You are a medical QA model. Use the knowledge graph context to answer the question.

CONTEXT:
{ctx}

QUESTION:
{question}

OPTIONS:
A) {opts['A']}
B) {opts['B']}
C) {opts['C']}
D) {opts['D']}

Answer ONLY with: A, B, C, or D.
"""

    return chat_with_llm(prompt, model)