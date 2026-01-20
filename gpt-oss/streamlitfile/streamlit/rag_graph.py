import os
from neo4j import GraphDatabase
from llm_df import chat_with_llm


def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )


def retrieve_with_graph(question, opts):
    driver = get_driver()
    triples = []

    with driver.session(database=os.getenv("NEO4J_DB")) as s:
        rows = s.run("""
            MATCH (h:Entity)-[r:REL]->(t:Entity)
            RETURN h.name + " " + r.type + " " + t.name AS triple
            LIMIT 50
        """).values()

        triples = [r[0] for r in rows if r and r[0]]

    driver.close()

    kg = "\n".join(triples) if triples else "No graph evidence."

    prompt = f"""
Use ONLY the graph evidence.

GRAPH:
{kg}

QUESTION:
{question}

OPTIONS:
A) {opts['A']}
B) {opts['B']}
C) {opts['C']}
D) {opts['D']}

Return ONLY A/B/C/D.
ANSWER:
"""
    return chat_with_llm(prompt).strip().upper()