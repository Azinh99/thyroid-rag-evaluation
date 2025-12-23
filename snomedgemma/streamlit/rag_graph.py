import os
from neo4j import GraphDatabase
from llm_df import chat_with_llm
from dotenv import load_dotenv

load_dotenv(".env")


def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )


def retrieve_with_graph(question, opts):
    driver = get_driver()

    with driver.session() as s:
        rows = s.run("""
            MATCH (h:Entity)-[r:REL]->(t:Entity)
            RETURN h.name + " " + r.type + " " + t.name AS triple
            LIMIT 80
        """).values()

    triples = "\n".join(r[0] for r in rows)

    prompt = f"""
You are answering a medical multiple-choice exam.

Use ONLY the knowledge graph triples below.
Select the option BEST supported by the triples.

=== KNOWLEDGE GRAPH ===
{triples}

QUESTION:
{question}

OPTIONS:
A) {opts['A']}
B) {opts['B']}
C) {opts['C']}
D) {opts['D']}

Rules:
- Return ONLY ONE LETTER
- No explanation
- Choose the most strongly supported option

ANSWER:
"""

    return chat_with_llm(prompt).strip()