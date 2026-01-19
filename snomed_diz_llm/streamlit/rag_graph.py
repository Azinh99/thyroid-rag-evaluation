import os
from neo4j import GraphDatabase

from llm_df import chat_with_llm

def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )

def extract_keywords(question: str):
    prompt = f"""
Extract 5-8 key medical terms from the question.
Return as a comma-separated list only.

QUESTION:
{question}
"""
    out = chat_with_llm(prompt).strip()
    terms = [t.strip() for t in out.split(",") if t.strip()]
    return terms[:8]

def retrieve_with_graph(question, opts):
    driver = get_driver()
    terms = extract_keywords(question)
    limit_total = int(os.getenv("GRAPH_TRIPLES_LIMIT", "50"))

    triples = []

    with driver.session() as s:
        for term in terms:
            rows = s.run("""
                MATCH (e:Entity)
                WHERE e.name_lc CONTAINS toLower($term)
                OPTIONAL MATCH (e)-[r:REL]->(t:Entity)
                RETURN e.name + " " + r.type + " " + t.name AS triple
                LIMIT 30
            """, term=term).values()
            for r in rows:
                if r and r[0]:
                    triples.append(r[0])

            rows2 = s.run("""
                MATCH (e:Entity)
                WHERE e.name_lc CONTAINS toLower($term)
                MATCH (e)-[:MAPS_TO]->(:Concept)-[:HAS_SNOMED]->(sn:SNOMED)
                OPTIONAL MATCH (sn)-[:IS_A]->(p:SNOMED)
                RETURN "SNOMED " + sn.id + " IS_A " + coalesce(p.id,"") AS triple
                LIMIT 30
            """, term=term).values()
            for r in rows2:
                if r and r[0]:
                    triples.append(r[0])

            if len(triples) >= limit_total:
                break

    driver.close()

    seen = set()
    uniq = []
    for tr in triples:
        if tr not in seen:
            seen.add(tr)
            uniq.append(tr)

    kg_text = "\n".join(uniq[:limit_total]) if uniq else "No graph evidence found."

    prompt = f"""
You are answering a medical multiple-choice exam.

Use ONLY the graph evidence below.

GRAPH EVIDENCE:
{kg_text}

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

ANSWER:
"""
    return chat_with_llm(prompt).strip().upper()
