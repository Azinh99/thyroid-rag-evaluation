import os
from neo4j import GraphDatabase

from rag_faiss import load_index
from llm_df import chat_with_llm

def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )

def graph_evidence_from_chunk_ids(chunk_ids, limit_triples=30):
    driver = get_driver()
    triples = []

    with driver.session() as s:
        for cid in chunk_ids:
            rows = s.run("""
                MATCH (c:Chunk {id:$cid})-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH (e)-[r:REL]->(t:Entity)
                RETURN e.name + " " + r.type + " " + t.name AS triple
                LIMIT 50
            """, cid=cid).values()
            for r in rows:
                if r and r[0]:
                    triples.append(r[0])

            rows2 = s.run("""
                MATCH (c:Chunk {id:$cid})-[:MENTIONS]->(e:Entity)
                MATCH (e)-[:MAPS_TO]->(:Concept)-[:HAS_SNOMED]->(sn:SNOMED)
                OPTIONAL MATCH (sn)-[:IS_A]->(p:SNOMED)
                RETURN "SNOMED " + sn.id + " IS_A " + coalesce(p.id,"") AS triple
                LIMIT 50
            """, cid=cid).values()
            for r in rows2:
                if r and r[0]:
                    triples.append(r[0])

            if len(triples) >= limit_triples:
                break

    driver.close()

    seen = set()
    uniq = []
    for tr in triples:
        if tr not in seen:
            seen.add(tr)
            uniq.append(tr)

    return uniq[:limit_triples]

def retrieve_with_hybrid(question, opts):
    db = load_index()

    k = int(os.getenv("FAISS_K", "8"))
    top = int(os.getenv("HYBRID_TEXT_TOP", "4"))
    graph_top = int(os.getenv("HYBRID_GRAPH_TOP", "30"))

    docs = db.as_retriever(search_kwargs={"k": k}).get_relevant_documents(question)
    top_docs = docs[:top]

    text_ctx = "\n\n---\n\n".join(d.page_content for d in top_docs)

    chunk_ids = []
    for d in top_docs:
        md = getattr(d, "metadata", {}) or {}
        cid = md.get("chunk_id")
        if cid:
            chunk_ids.append(cid)

    graph_triples = graph_evidence_from_chunk_ids(chunk_ids, limit_triples=graph_top)
    graph_ctx = "\n".join(graph_triples) if graph_triples else "No graph evidence found."

    prompt = f"""
You are a senior medical board examiner.

Answer using ONLY the evidence below.

TEXT EVIDENCE:
{text_ctx}

GRAPH AND SNOMED EVIDENCE:
{graph_ctx}

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
