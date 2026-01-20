import os
from neo4j import GraphDatabase
from rag_faiss import load_index
from llm_df import chat_with_llm


def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )


def graph_from_chunks(chunk_ids, limit=30):
    driver = get_driver()
    triples = []

    with driver.session(database=os.getenv("NEO4J_DB")) as s:
        for cid in chunk_ids:
            rows = s.run("""
                MATCH (c:Chunk {id:$cid})-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH (e)-[r:REL]->(t:Entity)
                RETURN e.name + " " + r.type + " " + t.name AS triple
                LIMIT 30
            """, cid=cid).values()

            for r in rows:
                if r and r[0]:
                    triples.append(r[0])

            if len(triples) >= limit:
                break

    driver.close()
    return list(dict.fromkeys(triples))[:limit]


def retrieve_with_hybrid(question, opts):
    db = load_index()
    docs = db.as_retriever(search_kwargs={"k": 8}).get_relevant_documents(question)
    top_docs = docs[:4]

    text_ctx = "\n\n---\n\n".join(d.page_content for d in top_docs)
    chunk_ids = [d.metadata.get("chunk_id") for d in top_docs if d.metadata]

    graph_ctx = "\n".join(graph_from_chunks(chunk_ids)) or "No graph evidence."

    prompt = f"""
TEXT:
{text_ctx}

GRAPH:
{graph_ctx}

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