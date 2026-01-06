from utils import get_driver, neo4j_rows, normalize_ws
from rag_faiss import load_index
from llm_df import chat_mcq


def retrieve_with_graph(question, opts):
    db = load_index()
    docs = db.as_retriever(search_kwargs={"k": 6}).invoke(question)
    chunk_ids = [d.metadata.get("chunk_id") for d in docs if d.metadata.get("chunk_id")][:5]

    driver = get_driver()
    rows = neo4j_rows(
        driver,
        """
        UNWIND $ids AS cid
        MATCH (c:Chunk {id:cid})-[:MENTIONS]->(x)-[r]->(y)
        RETURN coalesce(x.name,x.cui) AS h, type(r) AS rel, coalesce(y.name,y.cui) AS t
        LIMIT 40
        """,
        {"ids": chunk_ids},
    )

    graph = "\n".join(f"{normalize_ws(r['h'])} --{r['rel']}--> {normalize_ws(r['t'])}" for r in rows)

    prompt = f"""
Use ONLY the GRAPH.
Final answer: A/B/C/D

GRAPH:
{graph}

QUESTION:
{question}

A) {opts['A']}
B) {opts['B']}
C) {opts['C']}
D) {opts['D']}
"""
    return chat_mcq(prompt)
