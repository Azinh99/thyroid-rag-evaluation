from rag_faiss import load_index
from rag_graph import retrieve_with_graph
from llm_df import chat_mcq
from utils import normalize_ws


def retrieve_with_hybrid(question, opts):
    docs = load_index().as_retriever(search_kwargs={"k": 6}).invoke(question)
    text = "\n".join(normalize_ws(d.page_content) for d in docs[:4])

    graph_answer = retrieve_with_graph(question, opts)

    prompt = f"""
Use TEXT + GRAPH.
Prefer TEXT if conflict.
Final answer: A/B/C/D

TEXT:
{text}

GRAPH ANSWER SUGGESTION:
{graph_answer}

QUESTION:
{question}

A) {opts['A']}
B) {opts['B']}
C) {opts['C']}
D) {opts['D']}
"""
    return chat_mcq(prompt)
