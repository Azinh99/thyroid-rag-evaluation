from rag_faiss import retrieve_with_faiss, load_index, build_prompt
from utils import extract_keywords
from llm_df import chat_with_llm

def retrieve_with_hybrid(question, opts, model):

    # 1) Keywords
    kws = extract_keywords(question)
    kw_query = question + " " + " ".join(kws)

    # 2) FAISS retrieval
    db = load_index()
    retriever = db.as_retriever(search_kwargs={"k": 8})
    docs = retriever.get_relevant_documents(kw_query)

    if not docs:
        return retrieve_with_faiss(question, opts, model)

    raw_contexts = list(dict.fromkeys([d.page_content for d in docs]))
    ctx = "\n\n".join(raw_contexts[:4])

    # 3) Mini reranker
    rerank_prompt = f"""
Rank documents for answering the question.

QUESTION:
{question}

DOCUMENTS:
{ctx}

Return ONLY one number: 1, 2, 3, or 4.
"""
    rerank_raw = chat_with_llm(rerank_prompt, model)

    if rerank_raw in ["1", "2", "3", "4"]:
        chosen = raw_contexts[int(rerank_raw) - 1]
    else:
        chosen = ctx

    # 4) Final QA
    final_prompt = build_prompt(question, opts, chosen)
    return chat_with_llm(final_prompt, model)