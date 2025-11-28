import os
import pandas as pd
from datetime import datetime

from llm_df import chat_with_llm, test_llm_connection
from utils import load_questions
from rag_faiss import retrieve_with_faiss
from rag_hybrid import retrieve_with_hybrid
from rag_graph import retrieve_with_graph

MODEL = "medgemma-27b-it"


def run_rag(method, questions):
    preds = []
    for q in questions:
        if method == "faiss":
            p = retrieve_with_faiss(q["q"], q["opts"], MODEL)
        elif method == "hybrid":
            p = retrieve_with_hybrid(q["q"], q["opts"], MODEL)
        else:
            p = retrieve_with_graph(q["q"], q["opts"], MODEL)
        preds.append(p)

    acc = round(100 * sum(p == q["ans"] for p, q in zip(preds, questions)) / len(questions), 2)
    return preds, acc


def evaluate_all():
    questions = load_questions("question/thyroid_questions.txt")

    print(f"Loaded {len(questions)} questions.")

    if not test_llm_connection(MODEL):
        print("❌ Model not reachable.")
        return

    methods = ["faiss", "hybrid", "graph"]
    summary = []

    for m in methods:
        print(f"-- RAG = {m} --")
        preds, acc = run_rag(m, questions)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"results_saia/{MODEL}"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/{m}_{ts}.csv"

        df = pd.DataFrame({
            "q": [x["q"] for x in questions],
            "pred": preds,
            "ans": [x["ans"] for x in questions]
        })
        df.to_csv(path, index=False)

        print(f"Saved: {path}")
        summary.append({"rag": m, "acc": acc})

    # save summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    s_path = f"results_saia/{MODEL}/SUMMARY_{ts}.csv"
    pd.DataFrame(summary).to_csv(s_path, index=False)
    print("Summary saved:", s_path)


if __name__ == "__main__":
    evaluate_all()