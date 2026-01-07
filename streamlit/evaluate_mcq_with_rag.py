from rag_faiss import retrieve_with_faiss
from rag_graph import retrieve_with_graph
from rag_hybrid import retrieve_with_hybrid
from utils import load_questions
import pandas as pd, os
from datetime import datetime

def evaluate():
    qs = load_questions("question/thyroid_questions.txt")
    methods = {
        "faiss": retrieve_with_faiss,
        "graph": retrieve_with_graph,
        "hybrid": retrieve_with_hybrid,
    }

    summary = []

    for name, fn in methods.items():
        preds = []
        for q in qs:
            preds.append(fn(q["q"], q["opts"], None))

        acc = sum(p == q["ans"] for p, q in zip(preds, qs)) / len(qs)
        summary.append({"method": name, "accuracy": acc})

    df = pd.DataFrame(summary)
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/summary_{datetime.now().strftime('%H%M%S')}.csv", index=False)
    print(df)

if __name__ == "__main__":
    evaluate()
