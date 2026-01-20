import os
import re
import pandas as pd
from datetime import datetime
from pathlib import Path

from llm_df import test_llm_connection
from rag_faiss import retrieve_with_faiss, build_faiss_index
from rag_graph import retrieve_with_graph
from rag_hybrid import retrieve_with_hybrid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTION_FILE = PROJECT_ROOT / "question" / "thyroid_questions.txt"
OUT_DIR = PROJECT_ROOT / "output" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_questions(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    txt = path.read_text(encoding="utf-8", errors="ignore")

    pattern = re.compile(
        r"""
        \d+\.\s*(.*?)\n
        \s*A\)\s*(.*?)\n
        \s*B\)\s*(.*?)\n
        \s*C\)\s*(.*?)\n
        \s*D\)\s*(.*?)\n
        \s*(?:Correct\s+Answer|Answer)\s*:\s*([A-D](?:\s+or\s+[A-D])*)
        """,
        re.VERBOSE | re.IGNORECASE
    )

    qs = []
    for m in pattern.finditer(txt):
        ans_raw = m.group(6).upper()
        valid_answers = [a.strip() for a in ans_raw.split("OR")]

        qs.append({
            "q": m.group(1).strip(),
            "opts": {
                "A": m.group(2).strip(),
                "B": m.group(3).strip(),
                "C": m.group(4).strip(),
                "D": m.group(5).strip(),
            },
            "ans": valid_answers
        })

    return qs

def run(method, questions):
    preds = []
    flags = []

    for q in questions:
        if method == "faiss":
            pred = retrieve_with_faiss(q["q"], q["opts"])
        elif method == "graph":
            pred = retrieve_with_graph(q["q"], q["opts"])
        else:
            pred = retrieve_with_hybrid(q["q"], q["opts"])

        pred = (pred or "").strip().upper()
        if pred not in ["A", "B", "C", "D"]:
            pred = "A"

        preds.append(pred)
        flags.append(pred in q["ans"])

    acc = round(100.0 * sum(flags) / max(1, len(flags)), 2)
    return preds, flags, acc

def run_all():
    if not test_llm_connection():
        print("LLM is unreachable.")
        return

    questions = load_questions(QUESTION_FILE)
    if not questions:
        print("No questions loaded. Check formatting.")
        return

    faiss_path = PROJECT_ROOT / "output" / "faiss_index.pkl"
    if not faiss_path.exists():
        build_faiss_index()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    methods = ["faiss", "graph", "hybrid"]
    summary = []

    for m in methods:
        preds, flags, acc = run(m, questions)

        df = pd.DataFrame({
            "question": [x["q"] for x in questions],
            "pred": preds,
            "correct": [",".join(x["ans"]) for x in questions],
            "is_correct": flags
        })
        df.to_csv(OUT_DIR / f"{m}_{ts}.csv", index=False)
        summary.append({"method": m, "accuracy": acc})
        print(f"{m}: {acc}%")

    pd.DataFrame(summary).to_csv(OUT_DIR / f"SUMMARY_{ts}.csv", index=False)

if __name__ == "__main__":
    run_all()