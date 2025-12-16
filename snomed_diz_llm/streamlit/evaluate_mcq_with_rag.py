import os
import re
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(".env")

from rag_faiss import retrieve_with_faiss, build_faiss_index
from rag_graph import retrieve_with_graph
from rag_hybrid import retrieve_with_hybrid
from llm_df import test_llm_connection

QUESTION_FILE = "question/thyroid_questions.txt"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def load_questions(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Question file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

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

    print(f"Loaded {len(qs)} questions")
    return qs


def run(method, questions):
    preds = []
    correct_flags = []

    for q in questions:
        if method == "faiss":
            pred = retrieve_with_faiss(q["q"], q["opts"])
        elif method == "graph":
            pred = retrieve_with_graph(q["q"], q["opts"])
        else:
            pred = retrieve_with_hybrid(q["q"], q["opts"])

        pred = pred.strip().upper()
        if pred not in ["A", "B", "C", "D"]:
            pred = "A"

        preds.append(pred)
        correct_flags.append(pred in q["ans"])

    if len(correct_flags) == 0:
        return preds, correct_flags, 0.0

    acc = round(100 * sum(correct_flags) / len(correct_flags), 2)
    return preds, correct_flags, acc


def run_all():
    if not test_llm_connection():
        print("❌ LLM unreachable")
        return

    questions = load_questions(QUESTION_FILE)

    if len(questions) == 0:
        print("❌ No questions loaded. Check formatting.")
        return

    if not os.path.exists("output/faiss_index.pkl"):
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

        df.to_csv(f"{OUT_DIR}/{m}_{ts}.csv", index=False)
        summary.append({"method": m, "accuracy": acc})

        print(f"✅ {m} → {acc}%")

    pd.DataFrame(summary).to_csv(
        f"{OUT_DIR}/SUMMARY_{ts}.csv", index=False
    )


if __name__ == "__main__":
    run_all()