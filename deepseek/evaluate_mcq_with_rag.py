import os
import re
import time
import pandas as pd
from datetime import datetime

from llm_df import test_llm_connection
from rag_faiss import retrieve_with_faiss, build_faiss_index
from rag_graph import retrieve_with_graph
from rag_hybrid import retrieve_with_hybrid
from utils import safe_choice_letter, get_driver, neo4j_scalar

QUESTION_FILE = "question/thyroid_questions.txt"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

def load_questions(path):
    txt = open(path, encoding="utf-8").read()
    pat = re.compile(r"(\d+).*?A\)(.*?)\n.*?B\)(.*?)\n.*?C\)(.*?)\n.*?D\)(.*?)\n.*?Answer:\s*([A-D])", re.S)
    qs = []
    for m in pat.finditer(txt):
        qs.append({
            "num": int(m.group(1)),
            "question": m.group(0),
            "options": {"A": m.group(2), "B": m.group(3), "C": m.group(4), "D": m.group(5)},
            "answers": [m.group(6)],
        })
    return qs


def main():
    driver = get_driver()
    print("Neo4j:", neo4j_scalar(driver, "MATCH (n) RETURN count(n)"))

    if not test_llm_connection():
        print("LLM DOWN")
        return

    qs = load_questions(QUESTION_FILE)

    if not os.path.exists("output/faiss_index.pkl"):
        build_faiss_index()

    methods = {
        "faiss": retrieve_with_faiss,
        "graph": retrieve_with_graph,
        "hybrid": retrieve_with_hybrid,
    }

    for name, fn in methods.items():
        flags = []
        for q in qs:
            pred = safe_choice_letter(fn(q["question"], q["options"]))
            flags.append(pred in q["answers"])
            time.sleep(1)

        acc = round(100 * sum(flags) / len(flags), 2)
        print(name, acc)


if __name__ == "__main__":
    main()
