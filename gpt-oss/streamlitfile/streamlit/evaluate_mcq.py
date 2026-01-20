import os
import re
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from utils import load_questions
from rag_faiss import retrieve_with_faiss
from rag_hybrid import retrieve_with_hybrid
from rag_graph import retrieve_from_graph

from llm_df import LLMHandler, LLMConnectionTester

load_dotenv()

SAIA_KEY = os.getenv("SAIA_API_KEY")
SAIA_BASE = os.getenv("SAIA_API_BASE")

import openai
openai.api_key = SAIA_KEY
openai.base_url = SAIA_BASE

client = openai.AsyncOpenAI(
    api_key=SAIA_KEY,
    base_url=SAIA_BASE
)

llm_handler = LLMHandler(client)
llm_tester = LLMConnectionTester(client)

SAIA_MODELS = [
    "medgemma-27b-it",
    "deepseek-r1-distill-llama-70b",
    "qwen3-235b-a22b"
]


def run_rag_sync(rag_name, questions, model_name):

    preds = []

    for q in questions:

        question = q["q"]
        opts = q["opts"]

        if rag_name == "faiss":
            pred = retrieve_with_faiss(question, opts, model_name)

        elif rag_name == "hybrid":
            pred = retrieve_with_hybrid(question, opts, model_name)

        elif rag_name == "graph":
            pred = retrieve_from_graph(
                question=question,
                options=opts,
                model_name=model_name,
                top_k=10
            )

        else:
            pred = "N/A"

        preds.append(pred)

    correct = sum(1 for p, q in zip(preds, questions) if p == q["ans"])
    acc = round(100 * correct / len(questions), 2)

    return preds, acc



def save_results(model_name, rag_name, preds, questions):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = f"results_saia/{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame({
        "question": [q["q"] for q in questions],
        "pred": preds,
        "ans": [q["ans"] for q in questions]
    })

    out_path = f"{out_dir}/{rag_name}_{ts}.csv"
    df.to_csv(out_path, index=False)

    print(f"💾 Saved → {out_path}")

    return out_path



async def evaluate_all():

    print("Loading questions...")
    questions = load_questions("question/thyroid_questions.txt")
    print(f"✔ Loaded {len(questions)} questions")

    for model in SAIA_MODELS:
        print("\n=========================================")
        print(f" Running SAIA model → {model}")
        print("=========================================")

        ok = await llm_tester.test_connection(model)
        if not ok:
            print(f"❌ Model offline → skipping")
            continue

        rag_methods = ["faiss", "hybrid", "graph"]
        summary = []

        for rag in rag_methods:
            print(f"\n----- Model={model} | RAG={rag} -----")

            preds, acc = run_rag_sync(rag, questions, model)
            summary.append({"rag": rag, "acc": acc})

            save_results(model, rag, preds, questions)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_sum = pd.DataFrame(summary)
        df_sum.to_csv(f"results_saia/{model}/SUMMARY_{ts}.csv", index=False)

        print(f"\n📁 Summary saved → results_saia/{model}/SUMMARY_{ts}.csv")


if __name__ == "__main__":
    import asyncio
    asyncio.run(evaluate_all())