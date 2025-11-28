import os
import json
import re
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

from utils import create_graph_from_kg



# 🔹 Load Environment
# ============================
env_path = "streamlit/.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path}")
else:
    print("⚠️ .env file not found!")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

# ============================
# 🔹 NEW: LLM for KG extraction
# ============================
from openai import OpenAI
SAIA_KEY = os.getenv("SAIA_API_KEY")
SAIA_BASE = os.getenv("SAIA_API_BASE")
client = OpenAI(api_key=SAIA_KEY, base_url=SAIA_BASE)

def llm_for_kg(prompt, model="medgemma-27b-it"):
    """
    A dedicated LLM call for KG extraction.
    No regex, no forced A/B/C/D, no short max tokens.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You extract structured information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ LLM KG Error:", e)
        return ""


# ============================
# 🔹 Step 1: Cleaning
# ============================
def clean_and_standardize_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\(.*?et al.*?\)", "", text)
    text = re.sub(r"Fig\s*\d+.*?(?=\s[A-Z])", "", text)
    text = text.strip()

    synonyms = {
        "carcinoma": "cancer",
        "neoplasm": "tumor",
        "malignancy": "cancer",
        "thyroidectomy": "surgery",
        "radiotherapy": "radioiodine therapy",
        "chemotherapy": "systemic therapy",
        "metastatic": "metastasis",
    }
    for k, v in synonyms.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
    return text


def filter_useless_chunks(chunks):
    keep_keywords = ["thyroid", "cancer", "tumor", "metastasis", "surgery", "radioiodine"]
    out = []
    for c in chunks:
        if len(c.split()) < 15:
            continue
        if not any(k in c.lower() for k in keep_keywords):
            continue
        out.append(c)
    return out


# ============================
# 🔹 Step 2: Extract triples
# ============================
def extract_kg_from_text(chunk):
    prompt = f"""
Extract **only** knowledge triples from the following medical text.

Rules:
- Output MUST be a JSON array: [ ... ]
- No explanation.
- No markdown.
- No text before/after JSON.
- Each item must be: {{
    "head": "...",
    "relation": "...",
    "tail": "..."
  }}

Text:
{chunk}
"""

    raw = llm_for_kg(prompt)

    # Try full JSON parse
    try:
        return json.loads(raw)
    except:
        pass

    # Try extracting JSON array substring
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    print("⚠️ Could not parse JSON, skipping.")
    return []


# ============================
# 🔹 Step 3: Process multiple files
# ============================
def create_kg_from_multiple_txt_files(txt_files):
    all_triples = []

    for txt_file in txt_files:
        print(f"📄 Processing {txt_file}")

        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        text = clean_and_standardize_text(text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80
        )
        chunks = splitter.split_text(text)
        chunks = filter_useless_chunks(chunks)

        print(f"🧹 {len(chunks)} chunks kept after cleaning.")

        for i, chunk in enumerate(chunks):
            print(f"🔎 Sending chunk {i+1}/{len(chunks)}...")
            triples = extract_kg_from_text(chunk)
            if triples:
                all_triples.extend(triples)

    print(f"✅ Total extracted triples: {len(all_triples)}")

    if all_triples:
        create_graph_from_kg(driver, all_triples)
    else:
        print("⚠ No triples extracted.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="data")
    args = parser.parse_args()

    folder = args.input_folder
    txt_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".txt")
    ]

    if not txt_files:
        print("⚠️ No .txt files found!")
        return

    print("✅ Connected to Neo4j.")
    create_kg_from_multiple_txt_files(txt_files)


if __name__ == "__main__":
    main()