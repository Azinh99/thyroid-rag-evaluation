# main.py
import os, json, re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm_df import call_llm

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

def extract_triples(chunk):
    prompt = f"""
Extract medical knowledge triples.
Return ONLY valid JSON list:
[{{"head":"...","relation":"...","tail":"..."}}]

TEXT:
{chunk}
"""
    raw = call_llm(prompt, max_tokens=900, temperature=0.1)
    try:
        return json.loads(raw)
    except:
        return []


def create_graph_from_files(folder="data"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue

        with open(os.path.join(folder, fname), encoding="utf-8", errors="ignore") as f:
            chunks = splitter.split_text(f.read())

        for ch in chunks:
            triples = extract_triples(ch)
            with driver.session(database=os.getenv("NEO4J_DB")) as s:
                for t in triples:
                    s.run(
                        """
                        MERGE (h:Entity {name:$h})
                        MERGE (t:Entity {name:$t})
                        MERGE (h)-[:REL {type:$r}]->(t)
                        """,
                        h=t["head"], r=t["relation"], t=t["tail"]
                    )

if __name__ == "__main__":
    create_graph_from_files("data")
     
