import os
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(".env")


def get_driver():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")

    if not uri:
        raise RuntimeError("❌ Missing Neo4j URI in .env")

    return GraphDatabase.driver(uri, auth=(user, pwd))


def insert_triples_safe(driver, triples, source):

    for t in triples:
        head = t["head"]
        rel = t["relation"]
        tail = t["tail"]

        for attempt in range(3):
            try:
                with driver.session() as s:
                    s.run("""
                        MERGE (h:Entity {name:$h})
                        SET h.source = $src

                        MERGE (t:Entity {name:$t})
                        SET t.source = $src

                        MERGE (h)-[r:REL {type:$r}]->(t)
                        SET r.source = $src
                    """, h=head, t=tail, r=rel, src=source)

                break

            except Exception as e:
                print(f"⚠ Insert retry {attempt+1}/3:", e)
                time.sleep(1)


def clean_triple(t):
    if not isinstance(t, dict):
        return None

    if not all(k in t for k in ["head", "relation", "tail"]):
        return None

    return {
        "head": t["head"].strip(),
        "relation": t["relation"].strip().lower(),
        "tail": t["tail"].strip()
    }


def chunk_text(text, max_words=250):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]