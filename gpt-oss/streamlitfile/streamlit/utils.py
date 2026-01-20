import os
import re
import time
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_driver():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")
    if not uri or not user or not pwd:
        raise RuntimeError("Missing NEO4J env vars")
    return GraphDatabase.driver(uri, auth=(user, pwd))


def _session(driver):
    return driver.session(database=os.getenv("NEO4J_DB"))


def chunk_text(text: str, max_words: int = 300):
    words = (text or "").split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def clean_triple(t: dict):
    if not isinstance(t, dict):
        return None
    if not all(k in t for k in ("head", "relation", "tail")):
        return None

    h = re.sub(r"\s+", " ", t["head"]).strip()
    r = re.sub(r"[^a-zA-Z0-9_ -]", "", t["relation"]).strip().lower()
    r = r.replace(" ", "_").replace("-", "_")
    ta = re.sub(r"\s+", " ", t["tail"]).strip()

    if not h or not r or not ta or h.lower() == ta.lower():
        return None

    return {"head": h, "relation": r, "tail": ta}


def ensure_domain_schema(driver):
    with _session(driver) as s:
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        s.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name_lc)")
        s.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.source)")


def upsert_chunk(driver, chunk_id, text, source):
    with _session(driver) as s:
        s.run("""
            MERGE (c:Chunk {id:$id})
            ON CREATE SET c.text=$text, c.source=$src
            ON MATCH  SET c.text=$text
        """, id=chunk_id, text=text, src=source)


def upsert_entity(driver, name, source):
    with _session(driver) as s:
        s.run("""
            MERGE (e:Entity {name:$name})
            ON CREATE SET e.name_lc=toLower($name), e.source=$src
        """, name=name, src=source)


def link_chunk_mentions(driver, chunk_id, entity_name):
    with _session(driver) as s:
        s.run("""
            MATCH (c:Chunk {id:$cid})
            MATCH (e:Entity {name:$ename})
            MERGE (c)-[:MENTIONS]->(e)
        """, cid=chunk_id, ename=entity_name)


def insert_triples_safe(driver, triples, source, chunk_id=None):
    for t in triples:
        for _ in range(3):
            try:
                upsert_entity(driver, t["head"], source)
                upsert_entity(driver, t["tail"], source)

                with _session(driver) as s:
                    s.run("""
                        MATCH (h:Entity {name:$h})
                        MATCH (t:Entity {name:$t})
                        MERGE (h)-[r:REL {type:$r}]->(t)
                        ON CREATE SET r.source=$src
                    """, h=t["head"], t=t["tail"], r=t["relation"], src=source)

                if chunk_id:
                    link_chunk_mentions(driver, chunk_id, t["head"])
                    link_chunk_mentions(driver, chunk_id, t["tail"])
                break
            except Exception:
                time.sleep(1)