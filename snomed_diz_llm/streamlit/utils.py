import os
import re
import time
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------
# Neo4j
# --------------------------------------------------
def get_driver():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")
    if not uri or not user or not pwd:
        raise RuntimeError("Missing NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in .env")
    return GraphDatabase.driver(uri, auth=(user, pwd))


# --------------------------------------------------
# Text utils
# --------------------------------------------------
def chunk_text(text: str, max_words: int = 300):
    words = (text or "").split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _clean_space(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def sanitize_for_fulltext(text: str) -> str:
    """
    Remove Lucene special characters to avoid TokenMgrError
    """
    if not text:
        return ""
    text = re.sub(r'[+\-&|!(){}[\]^"~*?:\\/]', " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# Triple cleaning
# --------------------------------------------------
def clean_triple(t: dict):
    if not isinstance(t, dict):
        return None
    if not all(k in t for k in ["head", "relation", "tail"]):
        return None

    h = _clean_space(t["head"])
    r = _clean_space(t["relation"]).lower()
    ta = _clean_space(t["tail"])

    if not h or not r or not ta:
        return None
    if h.lower() == ta.lower():
        return None

    r = re.sub(r"[^a-z0-9_ -]", "", r).strip()
    r = r.replace(" ", "_").replace("-", "_")

    return {"head": h, "relation": r, "tail": ta}


# --------------------------------------------------
# Domain schema
# --------------------------------------------------
def ensure_domain_schema(driver):
    with driver.session() as s:
        s.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        s.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        s.run("CREATE INDEX entity_name_lc IF NOT EXISTS FOR (e:Entity) ON (e.name_lc)")
        s.run("CREATE INDEX chunk_source IF NOT EXISTS FOR (c:Chunk) ON (c.source)")


# --------------------------------------------------
# Upserts
# --------------------------------------------------
def upsert_chunk(driver, chunk_id: str, text: str, source: str):
    with driver.session() as s:
        s.run(
            """
            MERGE (c:Chunk {id:$id})
            ON CREATE SET c.text=$text, c.source=$src
            ON MATCH  SET c.text=$text
            """,
            id=chunk_id,
            text=text,
            src=source,
        )


def upsert_entity(driver, name: str, source: str):
    with driver.session() as s:
        s.run(
            """
            MERGE (e:Entity {name:$name})
            ON CREATE SET e.name_lc=toLower($name), e.source=$src
            ON MATCH  SET e.name_lc=toLower($name)
            """,
            name=name,
            src=source,
        )


def link_chunk_mentions(driver, chunk_id: str, entity_name: str):
    with driver.session() as s:
        s.run(
            """
            MATCH (c:Chunk {id:$cid})
            MATCH (e:Entity {name:$ename})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            cid=chunk_id,
            ename=entity_name,
        )


# --------------------------------------------------
# Entity -> Concept mapping (FULLTEXT)
# --------------------------------------------------
def map_entity_to_concept_fulltext(driver, entity_name: str, min_score: float = 0.6) -> bool:
    """
    Maps Entity to UMLS Concept using FULLTEXT index conceptNameFulltext
    """
    safe_q = sanitize_for_fulltext(entity_name)
    if not safe_q:
        return False

    with driver.session() as s:
        rows = s.run(
            """
            CALL db.index.fulltext.queryNodes('conceptNameFulltext', $q)
            YIELD node, score
            RETURN node.cui AS cui, score
            ORDER BY score DESC
            LIMIT 1
            """,
            q=safe_q,
        ).data()

        if not rows:
            return False

        cui = rows[0].get("cui")
        score = float(rows[0].get("score") or 0.0)

        if not cui or score < min_score:
            return False

        s.run(
            """
            MATCH (e:Entity {name:$ename})
            MATCH (c:Concept {cui:$cui})
            MERGE (e)-[:MAPS_TO]->(c)
            """,
            ename=entity_name,
            cui=cui,
        )
        return True


# --------------------------------------------------
# Insert triples
# --------------------------------------------------
def insert_triples_safe(driver, triples, source: str, chunk_id: str = None):
    """
    Inserts guideline triples and links them to:
    - Chunk
    - Concept (via FULLTEXT)
    """
    min_score = float(os.getenv("CONCEPT_FT_MIN_SCORE", "0.6"))

    for t in triples:
        head = t["head"]
        rel = t["relation"]
        tail = t["tail"]

        for attempt in range(3):
            try:
                upsert_entity(driver, head, source)
                upsert_entity(driver, tail, source)

                with driver.session() as s:
                    s.run(
                        """
                        MATCH (h:Entity {name:$h})
                        MATCH (t:Entity {name:$t})
                        MERGE (h)-[r:REL {type:$r}]->(t)
                        ON CREATE SET r.source=$src
                        """,
                        h=head,
                        t=tail,
                        r=rel,
                        src=source,
                    )

                if chunk_id:
                    link_chunk_mentions(driver, chunk_id, head)
                    link_chunk_mentions(driver, chunk_id, tail)

                map_entity_to_concept_fulltext(driver, head, min_score=min_score)
                map_entity_to_concept_fulltext(driver, tail, min_score=min_score)

                break

            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
            
