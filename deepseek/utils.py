import os
import re
import time
from typing import List, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, Neo4jError
from dotenv import load_dotenv

load_dotenv(".env")

_DRIVER = None

_FINAL_RE = re.compile(r"FINAL\s+ANSWER\s*:\s*([ABCD])", re.I)
_LAST_RE = re.compile(r"\b([ABCD])\b", re.I)
_LUCENE_BAD = re.compile(r'[\+\-\!\(\)\{\}\[\]\^"~\*\?:\\/]')

# ---------------- Neo4j ----------------
def get_driver():
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
            max_connection_pool_size=10,
            connection_timeout=15,
        )
    return _DRIVER


def neo4j_scalar(driver, cypher: str, params: Optional[dict] = None):
    for _ in range(4):
        try:
            with driver.session() as s:
                rec = s.run(cypher, **(params or {})).single()
                return list(rec.values())[0] if rec else 0
        except (ServiceUnavailable, SessionExpired, Neo4jError):
            time.sleep(0.5)
    return 0


def neo4j_rows(driver, cypher: str, params: Optional[dict] = None):
    for _ in range(4):
        try:
            with driver.session() as s:
                return s.run(cypher, **(params or {})).data()
        except (ServiceUnavailable, SessionExpired, Neo4jError):
            time.sleep(0.5)
    return []


# ---------------- Text helpers ----------------
def chunk_text(text: str, max_words: int = 260) -> List[str]:
    words = (text or "").split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def safe_choice_letter(text: str) -> str:
    if not text:
        return "A"
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).upper()
    hits = _LAST_RE.findall(text)
    return hits[-1].upper() if hits else "A"


# ---------------- Lucene safe ----------------
def lucene_safe_query(text: str, max_terms: int = 12) -> str:
    if not text:
        return ""
    t = text.lower()
    t = _LUCENE_BAD.sub(" ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    terms = [x for x in t.split() if len(x) >= 4]
    return " ".join(terms[:max_terms])


# ---------------- Chunk → Concept ----------------
def upsert_chunks_and_link_concepts(
    driver,
    source: str,
    chunks: List[str],
    top_k_concepts: int = 6,
):
    neo4j_rows(
        driver,
        """
        CALL db.index.fulltext.createNodeIndex(
          'conceptNameFulltext',
          ['Concept'],
          ['name']
        )
        """
    )

    for idx, chunk in enumerate(chunks):
        cid = f"{source}::{idx}"

        neo4j_rows(
            driver,
            """
            MERGE (c:Chunk {id:$id})
            SET c.text=$text, c.source=$src
            """,
            {"id": cid, "text": chunk[:3000], "src": source},
        )

        q = lucene_safe_query(chunk)
        if not q:
            continue

        rows = neo4j_rows(
            driver,
            """
            CALL db.index.fulltext.queryNodes('conceptNameFulltext', $q)
            YIELD node, score
            RETURN node.cui AS cui, score
            ORDER BY score DESC
            LIMIT $k
            """,
            {"q": q, "k": int(top_k_concepts)},
        )

        for r in rows:
            neo4j_rows(
                driver,
                """
                MATCH (c:Chunk {id:$cid})
                MATCH (n:Concept {cui:$cui})
                MERGE (c)-[m:MENTIONS]->(n)
                SET m.score=$score
                """,
                {"cid": cid, "cui": r["cui"], "score": float(r["score"])},
            )
