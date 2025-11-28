import os
import re
from neo4j import GraphDatabase


# ============================================================
# 🔹 Load Neo4j driver (used for graph inserts)
# ============================================================
def get_driver():
    """
    Creates and returns a Neo4j driver instance.
    Must ONLY be called inside create_graph_from_kg().
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not user or not password:
        print("❌ Missing Neo4j environment variables!")
        return None

    return GraphDatabase.driver(uri, auth=(user, password))


# ============================================================
# 🔹 Create Knowledge Graph in Neo4j from triples
# ============================================================
def create_graph_from_kg(driver, triples):
    """
    Inserts extracted triples into Neo4j Aura.
    triples is a list of dict objects:
    { "head": "...", "relation": "...", "tail": "..." }
    """

    if not triples:
        print("⚠ No triples provided. Nothing to insert.")
        return

    with driver.session() as session:
        for t in triples:
            try:
                h = t["head"].strip()
                r = t["relation"].strip()
                a = t["tail"].strip()

                session.run(
                    """
                    MERGE (h:Entity {name:$h})
                    MERGE (t:Entity {name:$t})
                    MERGE (h)-[r:REL {type:$r}]->(t)
                    """,
                    h=h, r=r, t=a
                )
            except Exception as e:
                print("❌ Error inserting triple:", t, "| Error:", e)

    print(f"✅ Inserted {len(triples)} triples into Neo4j.")


# ============================================================
# 🔹 Question Loader for MCQ files
# ============================================================
Q_PATTERN = re.compile(
    r"\s*\d+[\.\)]\s*(.*?)\n"
    r"\s*A\)\s*(.*?)\n"
    r"\s*B\)\s*(.*?)\n"
    r"\s*C\)\s*(.*?)\n"
    r"\s*D\)\s*(.*?)\n"
    r"\s*(?:Answer|Correct Answer)[: ]+\s*([A-D])",
    re.MULTILINE
)


def safe_open(path, mode="r", enc="utf-8", retries=5):
    """
    Safer file reader to avoid container IO issues.
    """
    for _ in range(retries):
        try:
            return open(path, mode, encoding=enc)
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"File access failed: {path}")


def load_questions(path):
    """
    Loads MCQ questions in the fixed A/B/C/D format.
    """
    with safe_open(path) as f:
        data = f.read()

    qs = []
    for m in Q_PATTERN.finditer(data):
        qs.append({
            "q": m.group(1).strip(),
            "opts": {
                "A": m.group(2).strip(),
                "B": m.group(3).strip(),
                "C": m.group(4).strip(),
                "D": m.group(5).strip(),
            },
            "ans": m.group(6).upper(),
        })
    return qs


# ============================================================
# 🔹 Chunking
# ============================================================
def chunk_text(text, size=300):
    """
    Splits text into stable word chunks.
    """
    words = text.split()
    out = []
    buf = []

    for w in words:
        buf.append(w)
        if len(buf) >= size:
            out.append(" ".join(buf))
            buf = []

    if buf:
        out.append(" ".join(buf))

    return out


# ============================================================
# 🔹 Simple keyword extractor (for Hybrid RAG only)
# ============================================================
def extract_keywords(text):
    """
    Extracts non-stopword medical-ish keywords (very light).
    """
    tokens = re.findall(r"[A-Za-z]{4,}", text.lower())
    tokens = [t for t in tokens if t not in ["this", "that", "with", "from", "into", "about"]]
    return list(set(tokens))[:20]