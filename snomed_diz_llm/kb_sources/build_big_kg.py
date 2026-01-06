import os
import csv
from neo4j import GraphDatabase
from tqdm import tqdm
from utils import clean_triple

# ---------------------------------------------------
# Load Neo4j driver
# ---------------------------------------------------
def get_driver():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")

    if not uri or not user or not pwd:
        raise RuntimeError("❌ Neo4j credentials missing in .env")

    return GraphDatabase.driver(uri, auth=(user, pwd))


driver = get_driver()

# ---------------------------------------------------
# Helper: Safe insert triple
# ---------------------------------------------------
def insert_triple(h, r, t, source):

    # Normalization
    triple = clean_triple({"head": h, "relation": r, "tail": t})
    if not triple:
        return

    h = triple["head"]
    r = triple["relation"]
    t = triple["tail"]

    with driver.session() as s:
        s.run("""
            MERGE (a:Entity {name:$h})
            MERGE (b:Entity {name:$t})
            MERGE (a)-[rel:REL {type:$r}]->(b)
            SET rel.source = $src
        """, h=h, t=t, r=r, src=source)


# ---------------------------------------------------
# 1) IMPORT UMLS — MRCONSO.RRF (concept names)
# ---------------------------------------------------
def import_umls_concepts():

    path = "kb_sources/umls/MRCONSO.RRF"
    if not os.path.exists(path):
        print("❌ MRCONSO.RRF not found")
        return

    print("\n📌 Importing UMLS Concepts (MRCONSO.RRF) ...")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, total=None):
            parts = line.split("|")
            cui = parts[0].strip()
            name = parts[14].strip()  # STR field

            if name:
                insert_triple(cui, "has_name", name, "UMLS")


# ---------------------------------------------------
# 2) IMPORT UMLS — MRREL.RRF (relations)
# ---------------------------------------------------
def import_umls_relations():

    path = "kb_sources/umls/MRREL.RRF"
    if not os.path.exists(path):
        print("❌ MRREL.RRF not found")
        return

    print("\n📌 Importing UMLS Relations (MRREL.RRF)...")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f):
            p = line.split("|")
            cui1 = p[0].strip()
            rel = p[3].strip().lower()
            cui2 = p[4].strip()

            if cui1 and cui2 and rel:
                insert_triple(cui1, rel, cui2, "UMLS")


# ---------------------------------------------------
# 3) IMPORT UMLS — MRSTY.RRF (semantic types)
# ---------------------------------------------------
def import_semantic_types():

    path = "kb_sources/umls/MRSTY.RRF"
    if not os.path.exists(path):
        print("❌ MRSTY.RRF not found")
        return

    print("\n📌 Importing Semantic Types (MRSTY.RRF)...")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f):
            p = line.split("|")
            cui = p[0].strip()
            sty = p[3].strip()  # semantic type code

            if cui and sty:
                insert_triple(cui, "semantic_type", sty, "UMLS")


# ---------------------------------------------------
# 4) IMPORT Semantic Groups (SemGroups.txt)
# ---------------------------------------------------
def import_semantic_groups():

    path = "kb_sources/semantic/SemGroups.txt"
    if not os.path.exists(path):
        print("❌ SemGroups.txt not found")
        return

    print("\n📌 Importing Semantic Groups ...")

    with open(path, "r") as f:
        for line in tqdm(f):
            if "|" not in line:
                continue

            group, tui, sty = line.strip().split("|")

            insert_triple(tui, "in_semantic_group", group, "SEMANTIC")


# ---------------------------------------------------
# 5) IMPORT SNOMED Transitive Closure
# ---------------------------------------------------
def import_snomed_tc():

    path = "kb_sources/snomed/res2_TransitiveClosure_Snapshot.txt"
    if not os.path.exists(path):
        print("❌ SNOMED Transitive Closure file not found")
        return

    print("\n📌 Importing SNOMED Transitive Closure ...")

    with open(path, "r") as f:
        next(f)  # skip header
        reader = csv.reader(f, delimiter="\t")

        for row in tqdm(reader):
            if len(row) < 2:
                continue

            child = row[0].strip()
            parent = row[1].strip()

            insert_triple(child, "is_a", parent, "SNOMED")


# ---------------------------------------------------
# MASTER BUILDER
# ---------------------------------------------------
def build_full_kg():

    print("\n🚀 Starting FULL KNOWLEDGE GRAPH BUILD ...")

    import_umls_concepts()
    import_umls_relations()
    import_semantic_types()
    import_semantic_groups()
    import_snomed_tc()

    print("\n✅ COMPLETE: UMLS + SNOMED + Semantic Groups graph built!")


if __name__ == "__main__":
    build_full_kg()
