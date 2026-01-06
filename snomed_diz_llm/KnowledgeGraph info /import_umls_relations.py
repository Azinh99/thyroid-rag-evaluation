# import_umls_relations.py
import os
from neo4j import GraphDatabase

UMLS_DIR = "KnowledgeGraph-info/kb_sources/umls"

def import_umls_relations(driver):
    mrrel_path = os.path.join(UMLS_DIR, "MRREL.RRF")

    if not os.path.exists(mrrel_path):
        raise FileNotFoundError("MRREL.RRF not found")

    print("📥 Importing UMLS Relations (MRREL)...")

    with driver.session() as session:
        with open(mrrel_path, "r", errors="ignore") as f:
            for line in f:
                cols = line.split("|")
                cui1 = cols[0].strip()
                rel = cols[3].strip()
                cui2 = cols[4].strip()

                session.run(
                    """
                    MATCH (a:Concept {cui:$c1})
                    MATCH (b:Concept {cui:$c2})
                    MERGE (a)-[:UMLS_REL {type:$rel}]->(b)
                    """,
                    c1=cui1,
                    c2=cui2,
                    rel=rel
                )
    print("UMLS relations imported.")
