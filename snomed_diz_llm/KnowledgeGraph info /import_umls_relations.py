# import_umls_concepts.py
import os
from neo4j import GraphDatabase

UMLS_DIR = "KnowledgeGraph-info/kb_sources/umls"

def import_umls_concepts(driver):
    mrconso_path = os.path.join(UMLS_DIR, "MRCONSO.RRF")

    if not os.path.exists(mrconso_path):
        raise FileNotFoundError("MRCONSO.RRF not found")

    print("📥 Importing UMLS Concepts (MRCONSO)...")

    with driver.session() as session:
        with open(mrconso_path, "r", errors="ignore") as f:
            for line in f:
                cols = line.split("|")
                cui = cols[0].strip()
                term = cols[14].strip()

                session.run(
                    """
                    MERGE (c:Concept {cui:$cui})
                    ON CREATE SET c.name = $name
                    """,
                    cui=cui,
                    name=term
                )
    print("Done importing MRCONSO concepts.")
