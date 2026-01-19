import os

UMLS_DIR = "KnowledgeGraph-info/kb_sources/umls"

def import_umls_relations(driver):
    mrrel_path = os.path.join(UMLS_DIR, "MRREL.RRF")
    if not os.path.exists(mrrel_path):
        raise FileNotFoundError(f"MRREL.RRF not found at: {mrrel_path}")

    with driver.session() as s:
        with open(mrrel_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                cols = line.split("|")
                if len(cols) < 6:
                    continue

                cui1 = cols[0].strip()
                rel = cols[3].strip()
                cui2 = cols[4].strip()

                if not cui1 or not cui2 or not rel:
                    continue

                s.run("""
                    MATCH (a:Concept {cui:$c1})
                    MATCH (b:Concept {cui:$c2})
                    MERGE (a)-[:UMLS_REL {type:$rel}]->(b)
                """, c1=cui1, c2=cui2, rel=rel)
