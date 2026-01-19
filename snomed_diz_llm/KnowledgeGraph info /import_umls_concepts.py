import os

UMLS_DIR = "KnowledgeGraph-info/kb_sources/umls"

def import_umls_concepts(driver):
    mrconso_path = os.path.join(UMLS_DIR, "MRCONSO.RRF")
    if not os.path.exists(mrconso_path):
        raise FileNotFoundError(f"MRCONSO.RRF not found at: {mrconso_path}")

    # MRCONSO: CUI|...|SAB(col11)|...|CODE(col13)|STR(col14)|...
    with driver.session() as s:
        with open(mrconso_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                cols = line.split("|")
                if len(cols) < 15:
                    continue

                cui = cols[0].strip()
                sab = cols[11].strip()
                code = cols[13].strip()
                term = cols[14].strip()

                if not cui or not term:
                    continue

                s.run("""
                    MERGE (c:Concept {cui:$cui})
                    ON CREATE SET c.name=$name, c.name_lc=toLower($name)
                    ON MATCH  SET c.name=coalesce(c.name, $name),
                                  c.name_lc=coalesce(c.name_lc, toLower($name))
                """, cui=cui, name=term)

                # Link UMLS Concept to SNOMED conceptId when MRCONSO atom is SNOMEDCT
                if sab.upper().startswith("SNOMEDCT") and code:
                    s.run("""
                        MATCH (c:Concept {cui:$cui})
                        MERGE (sn:SNOMED {id:$sid})
                        MERGE (c)-[:HAS_SNOMED]->(sn)
                    """, cui=cui, sid=code)
