import os

SNOMED_DIR = "KnowledgeGraph-info/kb_sources/snomed"

def import_snomed_tc(driver):
    # Auto-detect, then fallback to your known filename
    tc_file = None
    if os.path.isdir(SNOMED_DIR):
        for f in os.listdir(SNOMED_DIR):
            lf = f.lower()
            if lf.startswith("transitive") or "transitiveclosure" in lf:
                tc_file = os.path.join(SNOMED_DIR, f)
                break

    if not tc_file:
        fallback = os.path.join(SNOMED_DIR, "res2_TransitiveClosure_Snapshot.txt")
        if os.path.exists(fallback):
            tc_file = fallback

    if not tc_file:
        raise FileNotFoundError("No SNOMED transitive closure file found in KnowledgeGraph-info/kb_sources/snomed")

    with driver.session() as s:
        with open(tc_file, "r", encoding="utf-8", errors="ignore") as f:
            header = next(f, None)  # skip header if present
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue

                # Many TC files are: child \t parent
                child = parts[0].strip()
                parent = parts[1].strip()
                if not child or not parent:
                    continue

                s.run("""
                    MERGE (c:SNOMED {id:$child})
                    MERGE (p:SNOMED {id:$parent})
                    MERGE (c)-[:IS_A]->(p)
                """, child=child, parent=parent)
