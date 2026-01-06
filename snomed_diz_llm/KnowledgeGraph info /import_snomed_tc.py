# import_snomed_tc.py
import os

SNOMED_DIR = "KnowledgeGraph-info/kb_sources/snomed"

def import_snomed_tc(driver):
    # find the transitive closure file
    tc_file = None
    for f in os.listdir(SNOMED_DIR):
        if f.lower().startswith("transitive"):
            tc_file = os.path.join(SNOMED_DIR, f)

    if not tc_file:
        raise FileNotFoundError("No SNOMED Transitive Closure file found")

    print(f"📥 Importing SNOMED TC: {tc_file}")

    with driver.session() as session:
        with open(tc_file, "r", errors="ignore") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue

                parent = parts[0]
                child = parts[1]

                session.run(
                    """
                    MERGE (p:SNOMED {id:$p})
                    MERGE (c:SNOMED {id:$c})
                    MERGE (p)-[:IS_A]->(c)
                    """,
                    p=parent,
                    c=child
                )

    print("✅ SNOMED TC imported.")
