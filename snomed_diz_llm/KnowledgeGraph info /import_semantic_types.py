# import_semantic_types.py
import os

SEM_DIR = "KnowledgeGraph-info/kb_sources/semantic"

def import_semantic_types(driver):
    sem_path = os.path.join(SEM_DIR, "SemGroups.txt")

    if not os.path.exists(sem_path):
        raise FileNotFoundError("SemGroups.txt not found")

    print("📥 Importing Semantic Groups...")

    with driver.session() as session:
        with open(sem_path, "r", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 4:
                    continue

                cui, stype = parts[0], parts[3]

                session.run(
                    """
                    MATCH (c:Concept {cui:$cui})
                    SET c.semantic_type = $stype
                    """,
                    cui=cui,
                    stype=stype
                )

    print("✅ Semantic groups imported.")
