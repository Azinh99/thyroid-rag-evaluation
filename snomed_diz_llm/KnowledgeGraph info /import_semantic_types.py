import os

SEM_DIR = "KnowledgeGraph-info/kb_sources/semantic"

def import_semantic_types(driver):
    sem_path = os.path.join(SEM_DIR, "SemGroups.txt")
    if not os.path.exists(sem_path):
        raise FileNotFoundError(f"SemGroups.txt not found at: {sem_path}")

    # Expected format in your current codebase: parts[0]=CUI, parts[3]=semantic group/type
    with driver.session() as s:
        with open(sem_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 4:
                    continue
                cui = parts[0].strip()
                stype = parts[3].strip()
                if not cui or not stype:
                    continue

                s.run("""
                    MATCH (c:Concept {cui:$cui})
                    SET c.semantic_type = $stype
                """, cui=cui, stype=stype)
