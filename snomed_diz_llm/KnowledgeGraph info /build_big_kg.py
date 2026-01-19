import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

from import_umls_concepts import import_umls_concepts
from import_umls_relations import import_umls_relations
from import_semantic_types import import_semantic_types
from import_snomed_tc import import_snomed_tc

# .env is in project_root
load_dotenv()

def get_driver():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")
    if not uri or not user or not pwd:
        raise RuntimeError(
            "Neo4j env vars missing (NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)."
        )
    return GraphDatabase.driver(uri, auth=(user, pwd))

def ensure_constraints_and_indexes(driver):
    with driver.session() as s:
        # ---------- CONSTRAINTS ----------
        s.run("""
        CREATE CONSTRAINT concept_cui IF NOT EXISTS
        FOR (c:Concept) REQUIRE c.cui IS UNIQUE
        """)

        s.run("""
        CREATE CONSTRAINT snomed_id IF NOT EXISTS
        FOR (s:SNOMED) REQUIRE s.id IS UNIQUE
        """)

        # ---------- PROPERTY INDEXES ----------
        # Reminder: do NOT create simple index on c.name
        # Fulltext index is used instead (see below)

        s.run("""
        CREATE INDEX concept_name_lc IF NOT EXISTS
        FOR (c:Concept) ON (c.name_lc)
        """)

        s.run("""
        CREATE INDEX concept_semantic IF NOT EXISTS
        FOR (c:Concept) ON (c.semantic_type)
        """)

        # ---------- FULLTEXT INDEX ----------
        # Used for robust Entity -> UMLS Concept mapping
        s.run("""
        CREATE FULLTEXT INDEX conceptNameFT IF NOT EXISTS
        FOR (c:Concept) ON EACH [c.name]
        """)

def main():
    driver = get_driver()
    ensure_constraints_and_indexes(driver)

    print("BUILD_BIG_KG: importing UMLS concepts...")
    import_umls_concepts(driver)

    print("BUILD_BIG_KG: importing UMLS relations...")
    import_umls_relations(driver)

    print("BUILD_BIG_KG: importing semantic types...")
    import_semantic_types(driver)

    print("BUILD_BIG_KG: importing SNOMED transitive closure...")
    import_snomed_tc(driver)

    print("BUILD_BIG_KG: done.")
    driver.close()

if __name__ == "__main__":
    main()
