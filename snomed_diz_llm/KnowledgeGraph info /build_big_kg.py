# build_big_kg.py
import os
from neo4j import GraphDatabase

from import_umls_concepts import import_umls_concepts
from import_umls_relations import import_umls_relations
from import_semantic_types import import_semantic_types
from import_snomed_tc import import_snomed_tc

from dotenv import load_dotenv

load_dotenv("streamlit/.env")

def get_driver():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")

    if not uri:
        raise RuntimeError("Neo4j ENV missing")

    return GraphDatabase.driver(uri, auth=(user, pwd))


def main():
    driver = get_driver()

    print("\n==== BUILDING KNOWLEDGE GRAPH ====\n")

    import_umls_concepts(driver)
    import_umls_relations(driver)
    import_semantic_types(driver)
    import_snomed_tc(driver)

    print("\n🎉 DONE! FULL KNOWLEDGE GRAPH BUILT.\n")


if __name__ == "__main__":
    main()
