# SNOMED + DiZ LLM Hybrid RAG

This folder contains a complete snapshot of the SNOMED + DiZ LLM experiment used for evaluating FAISS, Graph-RAG, and Hybrid-RAG pipelines on thyroid-related MCQs.


## Components

### 1. Knowledge Graph (`KnowledgeGraph-info/`)

This project uses a large-scale medical knowledge graph constructed from
SNOMED CT and UMLS resources. Due to licensing and file size constraints,
raw source files (e.g. MRCONSO.RRF, MRREL.RRF) are not included in this repository.

### Data Sources
- SNOMED CT (Transitive Closure files)
- UMLS Metathesaurus:
  - MRCONSO.RRF
  - MRREL.RRF
  - MRSTY.RRF
- UMLS Semantic Groups

### Pipeline Overview
1. Parse UMLS concepts and semantic types
2. Import SNOMED CT relations
3. Normalize entities and relations
4. Build a unified Neo4j knowledge graph

- Graph construction scripts:
  - `build_big_kg.py`
  - `import_umls_concepts.py`
  - `import_umls_relations.py`
  - `import_snomed_tc.py`

**Note**:  
Raw UMLS/SNOMED files are not included in the repository due to size and license constraints.

### 2.Streamlit
  - FAISS RAG
  - Graph RAG (Neo4j)
  - Hybrid RAG (vector + graph traversal)
  - MCQ evaluation scripts

### 3. Vector Retrieval
- FAISS index built from curated thyroid-related documents

### 4. Hybrid RAG Pipeline
- Vector retrieval → chunk selection
- LLM-based node importance filtering
- Subgraph extraction
- Evidence-based prompt construction

### 5. Evaluation
- MCQ-based evaluation
- Accuracy comparison: FAISS vs Graph vs Hybrid
- Results stored in `results/`
