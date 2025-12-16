# SNOMED + DiZ LLM Hybrid RAG

This folder contains a complete snapshot of the SNOMED + DiZ LLM experiment used for evaluating FAISS, Graph-RAG, and Hybrid-RAG pipelines on thyroid-related MCQs.


## Components

### 1. Knowledge Graph (`KnowledgeGraph-info/`)
This folder contains scripts and metadata used to build the medical knowledge graph.

- Data sources:
  - SNOMED CT
  - UMLS (MRCONSO, MRREL, MRSTY)

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
