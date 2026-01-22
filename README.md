# Ontology-Aware Hybrid Graph RAG for Medical MCQ Evaluation

## Abstract
This repository presents a systematic evaluation of Retrieval-Augmented Generation (RAG) pipelines for medical multiple-choice question (MCQ) answering.  
We study and compare **Text-only RAG**, **Graph RAG**, and **Hybrid RAG** approaches under two settings:

1. **Without medical ontology** (purely text- and LLM-extracted relations)
2. **With SNOMED CT ontology integration**

The primary focus of this repository is a **llama3.2-vision:90b**, with additional experiments conducted using ** gemma3:27b** and ** gpt-oss:120b** models for comparison.  
All methods are evaluated using the same MCQ benchmark to ensure fair and reproducible comparison.

---

## Pipeline Overview

Our experimental pipeline consists of the following core components:

### Inputs
- **Medical documents** (PDF / text-based articles)
- **MCQ question file** containing:
  - Question text  
  - Four answer choices (A–D)  
  - Ground-truth answer  

---

## Step-by-Step Methodology

### 1. Document Processing
- Input documents are loaded and cleaned.
- Documents are split into semantically meaningful chunks.
- Chunking strategy is kept consistent across all experiments.

---

### 2. Retrieval Strategies

We evaluate three retrieval strategies:

#### 2.1 FAISS-based Text RAG
- Dense vector embeddings
- Similarity search via FAISS
- No explicit relational structure

#### 2.2 Graph RAG
- Knowledge graph constructed from:
- LLM-extracted entities and relations
- Retrieval performed over graph neighborhoods

#### 2.3 Hybrid RAG
- Combination of:
  - FAISS similarity retrieval
  - Graph-based relational retrieval
- Retrieved contexts are merged before generation

---

### 3. Ontology Integration (SNOMED CT)

To study the impact of structured medical knowledge, we extend the pipeline with **SNOMED CT**:

- Medical concepts are linked to SNOMED identifiers
- Graph relations are enriched using ontology relations
- The same FAISS, Graph, and Hybrid pipelines are re-evaluated with ontology support

This enables a controlled comparison:
- **With ontology vs. without ontology**
- Same data, same questions, same model

---

### 4. Language Models

#### Primary Model
- **llama3.2-vision:90b** (via local inference)

#### Additional Models
- **gpt-oss:120b**
- **gemma3:27b**

All models are evaluated using the **same retrieval outputs** to isolate the effect of:
- Retrieval strategy
- Ontology usage
- Model choice

---

## Evaluation Protocol

- Task: Medical Multiple-Choice Question Answering
- Output constraint: Model must answer with **A, B, C, or D**
- Metrics:
  - Accuracy
 

---

## Results (LLaMA – Main Focus)

### Without SNOMED Ontology

| Method | Accuracy (%) |
|------|--------------|
| FAISS (Text-only RAG) |90.91 |
| Graph RAG | 89.09 |
| Hybrid RAG |92.73|

---

### With SNOMED Ontology

| Method | Accuracy (%) |
|------|--------------|
| FAISS + SNOMED | 93.88 |
| Graph + SNOMED | 91.84 |
| Hybrid + SNOMED | **97.96** |

> **Observation:**  
> Ontology integration consistently improves performance, with the Hybrid Graph RAG achieving the highest accuracy.

---

## Cross-Model Comparison (Same Pipeline)

Using the same pipeline, we further evaluate:

- **gpt-oss:120b**
- **gemma3:27b**

Each model is tested:
- With SNOMED
- Without SNOMED

Results are stored in CSV format for reproducibility.

---

## Comparison with State of the Art

The evaluated baselines correspond to commonly used approaches in current Graph-RAG literature:

- **Text-only RAG** → Standard RAG baseline
- **LLM-extracted Graph RAG** → Graph-based retrieval without ontology
- **Hybrid RAG** → Combined text and graph retrieval
- **Ontology-aware Hybrid Graph RAG (Ours)** → Structured medical knowledge integration




# Folder structure
The repository is organized as follows:
test_kg/
├── data/                  # Thyroid cancer articles (.txt)
│   ├── article1.txt
│   ├── article2.txt
│   ├── ...
│   └── articles/          
│
├── question/
│   └── thyroid_questions.txt   # MCQ set: numbered questions + A/B/C/D + Answer
│
├── streamlit/
│   ├── .env                        # API keys and Neo4j credentials 
│   ├── main.py                     # Builds the Knowledge Graph in Neo4j
│   ├── llm_df.py                   # LLM wrapper for MedGEMMA (A/B/C/D output)
│   ├── rag_faiss.py                # FAISS-based RAG
│   ├── rag_hybrid.py               # Hybrid retrieval (keywords + FAISS + reranker)
│   ├── rag_graph.py                # Graph-based RAG using Neo4j
│   └── evaluate_mcq_with_rag.py    # Main evaluation script for all RAG modes
│
├── Dockerfile                      # Container to run the whole pipeline
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment (optional)
├── .gitignore
└── .dockerignore

