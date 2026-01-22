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
- **Medical documents** (text-based articles)
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
- **ministral-3:14b**

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

## Results

### Results without SNOMED Ontology

| Model | FAISS (%) | Graph (%) | Hybrid (%) |
|------|-----------|-----------|------------|
| LLaMA (llama3.2-vision:90b) | 90.91 | 89.09 | 92.73 |
| GPT (gpt-oss:120b) | 93.88 | 87.76 | 91.84 |
| Ministral (ministral-3:14b) | 91.84 | 48.98 | 30.61 |
| Gemma (gemma3:27b) | 93.88 | 87.76 | 91.84 |

---

### Results with SNOMED Ontology

| Model | FAISS + SNOMED (%) | Graph + SNOMED (%) | Hybrid + SNOMED (%) |
|------|--------------------|--------------------|---------------------|
| LLaMA (llama3.2-vision:90b) | 93.88 | 91.84 | **97.96** |
| GPT (gpt-oss:120b) | 93.88 | 87.76 | **91.84** |
| Ministral (ministral-3:14b) | 91.84 | 89.8 | **91.84** |
| Gemma (gemma3:27b) | 86 | 82 | **92** |




> **Observation:**  
> Ontology integration consistently improves performance, with the Hybrid Graph RAG achieving the highest accuracy.

---

## Cross-Model Comparison (Same Pipeline)

Using the same pipeline, we further evaluate:

- **gpt-oss:120b**
- **gemma3:27b**
- **ministral-3:14b**

Each model is tested:
- With SNOMED
- Without SNOMED

Results are stored in CSV format for reproducibility.

---

## Comparison with State of the Art

Comparison — Medical-Graph-RAG
https://github.com/ImprintLab/Medical-Graph-RAG

Medical-Graph-RAG focuses on building large medical knowledge graphs and using them to support LLM reasoning.
Our work is more practical and evaluation-driven: we study how FAISS, graph-only, and hybrid retrieval actually affect MCQ accuracy.
Instead of scaling the graph, we focus on understanding when and why graph knowledge (with or without SNOMED) helps.

⸻
Comparison — NVIDIA NeMo Guardrails
https://docs.nvidia.com/nemo/guardrails/latest/user-guides/guardrails-library.html

NVIDIA NeMo Guardrails is mainly about controlling how an LLM behaves, especially for safety and format constraints in medical use.
Our pipeline focuses on what evidence the model uses by grounding answers in text and knowledge graphs.
Guardrails could be a complementary layer, but our contribution lies in retrieval and evidence-based reasoning.




