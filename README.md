# Thyroid RAG Evaluation

RAG evaluation on thyroid cancer articles using MedGEMMA and (planned) DeepSeek-style models with FAISS, Hybrid retrieval, and Neo4j Knowledge Graph (KG).

This repository contains a full pipeline to:
- Build a *knowledge graph* from thyroid cancer review articles.
- Build a *FAISS vector store* over the same corpus.
- Run *multiple RAG strategies* (FAISS / Hybrid / Graph).
- Evaluate *multi-choice QA accuracy* on a curated thyroid MCQ set.

---


# 1. Project Overview

The goal of this project is to study how different RAG configurations perform on **thyroid cancer–related multiple-choice questions**.

Current status:

- **LLM backend:** `medgemma-27b-it` (via SAIA API).
- **RAG modes:**
  - `faiss` – pure vector search.
  - `hybrid` – keyword-augmented FAISS with a small LLM reranker.
  - `graph` – uses a Neo4j knowledge graph as an additional retrieval signal.
-  **Evaluation:** per-question predictions and global accuracy, saved as CSV.
- **Future Planned models:**  
  - `deepseek-r1-distill-llama-70b`  
  - other SAIA models such as Qwen3 (to be added in a similar way).

All results are currently produced for MedGEMMA and saved under:

```text
results_saia/medgemma-27b-it/


----

# 2. Folder structure
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
│   ├── evaluate_mcq.py             # (legacy) simple evaluation
│   └── evaluate_mcq_with_rag.py    # Main evaluation script for all RAG modes
│
├── Dockerfile                      # Container to run the whole pipeline
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment (optional)
├── .gitignore
└── .dockerignore

----
# 3. Environment and Dependencies

Can run the project either:
	•	Locally using Conda / venv, or
	•	Inside Docker (I used)

3.1. Python environment (local):
conda create -n thyroid_env python=3.11
conda activate thyroid_env

pip install -r requirements.txt

3.2. Required environment variables (streamlit/.env):

# --- Neo4j Database (Aura) ---
NEO4J_URI=neo4j+s://<your-db-id>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-neo4j-password>

# --- SAIA LLM API ---
SAIA_API_KEY=<your-saia-api-key>
SAIA_API_BASE=https://chat-ai.academiccloud.de/v1

"All LLM calls in this project are done via the OpenAI-compatible SAIA endpoint."
-----

# 4. Docker Usage
The project is designed to run inside Docker.

4.1. Build the image (From the project root):
**docker build -t test-kg**

4.2. Run a container( mount data, question, and streamlit so we can see outputs on the host):
docker run -it --name thyroid_container \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/question:/app/question \
    -v $(pwd)/streamlit:/app/streamlit \
    test-kg bash

**Inside the container, the project root is /app.
-----

# 5. Knowledge Graph Pipeline (Neo4j)
The KG is built from all .txt files in data/ using medgemma-27b-it as a triple extractor.

5.1. How triples are extracted
File: streamlit/main.py
	1.	Texts are cleaned and normalized:
	•	remove citations, figure captions, etc.
	•	normalize medical synonyms (e.g., “carcinoma” → “cancer”).
	2.	Text is split into overlapping chunks with RecursiveCharacterTextSplitter.
	3.	For each chunk, an LLM prompt asks MedGEMMA to output only JSON of the form:
[
  { "head": "thyroid cancer", "relation": "treated_with", "tail": "radioiodine therapy" },
  { "head": "papillary carcinoma", "relation": "is_a", "tail": "thyroid cancer" }
]

	4.	The JSON is parsed, and triples are inserted into Neo4j:
MERGE (h:Entity {name:$h})
MERGE (t:Entity {name:$t})
MERGE (h)-[r:REL {type:$r}]->(t)

5.2. Build the KG
Inside the running container:
"python3 streamlit/main.py --input_folder data"

This will:
	•	Process all text files in data/,
	•	Print how many chunks were used, and
	•	Insert several thousand triples into Neo4j.

You can inspect the graph visually in Neo4j Aura.

----
# 6. FAISS RAG Pipeline
File: streamlit/rag_faiss.py

6.1. Embedding model
We use a sentence-transformer:
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

6.2. Building the FAISS index
python3 streamlit/rag_faiss.py
This script:
	1.	Reads all .txt files in data/.
	2.	Splits them into small, overlapping chunks (chunk_text from utils.py).
	3.	Embeds all chunks and creates a FAISS index.
	4.	Saves the index to:
output/faiss_index.pkl

----
# 7. Hybrid RAG
File: streamlit/rag_hybrid.py
The hybrid mode combines:
	1.	Keyword extraction from the question (extract_keywords in utils.py),
	2.	Semantic search with FAISS (k=8),
	3.	A small LLM reranker that selects the most relevant few chunks,
	4.	Final QA using the same MedGEMMA-based letter-only answering logic.

This mode is meant to be more robust on noisy or ambiguous questions.

-----

# 8. Graph RAG
File: streamlit/rag_graph.py

The graph-based RAG queries Neo4j to retrieve concise text contexts built from:
	•	node names (a.name, b.name), and
	•	relationship types (r.type),

and then feeds those contexts into the same QA prompt as FAISS.

This allows the evaluation of knowledge-graph–aware retrieval vs. pure vector search.

-----
# 9. LLM Wrapper (MedGEMMA)

File: streamlit/llm_df.py

This module wraps the OpenAI-compatible API and forces the model to answer with a single letter A/B/C/D.

Key design choices:
	•	A strong system prompt:
	•	“You are a medical QA assistant.”
	•	“Always answer with a single letter: A, B, C, or D.”
	•	“No explanations. If unsure, choose the most likely option.”
	•	A robust regex (extract_letter) that understands:
	•	A, B, C, D
	•	“the answer is C”
	•	“Option B is correct”
	•	A second correction step if the first answer is messy:
	•	Ask again: “Extract the correct option letter ONLY (A/B/C/D).”

This makes it easier to evaluate multiple models in a unified way.


-----
# 10. MCQ Evaluation with RAG

File: streamlit/evaluate_mcq_with_rag.py

10.1. Question format

The file question/thyroid_questions.txt follows this format:
1) What is the most common type of differentiated thyroid cancer?
A) ...
B) ...
C) ...
D) ...
Answer: C

Questions are loaded by load_questions() in utils.py.

10.2. Running the evaluation (MedGEMMA)

Inside the container:
python3 streamlit/evaluate_mcq_with_rag.py

The script:
	1.	Loads all questions from thyroid_questions.txt.
	2.	For each RAG mode (faiss, hybrid, graph):
	•	Retrieves context,
	•	Builds a QA prompt,
	•	Calls chat_with_llm (MedGEMMA),
	•	Extracts the predicted letter,
	•	Compares it with the ground truth,
	•	Logs per-question correctness.
	3.	Prints accuracy in the terminal (for debugging), and
	4.	Saves detailed CSV results to:
results_saia/medgemma-27b-it/faiss_YYYYMMDD_HHMMSS.csv
results_saia/medgemma-27b-it/hybrid_YYYYMMDD_HHMMSS.csv
results_saia/medgemma-27b-it/graph_YYYYMMDD_HHMMSS.csv
results_saia/medgemma-27b-it/SUMMARY_YYYYMMDD_HHMMSS.csv

The summary file reports overall accuracy per RAG mode (e.g., ~89.8% for FAISS on the current setup).

-----
# 11. Planned Extensions
	•	DeepSeek R1 Distill LLaMA-70B
Add a second model via SAIA with a slightly different prompt and output parsing, to compare:
	•	MedGEMMA vs DeepSeek on the same RAG settings.
	•	Additional RAG ablations
	•	Change number of retrieved chunks (k),
	•	Compare “context length vs. accuracy,”
	•	Test hybrid vs graph on new datasets.
	•	Better KG extraction
	•	Filter out low-quality triples,
	•	Reduce LLM cost by sampling fewer chunks.

### Knowledge Graph (Neo4j Aura)

Below is a sample visualization of the extracted medical Knowledge Graph:
<img width="1470" height="956" alt="Screenshot 2025-11-28 at 10 33 45" src="https://github.com/user-attachments/assets/14067007-bad6-4d43-9b6b-41438ac1a1b3" />

### Evaluation Results (MedGEMMA)

| RAG Method | Accuracy | File |
|-----------|----------|------|
| FAISS     | 89.8%    | 
| Hybrid    | 87.1%    |
| Graph     | 76.4%    | 







