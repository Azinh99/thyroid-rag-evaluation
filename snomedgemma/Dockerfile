FROM python:3.11-slim

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    curl wget git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------------------------
# Python dependencies
# -------------------------
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install FAISS
RUN pip install --no-cache-dir faiss-cpu

# dotenv support
RUN pip install python-dotenv

# -------------------------
# Copy project files
# -------------------------
COPY . .

# Do NOT copy .env (we will mount it)
# COPY .env /app/.env   <-- REMOVE THIS

# Knowledge sources folder
COPY kb_sources /app/kb_sources

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV OPENAI_TIMEOUT=120
ENV OPENAI_MAX_RETRIES=8

# Default execution
CMD ["python3", "streamlit/evaluate_mcq_with_rag.py"]