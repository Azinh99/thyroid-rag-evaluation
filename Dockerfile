FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl wget git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ----------------------------
# Install NLTK datasets LOCALLY
# ----------------------------
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data punkt
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data stopwords
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data wordnet

# Guarantee NLTK reads from this path
ENV NLTK_DATA="/usr/local/share/nltk_data"

# Copy project files
COPY . .

CMD ["python3", "streamlit/evaluate_mcq_with_rag.py"]