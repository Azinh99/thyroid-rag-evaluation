import os
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MODEL = os.getenv("DIZ_MODEL")
API_BASE = (os.getenv("DIZ_API_BASE") or "").rstrip("/") + "/"
API_KEY = os.getenv("DIZ_API_KEY")

def _post(payload: dict) -> dict:
    if not API_BASE or not API_KEY or not MODEL:
        raise RuntimeError("Missing DIZ_API_BASE / DIZ_API_KEY / DIZ_MODEL in .env")
    url = API_BASE + "v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.post(url, json=payload, headers=headers, timeout=180)
    r.raise_for_status()
    return r.json()

def chat_with_llm(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    for _ in range(3):
        try:
            out = _post(payload)
            return out["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(1)

    return "[ERROR] LLM unavailable"

def test_llm_connection() -> bool:
    try:
        out = chat_with_llm("Return YES if you can read this.")
        return "YES" in (out or "").upper()
    except Exception:
        return False

def extract_kg(text_chunk: str):
    prompt = f"""
You extract clinical knowledge graph triples from guideline text.

Return ONLY valid JSON (no markdown, no comments).
Each triple must be directly supported by the text.

Allowed relation examples:
treated_with, recommended_for, indicated_for, contraindicated_for,
requires, followed_by, associated_with, risk_factor_for, diagnosed_by.

Return format:
[
  {{"head":"...", "relation":"...", "tail":"..."}}
]

TEXT:
{text_chunk}
"""
    for _ in range(5):
        out = chat_with_llm(prompt)
        try:
            triples = json.loads(out)
            if isinstance(triples, list):
                return triples
        except Exception:
            time.sleep(1)
    return []
