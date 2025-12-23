import os
import time
import json
import requests
from dotenv import load_dotenv

load_dotenv(".env")

MODEL = os.getenv("DIZ_MODEL")
API_BASE = os.getenv("DIZ_API_BASE")
API_KEY = os.getenv("DIZ_API_KEY")


def _post(payload):
    url = API_BASE + "v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.post(url, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    return r.json()


def chat_with_llm(prompt):
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


def test_llm_connection():
    try:
        out = chat_with_llm("Return YES if you can read this.")
        return "YES" in out.upper()
    except:
        return False


def extract_kg(text_chunk):
    prompt = f"""
Extract knowledge graph triples.
Return ONLY JSON list:
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
        except:
            time.sleep(1)

    return []