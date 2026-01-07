# llm_df.py
import os
import time
import re
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("DIZ_API_BASE")
API_KEY = os.getenv("DIZ_API_KEY")
MODEL = os.getenv("DIZ_MODEL")

if not API_BASE or not API_KEY or not MODEL:
    raise RuntimeError("Missing DIZ_API_BASE / DIZ_API_KEY / DIZ_MODEL")

URL = f"{API_BASE.rstrip('/')}/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

LETTER_REGEX = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def call_llm(prompt, max_tokens=60, temperature=0.0):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Answer ONLY with A, B, C, or D."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for _ in range(3):
        try:
            r = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            time.sleep(1)

    return "A"


def chat_with_llm(prompt):
    raw = call_llm(prompt)
    m = LETTER_REGEX.search(raw or "")
    return m.group(1).upper() if m else "A"


def test_llm_connection():
    try:
        out = call_llm("Say only A", max_tokens=5)
        return out is not None
    except:
        return False
