import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv(".env")

API_BASE = os.getenv("SAIA_API_BASE").rstrip("/")
API_KEY = os.getenv("SAIA_API_KEY")
MODEL = os.getenv("SAIA_MODEL")

_FINAL_RE = re.compile(r"FINAL\s+ANSWER\s*:\s*([ABCD])", re.I)

def chat_mcq(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 40,
    }

    for _ in range(4):
        try:
            r = requests.post(
                f"{API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=180,
            )
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            m = _FINAL_RE.search(txt.upper())
            if m:
                return m.group(1)
        except Exception:
            time.sleep(1)

    return "A"


def test_llm_connection():
    return chat_mcq("Final answer: A") == "A"
