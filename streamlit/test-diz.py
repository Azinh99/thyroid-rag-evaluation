import os, requests
from dotenv import load_dotenv

load_dotenv("streamlit/.env")

API_BASE = os.getenv("DIZ_API_BASE", "https://ki-plattform.diz-ag.med.ovgu.de/api/")
API_KEY = os.getenv("DIZ_API_KEY")
MODEL = os.getenv("DIZ_MODEL", "llama3.2-vision:90b")

url = f"{API_BASE.rstrip('/')}/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Hello, are you responding from DIZ LLM?"}
    ],
    "max_tokens": 20
}

print("🔹 Sending test request to:", url)
r = requests.post(url, headers=headers, json=payload)
print("🔹 Status:", r.status_code)
print("🔹 Response:", r.text[:300])