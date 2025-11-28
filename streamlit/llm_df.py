import os
import time
import re
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------------------
# Load environment variables
# -----------------------------------------
env_paths = ["streamlit/.env", "/app/streamlit/.env", ".env"]
for p in env_paths:
    if os.path.exists(p):
        load_dotenv(p)
        break

API_KEY = os.getenv("SAIA_API_KEY")
API_BASE = os.getenv("SAIA_API_BASE")

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# -----------------------------------------
# Ultra-robust regex for extracting A/B/C/D
# Handles:
#   "C"
#   "Answer is B"
#   "The correct option is D."
#   "Option: A"
# -----------------------------------------
LETTER_REGEX = re.compile(
    r"\b([A-D])\b|answer\s*is\s*([A-D])|option\s*([A-D])",
    re.IGNORECASE
)


def extract_letter(text: str):
    """Extract an answer letter (A–D) from messy model output."""
    if not text:
        return None

    matches = LETTER_REGEX.findall(text)
    if matches:
        for g1, g2, g3 in matches:
            for g in (g1, g2, g3):
                if g:
                    return g.upper()
    return None


# -----------------------------------------
# Core model call (sync, safe, retry logic)
# -----------------------------------------
def call_llm_sync(prompt: str, model_name="medgemma-27b-it", max_retries=4):
    """
    Stable synchronous call for MedGemma.
    max_tokens=50 → avoids overflow.
    temperature=0.0 → deterministic answers.
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a medical question-answering model. "
                            "Your output MUST be only one letter: A, B, C, or D. "
                            "Do not provide explanations or sentences. "
                            "If uncertain, choose the most likely option."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=50,
            )

            return resp.choices[0].message.content.strip()

        except Exception as e:
            print(f"⚠ LLM error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(1)

    print("❌ LLM call failed after retries.")
    return "N/A"


# -----------------------------------------
# High-level interface with fallback pass
# -----------------------------------------
def chat_with_llm(prompt: str, model_name="medgemma-27b-it"):
    """
    Runs model → extracts letter → if fail, re-asks model cleanly.
    """
    raw = call_llm_sync(prompt, model_name)

    letter = extract_letter(raw)
    if letter:
        return letter

    # fallback attempt
    fix_prompt = (
        f"Your previous answer was: {raw}\n"
        "Extract ONLY the final answer letter (A, B, C, or D)."
    )
    raw2 = call_llm_sync(fix_prompt, model_name)
    letter2 = extract_letter(raw2)

    return letter2 if letter2 else "N/A"


def test_llm_connection(model_name="medgemma-27b-it"):
    """Quick connectivity test."""
    try:
        out = call_llm_sync("A or B?", model_name)
        return out is not None
    except:
        return False