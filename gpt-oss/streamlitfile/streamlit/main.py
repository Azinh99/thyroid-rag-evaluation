import os
import argparse
from pathlib import Path

from llm_df import extract_kg
from utils import (
    get_driver,
    ensure_domain_schema,
    chunk_text,
    clean_triple,
    upsert_chunk,
    insert_triples_safe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default=str(PROJECT_ROOT / "data"))
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.exists() or not input_folder.is_dir():
        raise RuntimeError(f"Input folder not found: {input_folder}")

    driver = get_driver()
    ensure_domain_schema(driver)

    chunk_words = int(os.getenv("CHUNK_WORDS", "300"))

    for file_path in sorted(input_folder.glob("*.txt")):
        filename = file_path.name
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = chunk_text(text, max_words=chunk_words)

        for i, ch in enumerate(chunks):
            chunk_id = f"{filename}::chunk_{i:04d}"
            upsert_chunk(driver, chunk_id, ch, filename)

            triples_raw = extract_kg(ch)
            if not triples_raw:
                continue

            cleaned = []
            for t in triples_raw:
                ct = clean_triple(t)
                if ct:
                    cleaned.append(ct)

            if cleaned:
                insert_triples_safe(driver, cleaned, filename, chunk_id=chunk_id)

    driver.close()

if __name__ == "__main__":
    main()