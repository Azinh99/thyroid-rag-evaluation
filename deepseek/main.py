# main.py
import os
import argparse

from llm_df import extract_kg
from utils import (
    get_driver,
    chunk_text,
    clean_triple,
    insert_triples_safe,
)

# -------------------------------------------------
# Safe copy to /tmp to avoid macOS Docker deadlock
# -------------------------------------------------
def safe_copy_to_tmp(src_path: str) -> str:
    filename = os.path.basename(src_path)
    dst_path = os.path.join("/tmp", filename)

    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        while True:
            buf = src.read(1024 * 1024)
            if not buf:
                break
            dst.write(buf)

    return dst_path


def main():
    parser = argparse.ArgumentParser(description="Build SNOMED-based Knowledge Graph in Neo4j")
    parser.add_argument(
        "--input_folder",
        type=str,
        default="data",
        help="Folder containing medical text files (.txt)",
    )
    args = parser.parse_args()

    driver = get_driver()

    for filename in sorted(os.listdir(args.input_folder)):
        if not filename.endswith(".txt"):
            continue

        print(f"\nProcessing file: {filename}")
        src_path = os.path.join(args.input_folder, filename)

        # Critical fix for Docker + macOS
        tmp_path = safe_copy_to_tmp(src_path)

        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = chunk_text(text, max_words=250)
        print(f"Total chunks: {len(chunks)}")

        all_triples = []

        for i, chunk in enumerate(chunks, start=1):
            print(f"  Chunk {i}/{len(chunks)}")

            triples = extract_kg(chunk)
            if not triples:
                continue

            for t in triples:
                cleaned = clean_triple(t)
                if cleaned:
                    all_triples.append(cleaned)

        if all_triples:
            print(f"Inserting {len(all_triples)} triples into Neo4j")
            insert_triples_safe(driver, all_triples, source=filename)
        else:
            print("No valid triples extracted")

    driver.close()
    print("\nKnowledge Graph build completed.")


if __name__ == "__main__":
    main()
