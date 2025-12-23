import os
import argparse

from llm_df import extract_kg
from utils import (
    get_driver,
    chunk_text,
    clean_triple,
    insert_triples_safe
)

# ---------- SAFE COPY (NO DEADLOCK) ----------
def safe_copy_to_tmp(src_path):
    """
    Copy file from Docker-mounted volume to /tmp using
    manual chunked read/write to avoid macOS deadlock.
    """
    filename = os.path.basename(src_path)
    dst_path = os.path.join("/tmp", filename)

    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)  # 1MB buffer
            if not chunk:
                break
            dst.write(chunk)

    return dst_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="data")
    args = parser.parse_args()

    driver = get_driver()

    for filename in sorted(os.listdir(args.input_folder)):
        if not filename.endswith(".txt"):
            continue

        src_path = os.path.join(args.input_folder, filename)
        print(f"\n📄 Processing: {filename}")

        # 🔑 CRITICAL FIX: copy to /tmp first
        tmp_path = safe_copy_to_tmp(src_path)

        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"Chunks: {len(chunks)}")

        all_triples = []

        for i, chunk in enumerate(chunks, 1):
            print(f"  chunk {i}/{len(chunks)}")

            triples = extract_kg(chunk)
            if not triples:
                continue

            for t in triples:
                clean = clean_triple(t)
                if clean:
                    all_triples.append(clean)

        print(f"🔵 Inserting triples for: {filename}")
        insert_triples_safe(driver, all_triples, filename)


if __name__ == "__main__":
    main()