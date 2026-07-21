import json


INPUT_FILE = "ucsc_data.txt"
OUTPUT_FILE = "ucsc_chunks.json"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 40


def chunk_text(text, chunk_size=200, overlap=40):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    words = text.split()
    chunks = []

    step_size = chunk_size - overlap

    for start in range(0, len(words), step_size):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunks.append({
            "chunk_id": len(chunks),
            "text": " ".join(chunk_words),
            "word_count": len(chunk_words),
            "start_word": start,
            "end_word": end - 1
        })

        if end == len(words):
            break

    return chunks


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        document = file.read()

    chunks = chunk_text(
        document,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"Created {len(chunks)} chunks.")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()