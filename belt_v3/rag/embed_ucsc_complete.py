import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "ucsc_complete_chunks.json"
EMBEDDINGS_FILE = BASE_DIR / "ucsc_complete_embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    texts_to_embed = [
        f"{chunk['title']}\n\n{chunk['text']}"
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts_to_embed,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    np.save(EMBEDDINGS_FILE, embeddings)

    print(f"Model device: {model.device}")
    print(f"Embedded {len(chunks)} chunks")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()