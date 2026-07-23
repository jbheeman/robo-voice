import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

# Use the exact chunk file that belt_v3_rag.py will search.
CHUNKS_FILE = BASE_DIR / "ucsc_complete_chunks.json"

# Folder containing the fine-tuned model files.
MODEL_PATH = (
    BASE_DIR
    / "ucsc_rag_embedder_training"
    / "ucsc_minilm_finetuned"
)

# New embeddings created by the fine-tuned model.
OUTPUT_FILE = BASE_DIR / "ucsc_finetuned_embeddings.npy"

BATCH_SIZE = 32


def load_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found:\n{path}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    # Supports either:
    # [chunk, chunk, ...]
    #
    # or:
    # {"chunks": [chunk, chunk, ...]}
    if isinstance(data, list):
        chunks = data

    elif isinstance(data, dict) and isinstance(data.get("chunks"), list):
        chunks = data["chunks"]

    else:
        raise ValueError(
            "The chunk JSON must be a list, or a dictionary "
            'containing a list under the key "chunks".'
        )

    if not chunks:
        raise ValueError("The chunk file is empty.")

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk {index} is not a JSON object."
            )

        text = chunk.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Chunk {index} has missing or empty text."
            )

    return chunks


def main() -> None:
    print("Checking files...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fine-tuned model folder not found:\n{MODEL_PATH}\n\n"
            "Make sure the downloaded ZIP was extracted and update "
            "MODEL_PATH if the folder is somewhere else."
        )

    print("Loading fine-tuned model...")

    model = SentenceTransformer(
        str(MODEL_PATH),
        local_files_only=True
    )

    print("Loading chunks...")

    chunks = load_chunks(CHUNKS_FILE)

    # Use only chunk text because that is how the model was fine-tuned.
    texts = [
        chunk["text"].strip()
        for chunk in chunks
    ]

    print(f"Loaded {len(texts)} chunks.")
    print("Creating fine-tuned document embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "The number of generated embeddings does not match "
            "the number of chunks."
        )

    np.save(
        OUTPUT_FILE,
        embeddings
    )

    print("\nEmbedding complete")
    print("-" * 50)
    print(f"Chunks:              {len(chunks)}")
    print(f"Embedding shape:     {embeddings.shape}")
    print(
        "Embedding dimension:",
        model.get_sentence_embedding_dimension()
    )
    print(f"Saved embeddings to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()