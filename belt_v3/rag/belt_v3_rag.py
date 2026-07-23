import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "ucsc_complete_chunks.json"

# Embeddings created using the fine-tuned model.
EMBEDDINGS_FILE = BASE_DIR / "ucsc_finetuned_embeddings.npy"

# Entire folder containing the fine-tuned SentenceTransformer model.
MODEL_PATH = (
    BASE_DIR
    / "ucsc_rag_embedder_training"
    / "ucsc_minilm_finetuned"
)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Fine-tuned model folder not found:\n{MODEL_PATH}"
    )

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(
        f"Chunks file not found:\n{CHUNKS_FILE}"
    )

if not EMBEDDINGS_FILE.exists():
    raise FileNotFoundError(
        f"Fine-tuned embeddings file not found:\n{EMBEDDINGS_FILE}\n"
        "Run embed_finetuned_chunks.py first."
    )


print("Loading fine-tuned embedding model...")

model = SentenceTransformer(
    str(MODEL_PATH),
    local_files_only=True
)

print("Loading document chunks...")

with CHUNKS_FILE.open("r", encoding="utf-8") as file:
    document_chunks = json.load(file)

print("Loading fine-tuned document embeddings...")

document_embeddings = np.load(
    EMBEDDINGS_FILE
)


if len(document_chunks) != len(document_embeddings):
    raise ValueError(
        "The number of chunks does not match the number of embeddings.\n"
        f"Chunks: {len(document_chunks)}\n"
        f"Embeddings: {len(document_embeddings)}"
    )


expected_dimension = model.get_sentence_embedding_dimension()

if document_embeddings.ndim != 2:
    raise ValueError(
        "The embeddings file must contain a 2D NumPy array."
    )

if document_embeddings.shape[1] != expected_dimension:
    raise ValueError(
        "The embedding dimensions do not match.\n"
        f"Model dimension: {expected_dimension}\n"
        f"Saved embedding dimension: {document_embeddings.shape[1]}"
    )


print("Fine-tuned RAG model loaded successfully.")


def rag_search(
    text: str,
    top_k: int = 3
) -> list[dict]:
    """
    Retrieve the chunks most relevant to the user's text.

    Args:
        text:
            The user's search query.

        top_k:
            The maximum number of chunks to return.

    Returns:
        A list of dictionaries ordered from most relevant
        to least relevant.
    """
    text = text.strip()

    if not text:
        return []

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    top_k = min(
        top_k,
        len(document_chunks)
    )

    query_embedding = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Both query and document embeddings are normalized,
    # so dot product equals cosine similarity.
    similarity_scores = (
        document_embeddings
        @ query_embedding
    )

    top_indices = (
        np.argsort(similarity_scores)[::-1][:top_k]
    )

    results = []

    for rank, chunk_index in enumerate(
        top_indices,
        start=1
    ):
        chunk_index = int(chunk_index)
        chunk = document_chunks[chunk_index]

        result = {
            "rank": rank,
            "score": float(
                similarity_scores[chunk_index]
            ),
            "chunk_index": chunk_index,
            "title": chunk["title"],
            "text": chunk["text"],
            "token_count": chunk["token_count"]
        }

        # Include the real chunk ID when available.
        if "chunk_id" in chunk:
            result["chunk_id"] = chunk["chunk_id"]

        results.append(result)

    return results


if __name__ == "__main__":
    while True:
        query = input("Enter a question: ")

        results = rag_search(
            text=query,
            top_k=3
        )

        for result in results:
            print("\n" + "=" * 80)
            print(f"Rank: {result['rank']}")
            print(f"Score: {result['score']:.4f}")
            print(
                f"Chunk index: "
                f"{result['chunk_index']}"
            )

            if "chunk_id" in result:
                print(
                    f"Chunk ID: "
                    f"{result['chunk_id']}"
                )

            print(f"Title: {result['title']}")
            print(result["text"])