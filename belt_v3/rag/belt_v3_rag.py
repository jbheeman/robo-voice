import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "ucsc_complete_chunks.json"
EMBEDDINGS_FILE = BASE_DIR / "ucsc_complete_embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


model = SentenceTransformer(MODEL_NAME)

with CHUNKS_FILE.open("r", encoding="utf-8") as file:
    document_chunks = json.load(file)

document_embeddings = np.load(EMBEDDINGS_FILE)


if len(document_chunks) != len(document_embeddings):
    raise ValueError(
        "The number of chunks does not match the number of embeddings."
    )


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
        A list of dictionaries ordered from most relevant to least relevant.
    """
    text = text.strip()

    if not text:
        return []

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    top_k = min(top_k, len(document_chunks))

    query_embedding = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # The saved document embeddings and query embedding are normalized,
    # so their dot products are cosine similarity scores.
    similarity_scores = document_embeddings @ query_embedding

    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results = []

    for rank, chunk_index in enumerate(top_indices, start=1):
        chunk = document_chunks[int(chunk_index)]

        results.append({
            "rank": rank,
            "score": float(similarity_scores[chunk_index]),
            "chunk_index": int(chunk_index),
            "title": chunk["title"],
            "text": chunk["text"],
            "token_count": chunk["token_count"]
        })

    return results


if __name__ == "__main__":
    query = input("Enter a question: ")

    results = rag_search(
        text=query,
        top_k=3
    )

    for result in results:
        print("\n" + "=" * 80)
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk index: {result['chunk_index']}")
        print(f"Title: {result['title']}")
        print(result["text"])