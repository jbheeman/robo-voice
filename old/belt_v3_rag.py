import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = np.load(
    BASE_DIR / "ucsc_embeddings.npy"
)

with open(
    BASE_DIR / "ucsc_chunks.json",
    "r",
    encoding="utf-8"
) as file:
    document_chunks = json.load(file)


# Extract the text from each chunk dictionary.
document_texts = [
    chunk["text"]
    for chunk in document_chunks
]


# Make sure the chunks and embeddings still line up.
if len(document_texts) != len(document_embeddings):
    raise ValueError(
        "The number of JSON chunks does not match "
        "the number of saved embeddings."
    )


# Create TF-IDF representations of the same document chunks.
tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

document_tfidf = tfidf_vectorizer.fit_transform(document_texts)


def rag_search(
    query: str,
    top_k: int = 3,
    min_score: float = 0.2,
    embedding_weight: float = 0.7,
    tfidf_weight: float = 0.3
) -> list[dict]:

    if not query.strip():
        return []

    if not np.isclose(embedding_weight + tfidf_weight, 1.0):
        raise ValueError(
            "embedding_weight and tfidf_weight must add up to 1."
        )

    # Semantic embedding search
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    embedding_scores = document_embeddings @ query_embedding

    # TF-IDF keyword search
    query_tfidf = tfidf_vectorizer.transform([query])

    tfidf_scores = cosine_similarity(
        query_tfidf,
        document_tfidf
    ).flatten()

    # Combine semantic and keyword scores
    hybrid_scores = (
        embedding_weight * embedding_scores
        + tfidf_weight * tfidf_scores
    )

    # Rank every chunk from highest score to lowest score
    ranked_indices = np.argsort(hybrid_scores)[::-1]

    results = []

    for index in ranked_indices:
        score = float(hybrid_scores[index])

        # Stop once scores fall below the threshold
        if score < min_score:
            break

        results.append({
            "chunk": document_chunks[index]["text"],
            "chunk_id": document_chunks[index]["chunk_id"],
            "word_count": document_chunks[index]["word_count"],
            "start_word": document_chunks[index]["start_word"],
            "end_word": document_chunks[index]["end_word"],
            "score": score,
            "embedding_score": float(embedding_scores[index]),
            "tfidf_score": float(tfidf_scores[index])
        })

        # Return no more than top_k results
        if len(results) >= top_k:
            break

    return results