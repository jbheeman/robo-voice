import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCUMENT_FOLDER = Path("documents")

chunks = []
vectorizer = None
document_vectors = None


def load_documents():
    loaded_chunks = []

    for file_path in DOCUMENT_FOLDER.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        paragraphs = re.split(r"\n\s*\n", text)

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if paragraph:
                loaded_chunks.append({
                    "text": paragraph,
                    "source": file_path.name
                })

    return loaded_chunks


def build_index():
    global chunks
    global vectorizer
    global document_vectors

    chunks = load_documents()

    if not chunks:
        vectorizer = None
        document_vectors = None
        return

    document_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english"
    )

    document_vectors = vectorizer.fit_transform(document_texts)


def NOT_FINALIZED_rag_search(query, top_k=3, threshold=0.05):
    if vectorizer is None or document_vectors is None:
        return None

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:
        score = float(scores[index])

        if score >= threshold:
            results.append({
                "text": chunks[index]["text"],
                "source": chunks[index]["source"],
                "score": score
            })

    if not results:
        return None

    context_parts = []

    for result in results:
        context_parts.append(
            f"Source: {result['source']}\n"
            f"Information: {result['text']}\n"
            f"Similarity score: {result['score']:.3f}"
        )

    return "\n\n".join(context_parts)


# build_index()

def rag_search(text):
    return None