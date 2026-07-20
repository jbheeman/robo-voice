from sentence_transformers import SentenceTransformer
import json
import numpy as np


INPUT_CHUNKS_FILE = "ucsc_chunks.json"
OUTPUT_EMBEDDINGS_FILE = "ucsc_embeddings.npy"


model = SentenceTransformer("all-MiniLM-L6-v2")


with open(INPUT_CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunk_data = json.load(file)


# ucsc_chunks.json contains dictionaries like:
# {
#     "chunk_id": 0,
#     "text": "...",
#     "word_count": 200,
#     "start_word": 0,
#     "end_word": 199
# }

chunk_texts = [
    chunk["text"]
    for chunk in chunk_data
    if chunk["text"].strip()
]


embeddings = model.encode(
    chunk_texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


np.save(OUTPUT_EMBEDDINGS_FILE, embeddings)


print(f"Embedded {len(chunk_texts)} chunks.")
print(f"Embedding shape: {embeddings.shape}")
print(f"Saved embeddings to {OUTPUT_EMBEDDINGS_FILE}")