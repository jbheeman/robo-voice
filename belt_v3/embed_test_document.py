from sentence_transformers import SentenceTransformer
import json
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")

with open("documents/ucsc_mini_text.txt", "r", encoding="utf-8") as file:
    document = file.read()

chunks = [
    chunk.strip()
    for chunk in document.split("---CHUNK---")
    if chunk.strip()
]

embeddings = model.encode(
    chunks,
    normalize_embeddings=True
)

np.save("ucsc_test_embeddings.npy", embeddings)

with open("ucsc_test_chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=2)

print(f"Embedded {len(chunks)} chunks.")
print(embeddings.shape)