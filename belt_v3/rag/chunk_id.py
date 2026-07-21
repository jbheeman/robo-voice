import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

input_path = BASE_DIR / "ucsc_complete_chunks.json"
output_path = BASE_DIR / "ucsc_complete_chunks_with_ids.json"


with open(input_path, "r", encoding="utf-8-sig") as file:
    chunks = json.load(file)


for index, chunk in enumerate(chunks):
    chunk["chunk_id"] = f"chunk_{index:05d}"


with open(output_path, "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=2, ensure_ascii=False)