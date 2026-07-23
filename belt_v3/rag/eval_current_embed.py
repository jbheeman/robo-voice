import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


BASE_DIR = Path(__file__).resolve().parent

RAG_FILE = BASE_DIR / "belt_v3_rag.py"
DATASET_FILE = BASE_DIR / "ucsc_question_chunk_pairs.json"
CHUNKS_FILE = BASE_DIR / "ucsc_complete_chunks.json"
CHUNKS_WITH_IDS_FILE = BASE_DIR / "ucsc_complete_chunks_with_ids.json"
EMBEDDINGS_FILE = BASE_DIR / "ucsc_complete_embeddings.npy"
OUTPUT_FILE = BASE_DIR / "eval_current_embed_results.json"

TOP_K = 10
PROGRESS_INTERVAL = 50
RECALL_CUTOFFS = (1, 3, 5, 10)


class EvaluationError(Exception):
    """Raised when the saved data cannot produce a valid evaluation."""


def require_files() -> None:
    paths = (
        RAG_FILE,
        DATASET_FILE,
        CHUNKS_FILE,
        CHUNKS_WITH_IDS_FILE,
        EMBEDDINGS_FILE,
    )
    missing = [str(path) for path in paths if not path.is_file()]

    if missing:
        raise EvaluationError("Missing required file(s): " + ", ".join(missing))


def load_json(path: Path, description: str):
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise EvaluationError(
            f"Invalid JSON in {description} file {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise EvaluationError(f"Could not read {description} file {path}: {exc}") from exc


def get_embedding_count() -> int:
    try:
        embeddings = np.load(EMBEDDINGS_FILE, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise EvaluationError(
            f"Could not load embeddings file {EMBEDDINGS_FILE}: {exc}"
        ) from exc

    if embeddings.ndim != 2 or embeddings.shape[0] < TOP_K:
        raise EvaluationError(
            f"Expected at least {TOP_K} saved embeddings in a 2D array; "
            f"found shape {embeddings.shape}."
        )

    return int(embeddings.shape[0])


def chunk_signature(chunk: dict, location: str) -> tuple[str, str, int]:
    if not isinstance(chunk, dict):
        raise EvaluationError(f"Invalid chunk at {location}: expected an object.")

    title = chunk.get("title")
    text = chunk.get("text")
    token_count = chunk.get("token_count")

    if not isinstance(title, str) or not isinstance(text, str):
        raise EvaluationError(
            f"Invalid chunk at {location}: 'title' and 'text' must be strings."
        )
    if not isinstance(token_count, int) or isinstance(token_count, bool):
        raise EvaluationError(
            f"Invalid chunk at {location}: 'token_count' must be an integer."
        )

    return title, text, token_count


def build_chunk_id_lookup(chunks_with_ids) -> dict[tuple[str, str, int], str]:
    if not isinstance(chunks_with_ids, list) or not chunks_with_ids:
        raise EvaluationError(
            f"{CHUNKS_WITH_IDS_FILE} must contain a non-empty JSON array."
        )

    id_by_signature = {}
    seen_ids = set()

    for index, chunk in enumerate(chunks_with_ids):
        location = f"{CHUNKS_WITH_IDS_FILE} entry {index}"
        signature = chunk_signature(chunk, location)
        chunk_id = chunk.get("chunk_id")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise EvaluationError(
                f"Invalid chunk at {location}: 'chunk_id' must be a non-empty string."
            )
        chunk_id = chunk_id.strip()

        if chunk_id in seen_ids:
            raise EvaluationError(f"Duplicate chunk ID '{chunk_id}'.")
        if signature in id_by_signature:
            other_id = id_by_signature[signature]
            raise EvaluationError(
                f"Chunks '{other_id}' and '{chunk_id}' have identical content, "
                "so retrieval results cannot be mapped to one real ID."
            )

        seen_ids.add(chunk_id)
        id_by_signature[signature] = chunk_id

    return id_by_signature


def select_search_corpus(current_chunks, chunks_with_ids, embedding_count: int):
    if not isinstance(current_chunks, list) or not current_chunks:
        raise EvaluationError(f"{CHUNKS_FILE} must contain a non-empty JSON array.")

    if len(current_chunks) == embedding_count:
        return CHUNKS_FILE, current_chunks

    if len(chunks_with_ids) == embedding_count:
        print(
            f"Warning: {CHUNKS_FILE.name} contains {len(current_chunks)} chunks, "
            f"but the saved embeddings contain {embedding_count} rows. Using "
            "the existing ID-bearing chunk file that matches the embeddings; "
            "no RAG files will be modified.",
            file=sys.stderr,
            flush=True,
        )
        return CHUNKS_WITH_IDS_FILE, chunks_with_ids

    raise EvaluationError(
        "Saved artifact counts do not match: "
        f"{CHUNKS_FILE.name} has {len(current_chunks)} chunks, "
        f"{CHUNKS_WITH_IDS_FILE.name} has {len(chunks_with_ids)} chunks, and "
        f"{EMBEDDINGS_FILE.name} has {embedding_count} rows."
    )


def get_available_chunk_ids(
    search_chunks,
    source_path: Path,
    id_by_signature: dict[tuple[str, str, int], str],
) -> set[str]:
    available_ids = set()

    for index, chunk in enumerate(search_chunks):
        signature = chunk_signature(chunk, f"{source_path} entry {index}")
        chunk_id = id_by_signature.get(signature)
        if chunk_id is None:
            raise EvaluationError(
                f"Chunk at {source_path} entry {index} has no matching real ID "
                f"in {CHUNKS_WITH_IDS_FILE}."
            )
        available_ids.add(chunk_id)

    return available_ids


def validate_dataset(dataset, available_ids: set[str]) -> list[dict]:
    if not isinstance(dataset, list) or not dataset:
        raise EvaluationError(f"{DATASET_FILE} must contain a non-empty JSON array.")

    entries = []
    unavailable = []

    for index, entry in enumerate(dataset):
        if not isinstance(entry, dict):
            raise EvaluationError(f"Invalid dataset entry {index}: expected an object.")

        question = entry.get("question")
        chunk_id = entry.get("chunk_id")
        if not isinstance(question, str) or not question.strip():
            raise EvaluationError(
                f"Invalid dataset entry {index}: 'question' must be a non-empty string."
            )
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise EvaluationError(
                f"Invalid dataset entry {index}: 'chunk_id' must be a non-empty string."
            )

        clean_entry = {
            "question": question.strip(),
            "chunk_id": chunk_id.strip(),
        }
        entries.append(clean_entry)

        if clean_entry["chunk_id"] not in available_ids:
            unavailable.append((index, clean_entry["chunk_id"]))

    if unavailable:
        preview = ", ".join(
            f"entry {index} ('{chunk_id}')"
            for index, chunk_id in unavailable[:10]
        )
        if len(unavailable) > 10:
            preview += f", and {len(unavailable) - 10} more"
        raise EvaluationError(
            f"{len(unavailable)} dataset entries reference unavailable chunk IDs: "
            f"{preview}."
        )

    return entries


def load_rag_search(search_chunks_file: Path):
    """Load the existing rag_search without changing any RAG file."""
    spec = importlib.util.spec_from_file_location(
        "belt_v3_rag_for_evaluation",
        RAG_FILE,
    )
    if spec is None or spec.loader is None:
        raise EvaluationError(f"Could not load Python module from {RAG_FILE}.")

    module = importlib.util.module_from_spec(spec)

    try:
        if search_chunks_file == CHUNKS_FILE:
            spec.loader.exec_module(module)
        else:
            original_open = Path.open

            def open_aligned_chunks(path, *args, **kwargs):
                if path == CHUNKS_FILE:
                    return original_open(search_chunks_file, *args, **kwargs)
                return original_open(path, *args, **kwargs)

            # belt_v3_rag.py hardcodes CHUNKS_FILE. Redirect only that read to
            # the saved corpus whose row count matches the saved embeddings.
            with patch.object(Path, "open", open_aligned_chunks):
                spec.loader.exec_module(module)
    except Exception as exc:
        raise EvaluationError(
            f"Could not initialize rag_search from {RAG_FILE}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    rag_search = getattr(module, "rag_search", None)
    if not callable(rag_search):
        raise EvaluationError(f"No callable rag_search was found in {RAG_FILE}.")
    return rag_search


def evaluate(dataset: list[dict], rag_search, id_by_signature) -> dict:
    details = []

    for question_number, entry in enumerate(dataset, start=1):
        try:
            retrieved = rag_search(entry["question"], top_k=TOP_K)
        except Exception as exc:
            raise EvaluationError(
                f"rag_search failed for dataset entry {question_number - 1}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(retrieved, list) or len(retrieved) != TOP_K:
            count = len(retrieved) if isinstance(retrieved, list) else "non-list"
            raise EvaluationError(
                f"rag_search returned {count} results instead of {TOP_K} "
                f"for question {question_number}."
            )

        retrieved_ids = []
        retrieved_scores = []
        for rank, result in enumerate(retrieved, start=1):
            if not isinstance(result, dict) or result.get("rank") != rank:
                raise EvaluationError(
                    f"Invalid retrieval result at rank {rank} for question "
                    f"{question_number}."
                )

            signature = chunk_signature(
                result,
                f"question {question_number}, retrieved rank {rank}",
            )
            chunk_id = id_by_signature.get(signature)
            if chunk_id is None:
                raise EvaluationError(
                    f"Retrieved rank {rank} for question {question_number} has "
                    "no matching real chunk ID."
                )

            score = result.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise EvaluationError(
                    f"Invalid score at rank {rank} for question {question_number}."
                )
            retrieved_ids.append(chunk_id)
            retrieved_scores.append(float(score))

        expected_id = entry["chunk_id"]
        correct_rank = (
            retrieved_ids.index(expected_id) + 1
            if expected_id in retrieved_ids
            else None
        )
        reciprocal_rank = 1.0 / correct_rank if correct_rank is not None else 0.0

        detail = {
            "question": entry["question"],
            "expected_chunk_id": expected_id,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "correct_rank": correct_rank,
            "reciprocal_rank": reciprocal_rank,
        }
        for cutoff in RECALL_CUTOFFS:
            detail[f"correct_at_{cutoff}"] = (
                correct_rank is not None and correct_rank <= cutoff
            )
        details.append(detail)

        if question_number % PROGRESS_INTERVAL == 0:
            print(f"Processed {question_number}/{len(dataset)} questions", flush=True)

    total = len(details)
    metrics = {
        f"recall_at_{cutoff}": sum(
            result[f"correct_at_{cutoff}"] for result in details
        ) / total
        for cutoff in RECALL_CUTOFFS
    }
    metrics["mrr"] = sum(result["reciprocal_rank"] for result in details) / total

    return {
        "total_questions": total,
        "metrics": metrics,
        "results": details,
    }


def save_results(evaluation: dict) -> None:
    try:
        with OUTPUT_FILE.open("w", encoding="utf-8") as file:
            json.dump(evaluation, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except OSError as exc:
        raise EvaluationError(f"Could not save results to {OUTPUT_FILE}: {exc}") from exc


def main() -> None:
    require_files()
    dataset = load_json(DATASET_FILE, "evaluation dataset")
    current_chunks = load_json(CHUNKS_FILE, "current chunks")
    chunks_with_ids = load_json(CHUNKS_WITH_IDS_FILE, "ID-bearing chunks")
    embedding_count = get_embedding_count()

    id_by_signature = build_chunk_id_lookup(chunks_with_ids)
    search_file, search_chunks = select_search_corpus(
        current_chunks,
        chunks_with_ids,
        embedding_count,
    )
    available_ids = get_available_chunk_ids(
        search_chunks,
        search_file,
        id_by_signature,
    )
    dataset = validate_dataset(dataset, available_ids)

    print(
        f"Evaluating {len(dataset)} questions against {embedding_count} "
        f"saved embeddings (top {TOP_K}).",
        flush=True,
    )
    rag_search = load_rag_search(search_file)
    evaluation = evaluate(dataset, rag_search, id_by_signature)
    save_results(evaluation)

    metrics = evaluation["metrics"]
    print("\nEvaluation complete")
    print(f"Total questions: {evaluation['total_questions']}")
    for cutoff in RECALL_CUTOFFS:
        print(f"Recall@{cutoff}: {metrics[f'recall_at_{cutoff}']:.6f}")
    print(f"MRR: {metrics['mrr']:.6f}")
    print(f"Detailed results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except EvaluationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
