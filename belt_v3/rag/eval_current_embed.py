"""Evaluate the original MiniLM model on the trained model's test split.

This uses the same corpus text, held-out questions, retrieval evaluator, metric
names, and cutoffs as train_embedder.py so the original and fine-tuned model
results are directly comparable.
"""

import json
import sys
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
)


BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "ucsc_complete_chunks_with_ids.json"
TEST_SPLIT_FILE = (
    BASE_DIR
    / "ucsc_rag_embedder_training"
    / "dataset_splits"
    / "test.json"
)
OUTPUT_FILE = BASE_DIR / "eval_current_embed_results.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EVALUATOR_NAME = "test"
EVAL_BATCH_SIZE = 32
METRIC_CUTOFFS = (1, 3, 5, 10)


class EvaluationError(Exception):
    """Raised when the saved data cannot produce a valid evaluation."""


def require_files() -> None:
    missing = [
        str(path)
        for path in (CHUNKS_FILE, TEST_SPLIT_FILE)
        if not path.is_file()
    ]
    if missing:
        raise EvaluationError("Missing required file(s): " + ", ".join(missing))


def load_json(path: Path, description: str) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise EvaluationError(
            f"Invalid JSON in {description} file {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise EvaluationError(
            f"Could not read {description} file {path}: {exc}"
        ) from exc


def build_corpus(data: Any) -> dict[str, str]:
    if not isinstance(data, list) or not data:
        raise EvaluationError(
            f"{CHUNKS_FILE} must contain a non-empty JSON array."
        )

    corpus: dict[str, str] = {}
    for index, chunk in enumerate(data):
        if not isinstance(chunk, dict):
            raise EvaluationError(
                f"Invalid chunk at {CHUNKS_FILE} entry {index}: "
                "expected an object."
            )

        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise EvaluationError(
                f"Invalid chunk at {CHUNKS_FILE} entry {index}: "
                "'chunk_id' must be a non-empty string."
            )
        if not isinstance(text, str) or not text.strip():
            raise EvaluationError(
                f"Invalid chunk at {CHUNKS_FILE} entry {index}: "
                "'text' must be a non-empty string."
            )

        chunk_id = chunk_id.strip()
        if chunk_id in corpus:
            raise EvaluationError(f"Duplicate chunk ID '{chunk_id}'.")

        # train_embedder.py evaluates document text without prepending titles.
        corpus[chunk_id] = text.strip()

    return corpus


def build_test_queries(
    data: Any,
    corpus: dict[str, str],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    if not isinstance(data, list) or not data:
        raise EvaluationError(
            f"{TEST_SPLIT_FILE} must contain a non-empty JSON array."
        )

    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}

    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise EvaluationError(
                f"Invalid test entry {index}: expected an object."
            )

        question = entry.get("question")
        chunk_id = entry.get("chunk_id")
        if not isinstance(question, str) or not question.strip():
            raise EvaluationError(
                f"Invalid test entry {index}: "
                "'question' must be a non-empty string."
            )
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise EvaluationError(
                f"Invalid test entry {index}: "
                "'chunk_id' must be a non-empty string."
            )

        chunk_id = chunk_id.strip()
        if chunk_id not in corpus:
            raise EvaluationError(
                f"Test entry {index} references unavailable chunk ID "
                f"'{chunk_id}'."
            )

        # Match the query IDs created by train_embedder.py.
        query_id = f"{EVALUATOR_NAME}_question_{index:05d}"
        queries[query_id] = question.strip()
        relevant_docs[query_id] = {chunk_id}

    return queries, relevant_docs


def make_evaluator(
    queries: dict[str, str],
    corpus: dict[str, str],
    relevant_docs: dict[str, set[str]],
) -> InformationRetrievalEvaluator:
    """Create the same evaluator configuration used by train_embedder.py."""
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=[10],
        accuracy_at_k=list(METRIC_CUTOFFS),
        precision_recall_at_k=list(METRIC_CUTOFFS),
        ndcg_at_k=[10],
        map_at_k=[10],
        show_progress_bar=True,
        batch_size=EVAL_BATCH_SIZE,
        name=EVALUATOR_NAME,
        write_csv=False,
    )


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def save_results(
    metrics: dict[str, Any],
    corpus_count: int,
    test_question_count: int,
) -> None:
    results = {
        "base_model": MODEL_NAME,
        "dataset_split": str(TEST_SPLIT_FILE.relative_to(BASE_DIR)),
        "counts": {
            "corpus_chunks": corpus_count,
            "test_questions": test_question_count,
        },
        "test_metrics": make_json_serializable(metrics),
    }

    try:
        with OUTPUT_FILE.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except OSError as exc:
        raise EvaluationError(
            f"Could not save evaluation results to {OUTPUT_FILE}: {exc}"
        ) from exc


def print_metrics(metrics: dict[str, Any]) -> None:
    print("\nOriginal model evaluation complete")
    print(f"Model:          {MODEL_NAME}")
    print(f"Test split:     {TEST_SPLIT_FILE}")
    print(f"Results saved:  {OUTPUT_FILE}")
    print("\nTest metrics")

    prefix = f"{EVALUATOR_NAME}_cosine_"
    for metric_name, value in metrics.items():
        label = metric_name.removeprefix(prefix)
        print(f"{label}: {value:.6f}")


def main() -> None:
    require_files()

    chunks = load_json(CHUNKS_FILE, "corpus")
    test_split = load_json(TEST_SPLIT_FILE, "test split")
    corpus = build_corpus(chunks)
    queries, relevant_docs = build_test_queries(test_split, corpus)

    print(
        f"Evaluating original model on {len(queries)} held-out test questions "
        f"against {len(corpus)} corpus chunks.",
        flush=True,
    )
    print(f"Loading original model: {MODEL_NAME}", flush=True)

    try:
        model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        evaluator = make_evaluator(queries, corpus, relevant_docs)
        metrics = evaluator(model)
    except Exception as exc:
        raise EvaluationError(
            f"Original model evaluation failed: {type(exc).__name__}: {exc}"
        ) from exc

    save_results(metrics, len(corpus), len(queries))
    print_metrics(metrics)


if __name__ == "__main__":
    try:
        main()
    except EvaluationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
