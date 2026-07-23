import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers


BASE_DIR = Path(__file__).resolve().parent

QUESTIONS_FILE = BASE_DIR / "ucsc_question_chunk_pairs.json"
CHUNKS_FILE = BASE_DIR / "ucsc_complete_chunks_with_ids.json"

SPLITS_DIR = BASE_DIR / "dataset_splits"
CHECKPOINT_DIR = BASE_DIR / "models" / "ucsc_minilm_checkpoints"
FINAL_MODEL_DIR = BASE_DIR / "models" / "ucsc_minilm_finetuned"
EVALUATION_DIR = BASE_DIR / "embedding_training_evaluations"

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42

NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def extract_chunk_list(data: Any) -> list[dict[str, Any]]:
    """
    Accept either:

    [
        {
            "chunk_id": "chunk_00001",
            "title": "...",
            "text": "...",
            "token_count": 150
        }
    ]

    or:

    {
        "chunks": [...]
    }
    """
    if isinstance(data, list):
        chunks = data
    elif isinstance(data, dict) and isinstance(data.get("chunks"), list):
        chunks = data["chunks"]
    else:
        raise ValueError(
            "The chunks JSON must be a list, or a dictionary containing "
            'a list under the key "chunks".'
        )

    if not chunks:
        raise ValueError("The chunks JSON is empty.")

    return chunks


def extract_chunk_id(chunk: dict[str, Any]) -> str:
    for key in ("chunk_id", "id"):
        value = chunk.get(key)

        if value is not None:
            return str(value).strip()

    raise ValueError(
        "A chunk is missing its ID. Expected a 'chunk_id' or 'id' field."
    )


def extract_document_text(chunk: dict[str, Any]) -> str:
    """
    This uses the chunk's `text` field as the document text.

    Keep this formatting consistent when you later regenerate embeddings
    with the fine-tuned model.
    """
    text = chunk.get("text")

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"Chunk {extract_chunk_id(chunk)} has missing or empty text."
        )

    return text.strip()


def build_chunk_map(chunks: list[dict[str, Any]]) -> dict[str, str]:
    chunk_map: dict[str, str] = {}

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Every chunk must be a JSON object.")

        chunk_id = extract_chunk_id(chunk)

        if chunk_id in chunk_map:
            raise ValueError(f"Duplicate chunk ID found: {chunk_id}")

        chunk_map[chunk_id] = extract_document_text(chunk)

    return chunk_map


def load_question_pairs(
    data: Any,
    chunk_map: dict[str, str],
) -> list[dict[str, str]]:
    """
    Expected format:

    [
        {
            "question": "Why ...?",
            "chunk_id": "chunk_01023"
        }
    ]
    """
    if not isinstance(data, list):
        raise ValueError("The question dataset must be a JSON list.")

    if not data:
        raise ValueError("The question dataset is empty.")

    pairs: list[dict[str, str]] = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Question entry {index} must be a JSON object."
            )

        question = item.get("question")
        chunk_id = item.get("chunk_id")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Question entry {index} has a missing or empty question."
            )

        if chunk_id is None:
            raise ValueError(
                f"Question entry {index} is missing 'chunk_id'."
            )

        normalized_chunk_id = str(chunk_id).strip()

        if normalized_chunk_id not in chunk_map:
            raise ValueError(
                f"Question entry {index} points to unknown chunk ID "
                f"{normalized_chunk_id!r}."
            )

        pairs.append({
            "question": question.strip(),
            "chunk_id": normalized_chunk_id,
        })

    return pairs


def split_by_chunk_id(
    pairs: list[dict[str, str]],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    """
    Split unique chunk IDs first, then place every question for a given
    chunk into the same split. This prevents near-duplicate questions for
    one chunk from leaking between train, validation, and test.
    """
    unique_chunk_ids = sorted({
        pair["chunk_id"]
        for pair in pairs
    })

    if len(unique_chunk_ids) < 3:
        raise ValueError(
            "At least 3 distinct chunk IDs are required for "
            "train/validation/test splitting."
        )

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(unique_chunk_ids)

    total_chunks = len(unique_chunk_ids)

    validation_count = max(
        1,
        round(total_chunks * VALIDATION_RATIO),
    )
    test_count = max(
        1,
        round(total_chunks * TEST_RATIO),
    )
    train_count = total_chunks - validation_count - test_count

    if train_count < 1:
        raise ValueError(
            "Not enough distinct chunk IDs for the requested split ratios."
        )

    train_ids = set(unique_chunk_ids[:train_count])

    validation_start = train_count
    validation_end = train_count + validation_count

    validation_ids = set(
        unique_chunk_ids[validation_start:validation_end]
    )
    test_ids = set(unique_chunk_ids[validation_end:])

    train_pairs = [
        pair for pair in pairs
        if pair["chunk_id"] in train_ids
    ]
    validation_pairs = [
        pair for pair in pairs
        if pair["chunk_id"] in validation_ids
    ]
    test_pairs = [
        pair for pair in pairs
        if pair["chunk_id"] in test_ids
    ]

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    return train_pairs, validation_pairs, test_pairs


def make_training_dataset(
    pairs: list[dict[str, str]],
    chunk_map: dict[str, str],
) -> Dataset:
    rows = [
        {
            "anchor": pair["question"],
            "positive": chunk_map[pair["chunk_id"]],
        }
        for pair in pairs
    ]

    return Dataset.from_list(rows)


def make_ir_evaluator(
    pairs: list[dict[str, str]],
    chunk_map: dict[str, str],
    name: str,
) -> InformationRetrievalEvaluator:
    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}

    for index, pair in enumerate(pairs):
        query_id = f"{name}_question_{index:05d}"
        queries[query_id] = pair["question"]
        relevant_docs[query_id] = {pair["chunk_id"]}

    # Evaluate against the full corpus, not only the labeled split.
    corpus = dict(chunk_map)

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=[10],
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10],
        ndcg_at_k=[10],
        map_at_k=[10],
        show_progress_bar=True,
        batch_size=EVAL_BATCH_SIZE,
        name=name,
        write_csv=True,
    )


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, set):
        return sorted(
            make_json_serializable(item)
            for item in value
        )

    if hasattr(value, "item"):
        return value.item()

    return value


def main() -> None:
    random.seed(RANDOM_SEED)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    chunks_data = load_json(CHUNKS_FILE)
    question_data = load_json(QUESTIONS_FILE)

    chunks = extract_chunk_list(chunks_data)
    chunk_map = build_chunk_map(chunks)
    question_pairs = load_question_pairs(
        question_data,
        chunk_map,
    )

    train_pairs, validation_pairs, test_pairs = split_by_chunk_id(
        question_pairs
    )

    save_json(train_pairs, SPLITS_DIR / "train.json")
    save_json(validation_pairs, SPLITS_DIR / "validation.json")
    save_json(test_pairs, SPLITS_DIR / "test.json")

    print("Dataset split complete")
    print("-" * 40)
    print(f"Corpus chunks:       {len(chunk_map)}")
    print(f"All questions:       {len(question_pairs)}")
    print(f"Training questions:  {len(train_pairs)}")
    print(f"Validation questions:{len(validation_pairs):>6}")
    print(f"Test questions:      {len(test_pairs)}")
    print(
        "Training chunk IDs: ",
        len({pair["chunk_id"] for pair in train_pairs}),
    )
    print(
        "Validation chunk IDs:",
        len({pair["chunk_id"] for pair in validation_pairs}),
    )
    print(
        "Test chunk IDs:     ",
        len({pair["chunk_id"] for pair in test_pairs}),
    )

    train_dataset = make_training_dataset(
        train_pairs,
        chunk_map,
    )
    validation_dataset = make_training_dataset(
        validation_pairs,
        chunk_map,
    )

    validation_evaluator = make_ir_evaluator(
        validation_pairs,
        chunk_map,
        name="validation",
    )
    test_evaluator = make_ir_evaluator(
        test_pairs,
        chunk_map,
        name="test",
    )

    model = SentenceTransformer(BASE_MODEL_NAME)

    training_loss = losses.MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        seed=RANDOM_SEED,
        fp16=False,
        bf16=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        loss=training_loss,
        evaluator=validation_evaluator,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    model.save_pretrained(str(FINAL_MODEL_DIR))

    print("\nEvaluating the trained model on validation data...")
    trained_validation_metrics = validation_evaluator(
        model,
        output_path=str(EVALUATION_DIR),
        epoch=NUM_EPOCHS,
        steps=-1,
    )

    print("\nEvaluating the trained model on the held-out test data...")
    test_metrics = test_evaluator(
        model,
        output_path=str(EVALUATION_DIR),
        epoch=NUM_EPOCHS,
        steps=-1,
    )

    summary = {
        "base_model": BASE_MODEL_NAME,
        "final_model_directory": str(FINAL_MODEL_DIR),
        "random_seed": RANDOM_SEED,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "counts": {
            "corpus_chunks": len(chunk_map),
            "all_questions": len(question_pairs),
            "train_questions": len(train_pairs),
            "validation_questions": len(validation_pairs),
            "test_questions": len(test_pairs),
            "train_chunk_ids": len({
                pair["chunk_id"]
                for pair in train_pairs
            }),
            "validation_chunk_ids": len({
                pair["chunk_id"]
                for pair in validation_pairs
            }),
            "test_chunk_ids": len({
                pair["chunk_id"]
                for pair in test_pairs
            }),
        },
        "training": {
            "epochs": NUM_EPOCHS,
            "batch_size": TRAIN_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "loss": "MultipleNegativesRankingLoss",
        },
        "trained_validation_metrics": make_json_serializable(
            trained_validation_metrics
        ),
        "test_metrics": make_json_serializable(test_metrics),
    }

    summary_file = EVALUATION_DIR / "training_summary.json"
    save_json(summary, summary_file)

    print("\nTraining complete")
    print("-" * 40)
    print(f"Fine-tuned model: {FINAL_MODEL_DIR}")
    print(f"Dataset splits:   {SPLITS_DIR}")
    print(f"Metrics summary:  {summary_file}")
    print(
        "\nNext, regenerate every document embedding using the "
        "fine-tuned model before using it in rag_search."
    )


if __name__ == "__main__":
    main()