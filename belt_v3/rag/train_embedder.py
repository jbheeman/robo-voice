import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

print("[Stage 1/8] Importing training libraries...", flush=True)

import torch
from datasets import Dataset
from google.colab import files
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers

print("[Stage 1/8] Done importing libraries.", flush=True)


# ============================================================
# PATHS
# ============================================================

# Files uploaded through Colab's Files sidebar appear in /content.
INPUT_DIR = Path("/content")

QUESTIONS_FILE = INPUT_DIR / "ucsc_question_chunk_pairs.json"
CHUNKS_FILE = INPUT_DIR / "ucsc_complete_chunks_with_ids.json"

# Final outputs are stored here temporarily until the ZIP is downloaded.
OUTPUT_DIR = Path("/content/ucsc_rag_embedder_training")

INPUT_BACKUP_DIR = OUTPUT_DIR / "input_backups"
SPLITS_DIR = OUTPUT_DIR / "dataset_splits"
FINAL_MODEL_DIR = OUTPUT_DIR / "ucsc_minilm_finetuned"
EVALUATION_DIR = OUTPUT_DIR / "embedding_training_evaluations"

# Checkpoints stay outside OUTPUT_DIR so they are not included in the ZIP.
CHECKPOINT_DIR = Path("/content/ucsc_minilm_checkpoints")

ZIP_BASE_PATH = Path("/content/ucsc_rag_embedder_training")


# ============================================================
# TRAINING SETTINGS
# ============================================================

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42

NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_elapsed(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"

    if minutes:
        return f"{minutes}m {seconds}s"

    return f"{seconds}s"


def stage_message(stage: str, message: str) -> None:
    print(f"\n[{stage}] {message}", flush=True)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Upload the required JSON file through Colab's Files sidebar."
        )

    with path.open("r", encoding="utf-8-sig") as file:
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
    text = chunk.get("text")

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"Chunk {extract_chunk_id(chunk)} has missing or empty text."
        )

    return text.strip()


def build_chunk_map(
    chunks: list[dict[str, Any]],
) -> dict[str, str]:
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
    Split unique chunk IDs first so all questions for one chunk remain
    inside the same train, validation, or test split.
    """
    unique_chunk_ids = sorted({
        pair["chunk_id"]
        for pair in pairs
    })

    if len(unique_chunk_ids) < 3:
        raise ValueError(
            "At least 3 distinct chunk IDs are required."
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

    train_count = (
        total_chunks
        - validation_count
        - test_count
    )

    if train_count < 1:
        raise ValueError(
            "Not enough distinct chunk IDs for the requested splits."
        )

    train_ids = set(
        unique_chunk_ids[:train_count]
    )

    validation_ids = set(
        unique_chunk_ids[
            train_count:
            train_count + validation_count
        ]
    )

    test_ids = set(
        unique_chunk_ids[
            train_count + validation_count:
        ]
    )

    train_pairs = [
        pair
        for pair in pairs
        if pair["chunk_id"] in train_ids
    ]

    validation_pairs = [
        pair
        for pair in pairs
        if pair["chunk_id"] in validation_ids
    ]

    test_pairs = [
        pair
        for pair in pairs
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
        relevant_docs[query_id] = {
            pair["chunk_id"]
        }

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=dict(chunk_map),
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


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main() -> None:
    total_start = time.perf_counter()
    random.seed(RANDOM_SEED)

    stage_message(
        "Stage 2/8",
        "Checking hardware and preparing output folders...",
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
        flush=True,
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
            flush=True,
        )

    else:
        print(
            "Warning: No GPU detected. "
            "In Colab, select Runtime > Change runtime type > GPU.",
            flush=True,
        )

    for directory in (
        OUTPUT_DIR,
        INPUT_BACKUP_DIR,
        SPLITS_DIR,
        FINAL_MODEL_DIR,
        EVALUATION_DIR,
        CHECKPOINT_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        "[Stage 2/8] Hardware check and folder setup complete.",
        flush=True,
    )

    stage_message(
        "Stage 3/8",
        "Loading and validating uploaded JSON files...",
    )

    data_start = time.perf_counter()

    chunks_data = load_json(CHUNKS_FILE)
    question_data = load_json(QUESTIONS_FILE)

    # Save backup copies inside the final ZIP.
    shutil.copy2(
        QUESTIONS_FILE,
        INPUT_BACKUP_DIR / QUESTIONS_FILE.name,
    )

    shutil.copy2(
        CHUNKS_FILE,
        INPUT_BACKUP_DIR / CHUNKS_FILE.name,
    )

    chunks = extract_chunk_list(chunks_data)
    chunk_map = build_chunk_map(chunks)

    question_pairs = load_question_pairs(
        question_data,
        chunk_map,
    )

    print(
        f"[Stage 3/8] Loaded {len(chunk_map)} chunks and "
        f"{len(question_pairs)} question pairs in "
        f"{format_elapsed(time.perf_counter() - data_start)}.",
        flush=True,
    )

    stage_message(
        "Stage 4/8",
        "Splitting data and creating training datasets...",
    )

    split_start = time.perf_counter()

    train_pairs, validation_pairs, test_pairs = (
        split_by_chunk_id(question_pairs)
    )

    save_json(
        train_pairs,
        SPLITS_DIR / "train.json",
    )

    save_json(
        validation_pairs,
        SPLITS_DIR / "validation.json",
    )

    save_json(
        test_pairs,
        SPLITS_DIR / "test.json",
    )

    print("\nDataset split complete")
    print("-" * 45)
    print(f"Corpus chunks:        {len(chunk_map)}")
    print(f"All questions:        {len(question_pairs)}")
    print(f"Training questions:   {len(train_pairs)}")
    print(f"Validation questions: {len(validation_pairs)}")
    print(f"Test questions:       {len(test_pairs)}")

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

    print(
        f"[Stage 4/8] Splits and datasets ready in "
        f"{format_elapsed(time.perf_counter() - split_start)}.",
        flush=True,
    )

    stage_message(
        "Stage 5/8",
        "Loading the base MiniLM embedding model...",
    )

    model_start = time.perf_counter()

    model = SentenceTransformer(
        BASE_MODEL_NAME,
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    print(
        f"[Stage 5/8] Base model loaded in "
        f"{format_elapsed(time.perf_counter() - model_start)}.",
        flush=True,
    )

    training_loss = (
        losses.MultipleNegativesRankingLoss(model)
    )

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
        fp16=torch.cuda.is_available(),
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

    stage_message(
        "Stage 6/8",
        f"Starting fine-tuning for {NUM_EPOCHS} epochs. "
        "The progress bar below shows training steps.",
    )

    training_start = time.perf_counter()

    trainer.train()

    print(
        f"[Stage 6/8] Fine-tuning finished in "
        f"{format_elapsed(time.perf_counter() - training_start)}.",
        flush=True,
    )

    stage_message(
        "Stage 7/8",
        "Saving the model and evaluating validation/test sets...",
    )

    evaluation_start = time.perf_counter()

    print(
        "Saving the final model...",
        flush=True,
    )

    model.save_pretrained(
        str(FINAL_MODEL_DIR)
    )

    print(
        "Final model saved.",
        flush=True,
    )

    print(
        "Evaluating the trained model on validation data...",
        flush=True,
    )

    validation_metrics = validation_evaluator(
        model,
        output_path=str(EVALUATION_DIR),
        epoch=NUM_EPOCHS,
        steps=-1,
    )

    print(
        "Validation evaluation complete.",
        flush=True,
    )

    print(
        "Evaluating the trained model on held-out test data...",
        flush=True,
    )

    test_metrics = test_evaluator(
        model,
        output_path=str(EVALUATION_DIR),
        epoch=NUM_EPOCHS,
        steps=-1,
    )

    print(
        "Test evaluation complete.",
        flush=True,
    )

    print(
        f"[Stage 7/8] Saving and evaluation finished in "
        f"{format_elapsed(time.perf_counter() - evaluation_start)}.",
        flush=True,
    )

    stage_message(
        "Stage 8/8",
        "Writing the summary, creating the ZIP, and starting download...",
    )

    summary = {
        "base_model": BASE_MODEL_NAME,
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
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
        },
        "training": {
            "epochs": NUM_EPOCHS,
            "batch_size": TRAIN_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "loss": "MultipleNegativesRankingLoss",
        },
        "trained_validation_metrics": (
            make_json_serializable(validation_metrics)
        ),
        "test_metrics": (
            make_json_serializable(test_metrics)
        ),
    }

    summary_file = (
        EVALUATION_DIR
        / "training_summary.json"
    )

    save_json(
        summary,
        summary_file,
    )

    zip_path = shutil.make_archive(
        str(ZIP_BASE_PATH),
        "zip",
        root_dir=OUTPUT_DIR,
    )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print("\nTraining complete")
    print("-" * 45)
    print(f"Total runtime:       {format_elapsed(total_elapsed)}")
    print(f"Final model folder:  {FINAL_MODEL_DIR}")
    print(f"Dataset splits:      {SPLITS_DIR}")
    print(f"Metrics summary:     {summary_file}")
    print(f"ZIP file:            {zip_path}")
    print(
        "\nThe browser download should begin now. "
        "Do not reset the runtime until it finishes.",
        flush=True,
    )

    files.download(zip_path)


if __name__ == "__main__":
    main()