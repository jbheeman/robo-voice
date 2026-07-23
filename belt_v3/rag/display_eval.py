#!/usr/bin/env python3
"""Print evaluation metrics from one or more JSON files.

With no arguments, this script displays the original embedder evaluation and
the fine-tuned embedder's training summary stored beside this file.

Examples:
    python display_eval.py
    python display_eval.py eval_current_embed_results.json
    python display_eval.py original.json newly_trained.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_FILES = (
    BASE_DIR / "eval_current_embed_results.json",
    BASE_DIR / "embedding_train_evals" / "training_summary.json",
)

ACRONYMS = {
    "f1": "F1",
    "map": "MAP",
    "mrr": "MRR",
    "ndcg": "NDCG",
}
PERCENT_METRICS = ("accuracy", "precision", "recall", "hit_rate", "success")
IMPORTANT_TEST_METRICS = (
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "ndcg@10",
    "mrr@10",
)


class DisplayError(Exception):
    """Raised when an evaluation file cannot be displayed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print evaluation metric blocks from JSON files in a readable "
            "terminal format."
        )
    )
    parser.add_argument(
        "json_files",
        nargs="*",
        type=Path,
        metavar="JSON",
        help=(
            "evaluation JSON file(s); defaults to the original evaluation and "
            "fine-tuned training summary"
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        metavar="N",
        help="decimal places for non-percentage scores (default: 4)",
    )
    args = parser.parse_args()

    if not 0 <= args.precision <= 10:
        parser.error("--precision must be between 0 and 10")

    return args


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise DisplayError(f"file not found: {path}") from exc
    except PermissionError as exc:
        raise DisplayError(f"permission denied: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DisplayError(
            f"invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc
    except OSError as exc:
        raise DisplayError(f"could not read {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise DisplayError(
            f"{path} must contain a JSON object at the top level, "
            f"not {type(data).__name__}"
        )

    return data


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def find_metric_sections(
    data: dict[str, Any],
) -> list[tuple[str, dict[str, int | float]]]:
    """Find dictionaries named "metrics" or ending in "_metrics"."""
    sections: list[tuple[str, dict[str, int | float]]] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = path + (str(key),)
                key_lower = str(key).lower()

                if (
                    isinstance(child, dict)
                    and (key_lower == "metrics" or key_lower.endswith("_metrics"))
                ):
                    numeric_metrics = {
                        str(metric): metric_value
                        for metric, metric_value in child.items()
                        if is_number(metric_value)
                    }
                    if numeric_metrics:
                        sections.append((".".join(child_path), numeric_metrics))
                        continue

                visit(child, child_path)
        elif isinstance(value, list):
            # Evaluation result arrays can be very large and contain per-question
            # fields named like metrics. They are intentionally not displayed.
            return

    visit(data, ())
    return sections


def words_from_key(key: str) -> list[str]:
    words = re.sub(r"(?<=\D)@(?=\d)", "_at_", key)
    words = re.sub(r"[_\-.]+", " ", words).strip().lower().split()
    return words


def title_from_key(key: str) -> str:
    words = words_from_key(key)
    rendered = [ACRONYMS.get(word, word.capitalize()) for word in words]
    return " ".join(rendered)


def metric_label(metric_key: str, section_key: str) -> str:
    name = metric_key.lower()
    section_words = set(words_from_key(section_key))

    # Metric libraries often repeat the split name in every key.
    for prefix in ("trained_validation_", "validation_", "test_", "train_"):
        prefix_name = prefix.removesuffix("_")
        if prefix_name in section_words and name.startswith(prefix):
            name = name[len(prefix) :]
            break

    if name.startswith("cosine_"):
        name = name.removeprefix("cosine_")

    name = re.sub(r"_at_(\d+)$", r"@\1", name)
    label = title_from_key(name)
    return re.sub(r"\s+At\s+(\d+)$", r" @ \1", label)


def important_metric_order(metric_key: str) -> int | None:
    normalized = metric_key.lower().replace("_at_", "@")
    for index, suffix in enumerate(IMPORTANT_TEST_METRICS):
        if normalized.endswith(suffix):
            return index
    return None


def format_metric(metric_key: str, value: int | float, precision: int) -> str:
    key_lower = metric_key.lower()

    if isinstance(value, int):
        return f"{value:,}"
    if any(metric_name in key_lower for metric_name in PERCENT_METRICS):
        return f"{value:.2%}"
    return f"{value:.{precision}f}"


def print_rows(rows: list[tuple[str, str]], indent: str = "  ") -> None:
    if not rows:
        return

    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{indent}{label:<{label_width}}  {value}")


def evaluation_title(path: Path, data: dict[str, Any]) -> str:
    if path.name == "eval_current_embed_results.json":
        return "Original Model - Test Evaluation"
    if "trained_validation_metrics" in data or path.name == "training_summary.json":
        return "Fine-Tuned Model - Test Evaluation"
    return f"{title_from_key(path.stem)} - Test Evaluation"


def print_evaluation(path: Path, data: dict[str, Any], precision: int) -> None:
    all_sections = find_metric_sections(data)
    test_sections = [
        section
        for section in all_sections
        if section[0].split(".")[-1].lower() == "test_metrics"
    ]

    # Retain support for older original-model files that used a generic
    # top-level "metrics" block.
    if not test_sections:
        test_sections = [
            section
            for section in all_sections
            if section[0].split(".")[-1].lower() == "metrics"
        ]

    if not test_sections:
        raise DisplayError(
            f"no numeric test metrics block found in {path}"
        )

    selected_metrics: dict[str, int | float] = {}
    for _, metrics in test_sections:
        selected_metrics.update(
            {
                metric_key: value
                for metric_key, value in metrics.items()
                if important_metric_order(metric_key) is not None
            }
        )

    if not selected_metrics:
        raise DisplayError(f"no important retrieval metrics found in {path}")

    ordered_metrics = sorted(
        selected_metrics.items(),
        key=lambda item: important_metric_order(item[0]),
    )
    section_path = test_sections[0][0]
    rows = [
        (
            metric_label(metric_key, section_path),
            format_metric(metric_key, value, precision),
        )
        for metric_key, value in ordered_metrics
    ]

    divider = "=" * 72
    print(divider)
    print(evaluation_title(path, data))
    print()
    print_rows(rows)
    print(divider)


def main() -> int:
    args = parse_args()
    paths = args.json_files or list(DEFAULT_JSON_FILES)
    had_error = False
    printed_evaluation = False

    for path in paths:
        if printed_evaluation:
            print()
        try:
            data = load_json(path)
            print_evaluation(path, data, args.precision)
            printed_evaluation = True
        except DisplayError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            had_error = True

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
