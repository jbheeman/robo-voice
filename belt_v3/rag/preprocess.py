import json
import re
from pathlib import Path

from transformers import AutoTokenizer



RAG_DIR = Path(__file__).resolve().parent
BELT_V3_DIR = RAG_DIR.parent

INPUT_FILE = BELT_V3_DIR / "documents" / "ucsc_complete.txt"
OUTPUT_FILE = RAG_DIR / "ucsc_complete_chunks.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 200


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def count_tokens(title: str, text: str) -> int:
    full_text = f"{title}\n\n{text}"

    return len(
        tokenizer.encode(
            full_text,
            add_special_tokens=False
        )
    )


def parse_pages(raw_text: str) -> list[tuple[str, str]]:
    """
    Expected format:

    About UCSC;
    Page text here...

    Academic Advising;
    Page text here...
    """
    raw_text = raw_text.replace("\r\n", "\n").strip()

    page_blocks = re.split(
        r"\n\s*\n+",
        raw_text
    )

    pages = []

    for block in page_blocks:
        lines = [
            normalize_text(line)
            for line in block.splitlines()
            if line.strip()
        ]

        if len(lines) < 2:
            continue

        # First line is the webpage title.
        title = lines[0].rstrip(";").strip()

        # Everything after the first line is webpage content.
        text = " ".join(lines[1:]).strip()

        if title and text:
            pages.append((title, text))

    return pages


def split_into_units(text: str) -> list[str]:
    """
    Split using sentence endings and semicolons.

    Semicolons are useful boundaries because the scraped UCSC text
    contains many semicolon-separated webpage elements.
    """
    units = re.split(
        r"(?<=[.!?;])\s+",
        text
    )

    return [
        normalize_text(unit)
        for unit in units
        if normalize_text(unit)
    ]


def split_long_unit(
    title: str,
    unit: str
) -> list[str]:
    """
    Split a single unit that is too large to fit inside one chunk.
    """
    title_token_count = len(
        tokenizer.encode(
            title,
            add_special_tokens=False
        )
    )

    available_tokens = MAX_TOKENS - title_token_count - 2

    if available_tokens <= 0:
        raise ValueError(
            f"Title is too long to fit into {MAX_TOKENS} tokens: {title}"
        )

    unit_token_ids = tokenizer.encode(
        unit,
        add_special_tokens=False
    )

    pieces = []

    for start in range(0, len(unit_token_ids), available_tokens):
        piece_token_ids = unit_token_ids[
            start:start + available_tokens
        ]

        piece = tokenizer.decode(
            piece_token_ids,
            skip_special_tokens=True
        ).strip()

        if piece:
            pieces.append(piece)

    return pieces


def chunk_page(
    title: str,
    text: str
) -> list[dict]:
    units = split_into_units(text)

    expanded_units = []

    for unit in units:
        if count_tokens(title, unit) <= MAX_TOKENS:
            expanded_units.append(unit)
        else:
            expanded_units.extend(
                split_long_unit(title, unit)
            )

    chunks = []
    current_units = []

    for unit in expanded_units:
        candidate_units = current_units + [unit]
        candidate_text = " ".join(candidate_units)

        if (
            current_units
            and count_tokens(title, candidate_text) > MAX_TOKENS
        ):
            chunk_text = " ".join(current_units)

            chunks.append({
                "title": title,
                "text": chunk_text,
                "token_count": count_tokens(title, chunk_text)
            })

            current_units = [unit]

        else:
            current_units.append(unit)

    if current_units:
        chunk_text = " ".join(current_units)

        chunks.append({
            "title": title,
            "text": chunk_text,
            "token_count": count_tokens(title, chunk_text)
        })

    return chunks


def main() -> None:
    raw_text = Path(INPUT_FILE).read_text(
        encoding="utf-8"
    )

    pages = parse_pages(raw_text)

    all_chunks = []

    for title, text in pages:
        page_chunks = chunk_page(
            title=title,
            text=text
        )

        all_chunks.extend(page_chunks)

    Path(OUTPUT_FILE).write_text(
        json.dumps(
            all_chunks,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(f"Processed {len(pages)} webpages.")
    print(f"Created {len(all_chunks)} chunks.")
    print(f"Saved chunks to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()