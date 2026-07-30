"""Interactively inspect the model's ordered BELT output list."""

from belt_v3_helper import (
    _validated_llm_response,
    build_response_prompt,
)
from belt_v3_new_api import call_llm


def main() -> None:
    sentence = input("Type a sentence: ").strip()
    prompt = build_response_prompt(
        sentence,
        "No relevant document information found.",
    )
    raw_response = call_llm(prompt, debug=True)
    validated_response = _validated_llm_response(raw_response)

    print(f"Raw response: {raw_response}")
    print(f"Validated output: {validated_response}")


if __name__ == "__main__":
    main()
