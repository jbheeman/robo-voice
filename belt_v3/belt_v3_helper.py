import json
import re
import time
from typing import Any, Collection, Mapping, Sequence

from belt_v3_api import call_llm
from rag.belt_v3_rag import rag_search
from movement.belt_v3_valid_movements import VALID_MOVEMENTS
from navigation.belt_v3_valid_navigation import VALID_LOCATIONS

RAG_TOP_K = 3
RAG_MIN_SCORE = 0.30


def safely_parse_json_to_python_dict(input_data: Any) -> dict | None:
    """
    Converts an LLM response into a Python dictionary.

    Handles:
    - Normal JSON
    - Markdown code blocks such as ```json ... ```
    - Extra text before or after the JSON
    - Empty or invalid responses

    Returns:
        A Python dictionary if parsing succeeds.
        None if parsing fails.
    """

    # It is already a Python dictionary.
    if isinstance(input_data, dict):
        return input_data

    if not isinstance(input_data, str) or not input_data.strip():
        print("JSON parsing failed: input is empty or is not a string.")
        return None

    cleaned = input_data.strip()

    # Remove Markdown code fences.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # First, try parsing the entire response normally.
    try:
        parsed = json.loads(cleaned)

        if not isinstance(parsed, dict):
            print("JSON parsing failed: the JSON value is not an object.")
            return None

        return parsed

    except json.JSONDecodeError:
        pass

    # If the LLM added extra text, search for the first JSON object.
    decoder = json.JSONDecoder()

    for index, character in enumerate(cleaned):
        if character != "{":
            continue

        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            continue

    print("JSON parsing failed: no valid JSON object was found.")
    return None



def build_response_prompt(
    user_text: str,
    rag_context: list[dict] | str,
) -> str:
    return f"""
You are the response planner for a receptionist robot named BELT.

Return only valid JSON in exactly this format:

{{
    "simple_action": {{
        "requested": false,
        "actions": []
    }},
    "navigation": {{
        "requested": false,
        "locations": []
    }},
    "speech": "BELT's short natural-language response"
}}

Current user input:
{json.dumps(user_text)}

Relevant UCSC document information:
{json.dumps(rag_context, ensure_ascii=False)}

Supported locations:
{json.dumps(sorted(VALID_LOCATIONS), ensure_ascii=False)}

Supported movements:
{json.dumps(sorted(VALID_MOVEMENTS), ensure_ascii=False)}

Rules:
- Answer casual conversation and general questions naturally and concisely.
- Use relevant UCSC document information for UCSC questions, but ignore it when
  it is unrelated to the user's input.
- Never invent UCSC-specific facts.
- Set "simple_action.requested" to true only when the current user input asks
  BELT to perform a supported physical movement.
- Set "navigation.requested" to true only when the current user input asks to
  find, reach, visit, or be guided to a supported location.
- Do not extract actions or locations that are only mentioned, described, remembered, or discussed.
- Do not treat questions about BELT's abilities as requests.
- Extract every supported movement and location requested in the current input.
- Values in "actions" must exactly match entries in Supported movements.
- Values in "locations" must exactly match entries in Supported locations.
- "requested" must be true if and only if its corresponding list contains at
  least one supported request.
- When a requested movement or location is unsupported, leave it out of the
  command lists and explain the limitation in "speech".
- For valid requests, acknowledge them without claiming they already happened.
- "speech" must contain only what BELT should say, without stage directions.
- Do not include Markdown or any text outside the JSON object.
""".strip()


def _relevant_rag_context(rag_results: list[dict]) -> list[dict[str, str]]:
    relevant_context: list[dict[str, str]] = []

    for result in rag_results:
        score = result.get("score")
        title = result.get("title")
        text = result.get("text")

        if not isinstance(score, (int, float)) or score < RAG_MIN_SCORE:
            continue
        if not isinstance(title, str) or not isinstance(text, str):
            continue

        relevant_context.append({
            "title": title,
            "text": text,
        })

    return relevant_context


def _validated_values(
    raw_values: Any,
    valid_values: Collection[str],
) -> list[str]:
    if not isinstance(raw_values, list):
        return []

    canonical_values = {
        value.casefold(): value
        for value in valid_values
    }
    validated: list[str] = []

    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            continue

        canonical_value = canonical_values.get(
            raw_value.strip().casefold()
        )
        if canonical_value is not None and canonical_value not in validated:
            validated.append(canonical_value)

    return validated


def _validated_request(
    raw_request: Any,
    values_key: str,
    valid_values: Collection[str],
) -> dict:
    if not isinstance(raw_request, Mapping):
        return {
            "requested": False,
            values_key: [],
        }

    values = _validated_values(
        raw_request.get(values_key),
        valid_values,
    )
    requested = raw_request.get("requested") is True and bool(values)

    return {
        "requested": requested,
        values_key: values if requested else [],
    }


def _fallback_response() -> dict:
    return {
        "simple_action": {
            "requested": False,
            "actions": [],
        },
        "navigation": {
            "requested": False,
            "locations": [],
        },
        "speech": "Sorry, I couldn't process that request.",
    }


def _validated_llm_response(
    raw_response: Any,
) -> dict:
    if raw_response is None:
        return _fallback_response()

    parsed_response = safely_parse_json_to_python_dict(raw_response)

    if parsed_response is None:
        return _fallback_response()

    speech = parsed_response.get("speech")
    if not isinstance(speech, str) or not speech.strip():
        return _fallback_response()

    return {
        "simple_action": _validated_request(
            parsed_response.get("simple_action"),
            "actions",
            VALID_MOVEMENTS,
        ),
        "navigation": _validated_request(
            parsed_response.get("navigation"),
            "locations",
            VALID_LOCATIONS,
        ),
        "speech": speech.strip(),
    }


def compose_response(
    user_text,
    conversation: Sequence[Mapping[str, str]],
    debug: bool = False,
):
    if debug:
        rag_start = time.perf_counter()

    rag_results = rag_search(user_text, top_k=RAG_TOP_K)
    rag_context = _relevant_rag_context(rag_results)

    if debug:
        rag_time = time.perf_counter() - rag_start
        print(f"Done Searching RAG ({rag_time:.3f} seconds)")

    if not rag_context:
        rag_context = "No relevant document information found."

    prompt = build_response_prompt(user_text, rag_context)

    if debug:
        llm_start = time.perf_counter()

    llm_response = call_llm(
        prompt,
        conversation=conversation,
        debug=debug,
    )

    if debug:
        llm_time = time.perf_counter() - llm_start
        print(f"Done Calling Response LLM ({llm_time:.3f} seconds)")

    return _validated_llm_response(llm_response), rag_context
