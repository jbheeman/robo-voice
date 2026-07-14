import json
import re
from typing import Any

from belt_v3_api import call_llm


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
    print(f"Raw input: {input_data}")
    return None



def build_location_action_detection_prompt(sentence: str) -> str:
    return f"""
You are BELT's request-detection module.

Your job is to detect only:
1. Physical actions the user is asking BELT to perform.
2. Locations the user wants BELT to guide them to, help them find, or navigate toward.

Return only valid JSON in exactly this format:

{{
    "simple_action": {{
        "requested": false,
        "actions": []
    }},
    "navigation": {{
        "requested": false,
        "locations": []
    }}
}}

Rules:
- Set "simple_action.requested" to true only when the user directly or indirectly asks BELT to perform a physical action.
- Set "navigation.requested" to true only when the user wants to find, reach, visit, or be guided to a location.
- Do not extract actions or locations that are only mentioned, described, remembered, or discussed.
- Do not treat questions about BELT's abilities as requests.
- Extract every requested action and location.
- Use short normalized action names such as "wave", "spin", "nod", or "point left".
- Preserve location names as written by the user.
- Do not invent actions or locations.
- Do not include explanations or Markdown.

Examples:

Sentence: "Can you wave and take me to room 101?"
Output:
{{
    "simple_action": {{
        "requested": true,
        "actions": ["wave"]
    }},
    "navigation": {{
        "requested": true,
        "locations": ["room 101"]
    }}
}}

Sentence: "I saw someone waving near the library."
Output:
{{
    "simple_action": {{
        "requested": false,
        "actions": []
    }},
    "navigation": {{
        "requested": false,
        "locations": []
    }}
}}

Sentence: "Do you know how to dance?"
Output:
{{
    "simple_action": {{
        "requested": false,
        "actions": []
    }},
    "navigation": {{
        "requested": false,
        "locations": []
    }}
}}

Sentence: {json.dumps(sentence)}
""".strip()



def extract_nav_action(text_input):
    prompt = build_location_action_detection_prompt(text_input)
    