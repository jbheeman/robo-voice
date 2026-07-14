import json
import re
from typing import Any

from belt_v3_api import call_llm
from belt_v3_rag import rag_search


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
    llm_response = call_llm(prompt)
    python_dict_output = safely_parse_json_to_python_dict(llm_response)
    return python_dict_output


def compose_response(
    user_text,
    navigation_output,
    simple_action_output,
):
    rag_context = rag_search(user_text)

    if rag_context is None:
        rag_context = "No relevant document information found."

    prompt = f"""
You are the response composer for a receptionist robot.

User input:
{user_text}

Navigation result:
{navigation_output}

Simple action result:
{simple_action_output}

Relevant document information:
{rag_context}

Write a short natural response to the user.

Rules:
- Respond normally to casual conversation and general questions.
- Use the document information only when it is relevant.
- Do not invent building-specific information.
- If the user asks for building-specific information and no relevant
  document information was found, say that you do not know.
- Only mention navigation or actions that appear in the provided outputs.
- Only confirm navigation or actions marked as valid.
- If an action is valid, say that you can perform it.
- Do not claim that an action has already happened.
- If navigation is requested and valid, ask the user to confirm the
  destination before starting navigation.
- If a requested navigation destination or action is invalid, politely
  explain that it cannot be completed.
- Handle every part of the user's message.
- Keep the response concise.
"""

    return call_llm(prompt)