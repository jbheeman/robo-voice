from belt_v3_api import call_llm
from belt_v3_helper import safely_parse_json_to_python_dict
import json


import json


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

sentence = input("Type a sentence: ")

action_text =f'''
Detect physical actions in sentence, such as waving, turning around, pointing, etc.
Return it in a JSON file, not as a python dictionary.
Check if the sentence is user telling you to do an acton. Make requrested = True if that's the case
Format: {{"requested":false, actions:[]}}
Sentence: "{sentence}"
'''

location_text =f'''
Detect locations in sentence, such as break room, lab, classroom, room 101
Return it in a JSON file, not as a python dictionary.
Check if the sentence is user trying to find a place. Make requested = True if that's the case
Format: {{"requested":false, locations:[]}}
Sentence: "{sentence}"
'''
prompt = build_detection_prompt(sentence)
response = call_llm(prompt)
result = safely_parse_json_to_python_dict(response)

print(response)
print(result)