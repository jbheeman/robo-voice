from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from belt_v2_api import ROBOT_PROFILE, safe_chat_completion, print_balance_status
from belt_v2_memory import add_note, get_user_memory, save_memory, set_user

SHOW_MEMORY_DETECTION = True

YES_WORDS = {
    "yes",
    "y",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "please do",
    "remember it",
    "save it",
}

NO_WORDS = {
    "no",
    "n",
    "nope",
    "nah",
    "dont",
    "don't",
    "do not",
    "dont remember",
    "don't remember",
    "do not remember",
}

#clean up + capitalize name, mostly a chatbot feature
#in audio inputs this shouldnt be used that much
def clean_user_name(raw_name: str) -> Optional[str]:
    name = raw_name.strip()
    name = name.strip(".,!?;:()[]{}\"'")

    if not name:
        return None

    # Only keep first word so "Bella and I like dogs" does not become the name.
    name = name.split()[0]

    if not name:
        return None

    if len(name) > 40:
        return None

    if any(char.isdigit() for char in name):
        return None

    bad_names = {
        "a",
        "an",
        "the",
        "and",
        "but",
        "or",
        "this",
        "that",
        "it",
        "me",
        "you",
    }

    if name.lower() in bad_names:
        return None

    if name.islower():
        name = name.capitalize()

    return name

#checks if 2 names the same, ignore capitaization
def same_user_name(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False

    return a.strip().lower() == b.strip().lower()

#looks through memory, checks if name already exists
def get_existing_user_name(memory: Dict[str, Any], name: str) -> Optional[str]:
    for existing_name in memory.get("users", {}):
        if same_user_name(existing_name, name):
            return existing_name

    return None

#TODO: might remove, this is kinda sloppy
#detects name from natrual introductions
def detect_name_from_message(user_message: str) -> Optional[str]:
    patterns = [
        r"\b(?:hi|hello|hey)?[\s,]*(?:i['’]?m|i am|my name is|call me|this is)\s+([A-Za-z][A-Za-z'_-]*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, user_message, flags=re.IGNORECASE)

        if match:
            return clean_user_name(match.group(1))

    return None

#turns raw name into current active user
def activate_user(memory: Dict[str, Any], raw_name: str) -> Optional[str]:
    cleaned_name = clean_user_name(raw_name)

    if not cleaned_name:
        return None

    existing_name = get_existing_user_name(memory, cleaned_name)
    active_name = existing_name if existing_name else cleaned_name

    set_user(memory, active_name)

    return active_name

#builds system prompt for chatbot, includes belt robot profile and user saved notes
def build_instructions(
    current_user: Optional[str],
    user_memory: Dict[str, Any],
) -> str:
    profile_text = json.dumps(ROBOT_PROFILE, indent=2, ensure_ascii=False)
    memory_text = json.dumps(user_memory, indent=2, ensure_ascii=False)

    return f"""
You are {ROBOT_PROFILE["name"]}, the {ROBOT_PROFILE["role"]}.

Robot profile:
{profile_text}

Current user:
{current_user if current_user else "Unknown"}

Known notes about this user:
{memory_text if user_memory else "No saved memory yet."}

Behavior:
- Speak as BELT, a campus assistant robot.
- Be warm, concise, and helpful.
- If you know the current user's name, you may naturally use it.
- If you use memory, use it naturally.
- Do not invent memory.
- Do not claim you can physically move, see, wave, or point yet.
- You may say things like "In the robot version, I could wave here."
- Do not tell the user that you are running a memory detector. The program handles that separately.
"""

#asks full prompt to llm
def ask_llm(
    client: OpenAI,
    user_message: str,
    current_user: Optional[str],
    memory: Dict[str, Any],
    conversation: List[Dict[str, str]],
) -> Optional[str]:
    user_memory = get_user_memory(memory, current_user)
    instructions = build_instructions(current_user, user_memory)

    messages = [
        {"role": "system", "content": instructions},
        *conversation,
        {"role": "user", "content": user_message},
    ]

    return safe_chat_completion(client, messages)

#tries to parse json from model output
def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or start >= end:
        return None

    try:
        parsed = json.loads(text[start:end + 1])
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None

#filters memory that should not be saved, manual creation
def looks_like_bad_memory(memory_note: str) -> bool:
    note = memory_note.strip().lower()

    if not note:
        return True

    bad_starts = (
        "asked about ",
        "asked for ",
        "asked to ",
        "wants to know ",
        "wanted to know ",
        "is asking about ",
        "requested ",
        "needs help with ",
        "needs an explanation of ",
        "name is ",
        "is named ",
        "is called ",
        "called ",
        "goes by ",
        "introduced themself",
        "introduced themselves",
    )

    bad_exact = {
        "has shared their name.",
        "shared their name.",
        "introduced themself.",
        "introduced themselves.",
    }

    if note in bad_exact:
        return True

    for bad_start in bad_starts:
        if note.startswith(bad_start):
            return True

    return False

#use llm as memory detector, then validates result with other functions
def detect_memory_candidate(
    client: OpenAI,
    user_message: str,
    assistant_reply: str,
    current_user: Optional[str],
) -> Optional[Dict[str, Optional[str]]]:
    detector_prompt = f"""
You are BELT's memory detection system.

Your job:
Detect whether the user's latest message contains a stable personal detail worth saving as a note.

Current user, if known:
{current_user if current_user else "Unknown"}

Only remember stable, useful personal details, such as:
- preferences
- hobbies
- long-term goals
- ongoing projects
- learning interests
- communication preferences
- harmless background information the user clearly shared about themself

Do NOT remember:
- random one-time requests
- weather questions
- jokes
- definitions
- homework questions
- temporary moods
- temporary plans
- secrets
- passwords
- API keys
- addresses
- phone numbers
- private IDs
- payment information
- sensitive personal information unless the user explicitly asks BELT to remember it
- facts about other people unless the user clearly wants that saved
- anything inferred indirectly

Important rules:
- Store memory as a natural-language note, NOT key=value.
- The user's name is identity routing, NOT a memory note.
- If the user only gives their name, return should_remember=false.
- If the user gives a name and another personal detail, put the name in detected_user_name, but memory_note should contain ONLY the non-name detail.
- If current user is already known, do NOT ask to remember or confirm the user's name.
- The consent question should ask only about saving the memory note, not the user's name.
- If there is no stable personal detail besides the name, return should_remember=false.
- Return ONLY valid JSON.
- Do not use markdown.

JSON format:
{{
  "should_remember": true or false,
  "detected_user_name": string or null,
  "memory_note": string or null,
  "consent_question": string or null
}}

Example 1:
User: "Hi, I'm Bella."
Output:
{{
  "should_remember": false,
  "detected_user_name": "Bella",
  "memory_note": null,
  "consent_question": null
}}

Example 2:
User: "Hi, I'm Bella and I like dogs."
Output:
{{
  "should_remember": true,
  "detected_user_name": "Bella",
  "memory_note": "likes dogs.",
  "consent_question": "Do you want me to remember that you like dogs?"
}}

Example 3:
User: "I prefer short explanations."
Output:
{{
  "should_remember": true,
  "detected_user_name": null,
  "memory_note": "prefers short explanations.",
  "consent_question": "Do you want me to remember that you prefer short explanations?"
}}

Example 4:
User: "What's the weather?"
Output:
{{
  "should_remember": false,
  "detected_user_name": null,
  "memory_note": null,
  "consent_question": null
}}

Example 5:
User: "Explain PCA."
Output:
{{
  "should_remember": false,
  "detected_user_name": null,
  "memory_note": null,
  "consent_question": null
}}

Latest user message:
{user_message}

BELT's reply:
{assistant_reply}
"""

    messages = [
        {
            "role": "system",
            "content": "You output only valid JSON. No markdown. No extra text.",
        },
        {
            "role": "user",
            "content": detector_prompt,
        },
    ]

    raw = safe_chat_completion(client, messages)

    if raw is None:
        return None

    parsed = extract_json_object(raw)

    if parsed is None:
        return None

    should_remember = bool(parsed.get("should_remember", False))
    detected_user_name = parsed.get("detected_user_name")
    memory_note = parsed.get("memory_note")
    consent_question = parsed.get("consent_question")

    if not should_remember:
        return None

    if not isinstance(memory_note, str):
        return None

    memory_note = memory_note.strip()

    if looks_like_bad_memory(memory_note):
        return None

    if detected_user_name is not None and not isinstance(detected_user_name, str):
        detected_user_name = None

    if isinstance(detected_user_name, str):
        detected_user_name = clean_user_name(detected_user_name)

    if not isinstance(consent_question, str) or not consent_question.strip():
        consent_question = f"Do you want me to remember that you {memory_note}"

    consent_question = consent_question.strip()

    return {
        "detected_user_name": detected_user_name,
        "memory_note": memory_note,
        "consent_question": consent_question,
    }

#turns user's yes/no response into True/False, manual detection
def parse_consent(user_input: str) -> Optional[bool]:
    text = user_input.strip().lower()

    if text in YES_WORDS:
        return True

    if text in NO_WORDS:
        return False

    for word in YES_WORDS:
        if text.startswith(word + " "):
            return True

    for word in NO_WORDS:
        if text.startswith(word + " "):
            return False

    return None

#saves memory after user gives consent
def save_pending_memory(
    memory: Dict[str, Any],
    pending_memory: Dict[str, Optional[str]],
    fallback_user: Optional[str],
) -> Optional[str]:
    user_name = pending_memory.get("detected_user_name") or fallback_user
    memory_note = pending_memory.get("memory_note")

    if not user_name or not memory_note:
        return None

    add_note(memory, user_name, memory_note)
    save_memory(memory)

    return user_name

#print helpful commands
def print_help() -> None:
    print(
        """
Commands:
  /help                         Show this help menu
  /active_user                  Show which user BELT thinks it is talking to
  /memory                       Show active user's saved notes
  /users                        Show all saved users
  /balance                      Check DeepSeek API balance
  /quit                         Exit

BELT v2 memory behavior:
  - No /user command
  - No /remember key=value command
  - No /note command
  - BELT detects the active user from natural introductions like "I'm Bella"
  - BELT saves only notes, not key=value details
  - BELT does not save the user's name as a memory note
  - BELT asks consent before saving a note

Example:
  You: Hi, I'm Bella and I like dogs.
  BELT: Hi Bella! Nice to meet you.

  Memory detected: {"User": "Bella", "Memory": "likes dogs."}
  BELT: Do you want me to remember that you like dogs?

  You: yes
  BELT: Got it — I'll remember that for Bella.

  You: /active_user
  BELT: active_user = Bella
"""
    )

#handle slash commands
def handle_command(
    user_input: str,
    memory: Dict[str, Any],
    current_user: Optional[str],
) -> bool:
    if user_input == "/help":
        print_help()
        return True

    if user_input == "/balance":
        print_balance_status()
        return True

    if user_input == "/active_user":
        print(f"BELT: active_user = {current_user if current_user else 'null'}")
        return True

    if user_input == "/memory":
        if not current_user:
            print(
                "BELT: active_user = null, so I don't know whose memory to show yet. "
                "Introduce yourself naturally, like: I'm Bella."
            )
            return True

        print(json.dumps(get_user_memory(memory, current_user), indent=2, ensure_ascii=False))
        return True

    if user_input == "/users":
        print(list(memory["users"].keys()))
        return True

    if user_input.startswith("/"):
        print("BELT: Unknown command. Type /help to see available commands.")
        return True

    return False
