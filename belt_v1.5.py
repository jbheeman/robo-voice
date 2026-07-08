'''
BELT Version 1.5

Features:
- Simple terminal-based chatbot for BELT
- DeepSeek API for main LLM responses
- Notes-only memory stored in memory_v1_5.json
- No manual /user command
- No manual /remember command
- No manual /note command
- Uses /active_user to show who BELT thinks it is talking to
- Infers active user from natural introductions like "Hi, I'm Bella"
- Does NOT save the user's name as a memory note
- Automatically detects stable personal details worth remembering
- Asks for consent before saving detected memory notes
- Does not save random one-time requests like weather questions
'''

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import APIStatusError, AuthenticationError, OpenAI

load_dotenv()

MEMORY_FILE = Path("memory_v1_5.json")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"

MODEL = "deepseek-v4-flash"

SHOW_MEMORY_DETECTION = True

ROBOT_PROFILE = {
    "name": "BELT",
    "role": "UCSC Campus Robot Assistant",
    "favorite_color": "blue",
    "favorite_animal": "porcupine",
    "favorite_ice_cream_flavor": "mint chocolate chip",
    "rules": [
        "Keep responses short unless the user asks for detail.",
        "Do not claim to see objects, faces, or locations unless a CV module provides that info.",
        "Ask for consent before remembering personal details.",
        "For now, only use text memory. Face recognition is not implemented yet.",
        "If unsure, say you are not sure.",
    ],
    "personality": [
        "friendly",
        "curious",
        "helpful",
        "slightly witty",
        "patient with beginners",
    ],
}

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


# =============================================================================
# DeepSeek setup
# =============================================================================

def get_deepseek_api_key() -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Put it in your .env file like:\n"
            "DEEPSEEK_API_KEY=sk-your-key-here"
        )

    return api_key


def make_deepseek_client() -> OpenAI:
    return OpenAI(
        api_key=get_deepseek_api_key(),
        base_url=DEEPSEEK_BASE_URL,
    )


def check_deepseek_balance() -> Dict[str, Any]:
    api_key = get_deepseek_api_key()

    request = urllib.request.Request(
        DEEPSEEK_BALANCE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw_data = response.read().decode("utf-8")
            return json.loads(raw_data)

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")

        if e.code == 401:
            raise RuntimeError(
                "DeepSeek says your API key is invalid. Check DEEPSEEK_API_KEY in your .env file."
            ) from e

        raise RuntimeError(
            f"Could not check DeepSeek balance. HTTP {e.code}: {body}"
        ) from e

    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not connect to DeepSeek balance endpoint: {e}"
        ) from e


def print_balance_status() -> bool:
    try:
        balance = check_deepseek_balance()

        is_available = balance.get("is_available", False)
        balance_infos = balance.get("balance_infos", [])

        print("\nDeepSeek balance check:")
        print(f"  API available: {is_available}")

        if balance_infos:
            for info in balance_infos:
                currency = info.get("currency", "UNKNOWN")
                total = info.get("total_balance", "0")
                granted = info.get("granted_balance", "0")
                topped_up = info.get("topped_up_balance", "0")

                print(f"  Currency: {currency}")
                print(f"  Total balance: {total}")
                print(f"  Granted balance: {granted}")
                print(f"  Topped-up balance: {topped_up}")
        else:
            print("  No balance info returned.")

        print()
        return is_available

    except Exception as e:
        print(f"\nBELT: Could not check DeepSeek balance: {e}\n")
        return False


def safe_chat_completion(
    client: OpenAI,
    messages: List[Dict[str, str]],
) -> Optional[str]:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        return response.choices[0].message.content

    except AuthenticationError:
        print(
            "BELT: DeepSeek rejected the API key. Check your .env file and make sure "
            "DEEPSEEK_API_KEY is correct."
        )
        return None

    except APIStatusError as e:
        if e.status_code == 402:
            print(
                "BELT: DeepSeek API connection works, but your account has insufficient balance. "
                "Please top up your DeepSeek balance before chatting."
            )
            return None

        if e.status_code == 429:
            print(
                "BELT: DeepSeek says we are sending requests too quickly. "
                "Wait a bit and try again."
            )
            return None

        if e.status_code in [500, 503]:
            print(
                "BELT: DeepSeek's server is having issues right now. "
                "Try again later."
            )
            return None

        print(f"BELT: DeepSeek API error {e.status_code}: {e}")
        return None

    except Exception as e:
        print(f"BELT: Unexpected error: {e}")
        return None


# =============================================================================
# Memory storage: notes only
# =============================================================================

def normalize_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {"users": {}}

    users = memory.get("users", {})
    if not isinstance(users, dict):
        return normalized

    for name, data in users.items():
        if not isinstance(name, str):
            continue

        notes: List[str] = []

        if isinstance(data, dict):
            old_notes = data.get("notes", [])
            if isinstance(old_notes, list):
                notes = [
                    str(note).strip()
                    for note in old_notes
                    if str(note).strip()
                ]

        normalized["users"][name] = {"notes": notes}

    return normalized


def load_memory() -> Dict[str, Any]:
    if not MEMORY_FILE.exists():
        return {"users": {}}

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        raw_memory = json.load(f)

    return normalize_memory(raw_memory)


def save_memory(memory: Dict[str, Any]) -> None:
    memory = normalize_memory(memory)

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def set_user(memory: Dict[str, Any], name: str) -> None:
    name = name.strip()

    if not name:
        return

    if name not in memory["users"]:
        memory["users"][name] = {"notes": []}


def get_user_memory(memory: Dict[str, Any], name: Optional[str]) -> Dict[str, Any]:
    if not name:
        return {}

    return memory["users"].get(name, {"notes": []})


def add_note(memory: Dict[str, Any], name: str, note: str) -> bool:
    name = name.strip()
    note = note.strip()

    if not name or not note:
        return False

    set_user(memory, name)

    notes = memory["users"][name]["notes"]
    existing_notes = {old_note.strip().lower() for old_note in notes}

    if note.lower() in existing_notes:
        return False

    notes.append(note)
    return True


# =============================================================================
# Active user detection
# =============================================================================

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


def same_user_name(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False

    return a.strip().lower() == b.strip().lower()


def get_existing_user_name(memory: Dict[str, Any], name: str) -> Optional[str]:
    for existing_name in memory.get("users", {}):
        if same_user_name(existing_name, name):
            return existing_name

    return None


def detect_name_from_message(user_message: str) -> Optional[str]:
    patterns = [
        r"\b(?:hi|hello|hey)?[\s,]*(?:i['’]?m|i am|my name is|call me|this is)\s+([A-Za-z][A-Za-z'_-]*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, user_message, flags=re.IGNORECASE)

        if match:
            return clean_user_name(match.group(1))

    return None


def activate_user(memory: Dict[str, Any], raw_name: str) -> Optional[str]:
    cleaned_name = clean_user_name(raw_name)

    if not cleaned_name:
        return None

    existing_name = get_existing_user_name(memory, cleaned_name)
    active_name = existing_name if existing_name else cleaned_name

    set_user(memory, active_name)

    return active_name


# =============================================================================
# Prompt building
# =============================================================================

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


# =============================================================================
# Automatic memory detection
# =============================================================================

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


# =============================================================================
# Consent handling
# =============================================================================

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


# =============================================================================
# Commands
# =============================================================================

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

BELT v1.5 memory behavior:
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


# =============================================================================
# Main loop
# =============================================================================

def main() -> None:
    client = make_deepseek_client()
    memory = load_memory()

    current_user: Optional[str] = None
    pending_memory: Optional[Dict[str, Optional[str]]] = None
    waiting_for_name_to_save_memory = False

    conversation: List[Dict[str, str]] = []

    print("BELT v1.5 chatbot prototype online 🤖")
    print(f"Using model: {MODEL}")
    print("Type /help for commands.\n")

    has_balance = print_balance_status()

    if not has_balance:
        print(
            "BELT: Warning: DeepSeek says your account may not have usable balance. "
            "You can still use commands like /active_user and /memory, but chatting with the model may fail.\n"
        )

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input == "/quit":
            print("BELT: Goodbye! Powering down politely.")
            break

        if handle_command(user_input, memory, current_user):
            continue

        # ---------------------------------------------------------------------
        # Case 1:
        # BELT previously asked for consent, user said yes,
        # but BELT did not know which user to save the memory under.
        # ---------------------------------------------------------------------
        if waiting_for_name_to_save_memory and pending_memory:
            detected_name = detect_name_from_message(user_input)

            if detected_name is None:
                detected_name = user_input

            activated_user = activate_user(memory, detected_name)

            if not activated_user:
                print("BELT: I still need a simple name to save that under, like: I'm Bella.")
                continue

            current_user = activated_user
            pending_memory["detected_user_name"] = current_user

            save_pending_memory(
                memory=memory,
                pending_memory=pending_memory,
                fallback_user=current_user,
            )

            pending_memory = None
            waiting_for_name_to_save_memory = False
            conversation = []

            print(f"BELT: Got it — I'll remember that for {current_user}.")
            continue

        # ---------------------------------------------------------------------
        # Case 2:
        # BELT previously detected a memory and is waiting for yes/no.
        # ---------------------------------------------------------------------
        if pending_memory:
            consent = parse_consent(user_input)

            if consent is True:
                saved_user = save_pending_memory(
                    memory=memory,
                    pending_memory=pending_memory,
                    fallback_user=current_user,
                )

                if saved_user:
                    current_user = activate_user(memory, saved_user) or saved_user
                    save_memory(memory)

                    print(f"BELT: Got it — I'll remember that for {current_user}.")
                    pending_memory = None
                    continue

                waiting_for_name_to_save_memory = True
                print("BELT: Sure — what name should I save that under? You can say: I'm Bella.")
                continue

            if consent is False:
                print("BELT: No worries — I won't save that.")
                pending_memory = None
                continue

            print("BELT: Should I save that memory? Please say yes or no.")
            continue

        # ---------------------------------------------------------------------
        # Case 3:
        # Natural active user detection.
        # This happens BEFORE the LLM reply so BELT can say "Hi Bella".
        # The name is NOT saved as a memory note.
        # ---------------------------------------------------------------------
        introduced_name = detect_name_from_message(user_input)

        if introduced_name:
            activated_user = activate_user(memory, introduced_name)

            if activated_user:
                if current_user and not same_user_name(current_user, activated_user):
                    conversation = []

                current_user = activated_user
                save_memory(memory)

        # ---------------------------------------------------------------------
        # Case 4:
        # Normal chatbot response.
        # ---------------------------------------------------------------------
        answer = ask_llm(
            client=client,
            user_message=user_input,
            current_user=current_user,
            memory=memory,
            conversation=conversation,
        )

        if answer is None:
            continue

        print(f"BELT: {answer}\n")

        # ---------------------------------------------------------------------
        # Case 5:
        # Automatic memory detection after the normal reply.
        # ---------------------------------------------------------------------
        memory_candidate = detect_memory_candidate(
            client=client,
            user_message=user_input,
            assistant_reply=answer,
            current_user=current_user,
        )

        assistant_history_text = answer

        if memory_candidate:
            detected_name = memory_candidate.get("detected_user_name")
            memory_note = memory_candidate.get("memory_note")
            consent_question = memory_candidate.get("consent_question")

            if detected_name:
                activated_user = activate_user(memory, detected_name)

                if activated_user:
                    current_user = activated_user
                    memory_candidate["detected_user_name"] = current_user
                    save_memory(memory)

            pending_memory = memory_candidate

            if SHOW_MEMORY_DETECTION:
                display_user = current_user if current_user else "Unknown"
                print(
                    "Memory detected: "
                    + json.dumps(
                        {
                            "User": display_user,
                            "Memory": memory_note,
                        },
                        ensure_ascii=False,
                    )
                )

            print(f"BELT: {consent_question}\n")

            assistant_history_text = f"{answer}\n{consent_question}"

        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": assistant_history_text})

        conversation = conversation[-10:]


if __name__ == "__main__":
    main()