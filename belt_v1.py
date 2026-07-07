'''




'''


import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, AuthenticationError

load_dotenv()

MEMORY_FILE = Path("memory.json")

# DeepSeek setup
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"

# Cheap/simple model for your prototype
MODEL = "deepseek-v4-flash"

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
    """
    Checks if your DeepSeek account has usable API balance.

    Returns DeepSeek's balance response, something like:
    {
      "is_available": true,
      "balance_infos": [...]
    }
    """
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
    """
    Prints current DeepSeek balance status.

    Returns:
        True if DeepSeek says API calls are available.
        False otherwise.
    """
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


def load_memory() -> Dict[str, Any]:
    if not MEMORY_FILE.exists():
        return {"users": {}}

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory: Dict[str, Any]) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


def get_user_memory(memory: Dict[str, Any], name: Optional[str]) -> Dict[str, Any]:
    if not name:
        return {}

    return memory["users"].get(name, {})


def set_user(memory: Dict[str, Any], name: str) -> None:
    if name not in memory["users"]:
        memory["users"][name] = {
            "details": {},
            "notes": [],
        }


def remember_detail(memory: Dict[str, Any], name: str, key: str, value: str) -> None:
    set_user(memory, name)
    memory["users"][name]["details"][key] = value


def add_note(memory: Dict[str, Any], name: str, note: str) -> None:
    set_user(memory, name)
    memory["users"][name]["notes"].append(note)


def build_instructions(current_user: Optional[str], user_memory: Dict[str, Any]) -> str:
    profile_text = json.dumps(ROBOT_PROFILE, indent=2)
    memory_text = json.dumps(user_memory, indent=2)

    return f"""
You are {ROBOT_PROFILE["name"]}, the {ROBOT_PROFILE["role"]}.

Robot profile:
{profile_text}

Current user:
{current_user if current_user else "Unknown"}

Known memory about this user:
{memory_text if user_memory else "No saved memory yet."}

Behavior:
- Speak as BELT, a campus assistant robot.
- Be warm, concise, and helpful.
- If you use memory, do it naturally.
- Do not invent memory.
- Do not claim you can physically move, see, wave, or point yet.
- You may say things like "In the robot version, I could wave here."
"""


def ask_llm(
    client: OpenAI,
    user_message: str,
    current_user: Optional[str],
    memory: Dict[str, Any],
    conversation: List[Dict[str, str]],
) -> str:
    user_memory = get_user_memory(memory, current_user)
    instructions = build_instructions(current_user, user_memory)

    # DeepSeek uses Chat Completions style.
    # So we put the robot instructions as the system message.
    messages = [
        {"role": "system", "content": instructions},
        *conversation,
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    return response.choices[0].message.content


def safe_ask_llm(
    client: OpenAI,
    user_message: str,
    current_user: Optional[str],
    memory: Dict[str, Any],
    conversation: List[Dict[str, str]],
) -> Optional[str]:
    """
    Wrapper around ask_llm.

    It catches common API/payment errors and prints nicer messages.
    """
    try:
        return ask_llm(
            client=client,
            user_message=user_message,
            current_user=current_user,
            memory=memory,
            conversation=conversation,
        )

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


def print_help() -> None:
    print(
        """
Commands:
  /help                         Show this help menu
  /user NAME                    Set current user
  /remember key=value           Save a detail for current user
  /note something               Save a note for current user
  /memory                       Show current user's memory
  /users                        Show all saved users
  /balance                      Check DeepSeek API balance
  /quit                         Exit

Examples:
  /user Maya
  /remember favorite_color=green
  /remember favorite_topic=robotics
  /note Maya likes short explanations
"""
    )


def main() -> None:
    client = make_deepseek_client()
    memory = load_memory()
    current_user: Optional[str] = None
    conversation: List[Dict[str, str]] = []

    print("BELT chatbot prototype online 🤖")
    print(f"Using model: {MODEL}")
    print("Type /help for commands.\n")

    # Optional startup balance check
    has_balance = print_balance_status()

    if not has_balance:
        print(
            "BELT: Warning: DeepSeek says your account may not have usable balance. "
            "You can still use commands like /user and /memory, but chatting with the model may fail.\n"
        )

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input == "/quit":
            print("BELT: Goodbye! Powering down politely.")
            break

        if user_input == "/help":
            print_help()
            continue

        if user_input == "/balance":
            print_balance_status()
            continue

        if user_input.startswith("/user "):
            current_user = user_input.replace("/user ", "", 1).strip()
            set_user(memory, current_user)
            save_memory(memory)
            conversation = []
            print(f"BELT: Hi {current_user}! I’ll use your saved memory for this chat.")
            continue

        if user_input.startswith("/remember "):
            if not current_user:
                print("BELT: Set a user first with /user NAME.")
                continue

            raw = user_input.replace("/remember ", "", 1).strip()

            if "=" not in raw:
                print("BELT: Use format /remember key=value")
                continue

            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip()

            remember_detail(memory, current_user, key, value)
            save_memory(memory)

            print(f"BELT: Got it. I’ll remember {key} = {value} for {current_user}.")
            continue

        if user_input.startswith("/note "):
            if not current_user:
                print("BELT: Set a user first with /user NAME.")
                continue

            note = user_input.replace("/note ", "", 1).strip()
            add_note(memory, current_user, note)
            save_memory(memory)

            print(f"BELT: Saved that note for {current_user}.")
            continue

        if user_input == "/memory":
            if not current_user:
                print("BELT: Set a user first with /user NAME.")
                continue

            print(json.dumps(get_user_memory(memory, current_user), indent=2))
            continue

        if user_input == "/users":
            print(list(memory["users"].keys()))
            continue

        answer = safe_ask_llm(
            client=client,
            user_message=user_input,
            current_user=current_user,
            memory=memory,
            conversation=conversation,
        )

        if answer is None:
            continue

        print(f"BELT: {answer}\n")

        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": answer})

        # Keep only recent context so it doesn't grow forever.
        conversation = conversation[-10:]


if __name__ == "__main__":
    main()