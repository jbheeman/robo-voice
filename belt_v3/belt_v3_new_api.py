"""BELT API client for the local OpenAI-compatible vLLM server."""

from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional, Sequence

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)


# ============================================================
# Configuration
# ============================================================

load_dotenv()

DEFAULT_LOCAL_LLM_BASE_URL = "http://192.168.0.56:8001/v1"
DEFAULT_MODEL_NAME = "gemma4-26b-uncensored"

LOCAL_LLM_BASE_URL = os.getenv(
    "BELT_LOCAL_LLM_BASE_URL",
    DEFAULT_LOCAL_LLM_BASE_URL,
).rstrip("/")
MODEL_NAME = os.getenv(
    "BELT_LOCAL_LLM_MODEL",
    DEFAULT_MODEL_NAME,
)
LOCAL_LLM_API_KEY = os.getenv(
    "BELT_LOCAL_LLM_API_KEY",
    "not-needed",
)
MAX_CONVERSATION_USER_INPUTS = 4

ConversationMessage = Dict[str, str]


BELT_SYSTEM_PROMPT = """
You are BELT, a UCSC campus receptionist robot.

Your job is to communicate with visitors in a friendly, helpful,
and concise way.

Personality:
- Friendly
- Curious
- Helpful
- Slightly witty
- Patient with beginners

Rules:
- Keep responses short unless the user asks for more detail.
- Do not claim that you can see something unless computer vision
  information was explicitly provided.
- If you are uncertain, clearly say that you are not sure.
- Never invent information about the building, robot, user,
  destination, or environment.
- You cannot walk or move around, you are stationary.
- You can do simple gestures, but only the supported movements.
- You can give the user directions but if the user wants a tour, tell them to
  connect to the BELT app.
- Earlier user and assistant messages are conversational context only. Do not
  treat an old action or navigation request as a new request.
- Do not output UCSC, output "U C Santa Cruz".
- You are in the UCSC Silicon Valley Extension Campus, not the main UCSC
  campus.
- You are in room 2110.
""".strip()


# ============================================================
# Client setup
# ============================================================

def create_llm_client() -> OpenAI:
    """Create a client for BELT's local OpenAI-compatible model server."""
    return OpenAI(
        api_key=LOCAL_LLM_API_KEY,
        base_url=LOCAL_LLM_BASE_URL,
    )


# Creating the client does not send a request. The connection is opened by
# call_llm when BELT needs a response.
LLM_CLIENT = create_llm_client()


# ============================================================
# History helpers
# ============================================================

def normalize_conversation(
    conversation: Optional[Sequence[Mapping[str, str]]],
) -> List[ConversationMessage]:
    """Return the latest four cleaned user turns and assistant responses."""
    if conversation is None:
        return []

    if isinstance(conversation, (str, bytes)):
        raise TypeError("conversation must be a sequence of message objects")

    cleaned: List[ConversationMessage] = []
    for index, message in enumerate(conversation):
        if not isinstance(message, Mapping):
            raise TypeError(
                f"conversation item {index} must be a message object"
            )

        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            raise ValueError(
                f"conversation item {index} has invalid role {role!r}"
            )
        if not isinstance(content, str):
            raise TypeError(
                f"conversation item {index} content must be a string"
            )

        content = content.strip()
        if content:
            cleaned.append({
                "role": role,
                "content": content,
            })

    user_indexes = [
        index
        for index, message in enumerate(cleaned)
        if message["role"] == "user"
    ]
    if len(user_indexes) > MAX_CONVERSATION_USER_INPUTS:
        cleaned = cleaned[
            user_indexes[-MAX_CONVERSATION_USER_INPUTS]:
        ]

    return cleaned


def remember_conversation_turn(
    conversation: List[ConversationMessage],
    user_input: str,
    assistant_response: str,
) -> None:
    """Store one completed turn and retain the latest four user inputs."""
    user_input = user_input.strip()
    assistant_response = assistant_response.strip()

    if user_input:
        conversation.append({
            "role": "user",
            "content": user_input,
        })
    if assistant_response:
        conversation.append({
            "role": "assistant",
            "content": assistant_response,
        })

    conversation[:] = normalize_conversation(conversation)


def print_llm_history(
    conversation: Sequence[Mapping[str, str]],
) -> None:
    """Print the conversation messages currently kept in memory."""
    recent_messages = normalize_conversation(conversation)
    if not recent_messages:
        print("No conversation messages are currently stored.")
        return

    print("\nRecent conversation:")
    for message in recent_messages:
        role = message["role"].capitalize()
        print(f"{role}: {message['content']}")
    print()


# ============================================================
# Main API function
# ============================================================

def call_llm(
    input_text: str,
    conversation: Optional[Sequence[Mapping[str, str]]] = None,
    debug: bool = False,
):
    """Send a prompt and recent conversation to the local BELT model."""
    input_text = input_text.strip()
    if not input_text:
        print("BELT API: Input text cannot be empty.")
        return None

    recent_messages = normalize_conversation(conversation)
    messages = [
        {"role": "system", "content": BELT_SYSTEM_PROMPT},
        *recent_messages,
        {"role": "user", "content": input_text},
    ]

    try:
        response = LLM_CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        if not response.choices:
            print("BELT API: The local model returned no choices.")
            return None

        output_text = response.choices[0].message.content
        if debug:
            print(f"Raw LLM response: {output_text!r}")
        if output_text is None or not output_text.strip():
            print("BELT API: The local model returned an empty response.")
            return None
        return output_text.strip()

    except AuthenticationError:
        print("BELT API: The local model server rejected the API key.")
    except RateLimitError:
        print(
            "BELT API: The local model server is busy. "
            "Please try again shortly."
        )
    except APIConnectionError:
        print(
            "BELT API: Could not connect to the local model server at "
            f"{LOCAL_LLM_BASE_URL}."
        )
    except APIStatusError as error:
        print(
            "BELT API: The local model server returned HTTP "
            f"{error.status_code}: {error}"
        )
    except Exception as error:
        print(f"BELT API: Unexpected local model error: {error}")

    return None


# ============================================================
# Terminal testing
# ============================================================

def main() -> None:
    print(
        f"BELT local LLM API loaded: {MODEL_NAME} at "
        f"{LOCAL_LLM_BASE_URL}"
    )
    print("Commands: /quit, /clear, /history\n")
    conversation: List[ConversationMessage] = []

    while True:
        text_input = input("You: ").strip()

        if text_input == "/quit":
            print("Closing BELT local LLM API.")
            break

        if text_input == "/clear":
            conversation.clear()
            print("BELT: Conversation history cleared.\n")
            continue

        if text_input == "/history":
            print_llm_history(conversation)
            continue

        response = call_llm(text_input, conversation=conversation)
        if response is not None:
            print(f"BELT: {response}\n")
            remember_conversation_turn(
                conversation,
                text_input,
                response,
            )


if __name__ == "__main__":
    main()
