from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import joblib
from openai import OpenAI

from belt_v2_api import MODEL, make_deepseek_client, print_balance_status
from belt_v2_helpers import (
    SHOW_MEMORY_DETECTION,
    activate_user,
    ask_llm,
    detect_memory_candidate,
    detect_name_from_message,
    handle_command,
    parse_consent,
    same_user_name,
    save_pending_memory,
)
from belt_v2_memory import load_memory, save_memory

START_DEBUG = True

Intent = Literal[
    "command",
    "chat",
    "memory_retrieval",
    "memory_update",
    "movement",
]

INTENT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "train_intent_detector"
    / "belt_intent_router_best_logreg.joblib"
)

INTENT_CONFIDENCE_THRESHOLD = 0.80

VALID_MODEL_INTENTS = {
    "chat",
    "memory_retrieval",
    "memory_update",
    "movement",
}

MAX_HISTORY_MESSAGES = 10

# Turn this on if you want to debug what the intent router predicted.
SHOW_INTENT_ROUTING = False


@dataclass
class BeltState:
    """
    Temporary runtime state for one terminal session.

    This is NOT the saved long-term memory file.
    This just tracks what BELT needs while the program is running.
    """

    current_user: Optional[str] = None
    pending_memory: Optional[Dict[str, Optional[str]]] = None
    waiting_for_name_to_save_memory: bool = False
    conversation: List[Dict[str, str]] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Startup / routing helpers
# -----------------------------------------------------------------------------


def load_intent_model():
    return joblib.load(INTENT_MODEL_PATH)


def predict_intent(intent_model, user_input: str) -> tuple[str, float]:
    predicted_intent = intent_model.predict([user_input])[0]

    probabilities = intent_model.predict_proba([user_input])[0]
    classes = intent_model.classes_

    confidence_scores = dict(zip(classes, probabilities))
    confidence = float(confidence_scores[predicted_intent])

    return predicted_intent, confidence


def is_quit_command(user_input: str) -> bool:
    return user_input == "/quit"


def is_command(user_input: str) -> bool:
    """
    Commands are a deterministic terminal feature.

    They should not depend on the ML intent classifier because slash commands
    are easy to recognize and are mostly for testing/debugging.
    """

    return user_input.startswith("/")


def route_intent(intent_model, user_input: str) -> tuple[Intent, float]:
    """
    Convert raw user text into BELT's high-level route.

    Final routes:
      command          -> slash command; handled before this function normally
      chat             -> normal conversation
      memory_retrieval -> answer using saved memory
      memory_update    -> answer, then check whether something should be saved
      movement         -> future robot action; currently just a stub
    """

    if is_command(user_input):
        return "command", 1.0

    predicted_intent, confidence = predict_intent(intent_model, user_input)

    if confidence < INTENT_CONFIDENCE_THRESHOLD:
        return "chat", confidence

    if predicted_intent not in VALID_MODEL_INTENTS:
        raise ValueError(f"prediction {predicted_intent} with confidence {confidence} is not in listed intents")

    return predicted_intent, confidence  # type: ignore[return-value]


def print_startup_banner() -> None:
    print("BELT v2 chatbot prototype online 🤖")
    print(f"Using model: {MODEL}")
    print("Type /help for commands.\n")


def warn_if_no_balance() -> None:
    has_balance = print_balance_status()

    if not has_balance:
        print(
            "BELT: Warning: DeepSeek says your account may not have usable balance. "
            "You can still use commands like /active_user and /memory, but chatting with the model may fail.\n"
        )


# -----------------------------------------------------------------------------
# Session-state handlers
# -----------------------------------------------------------------------------


def append_to_conversation(
    state: BeltState,
    user_input: str,
    assistant_text: str,
) -> None:
    state.conversation.append({"role": "user", "content": user_input})
    state.conversation.append({"role": "assistant", "content": assistant_text})

    state.conversation = state.conversation[-MAX_HISTORY_MESSAGES:]


def update_active_user_from_message(
    memory: Dict[str, Any],
    state: BeltState,
    user_input: str,
) -> None:
    """
    Detect introductions like "Hi, I'm Bella" before the LLM reply.

    The detected name becomes active_user, but it is not saved as a memory note.
    The memory detector handles save-worthy personal notes separately.
    """

    introduced_name = detect_name_from_message(user_input)

    if not introduced_name:
        return

    activated_user = activate_user(memory, introduced_name)

    if not activated_user:
        return

    if state.current_user and not same_user_name(state.current_user, activated_user):
        state.conversation = []

    state.current_user = activated_user
    save_memory(memory)


def handle_pending_memory_owner(
    memory: Dict[str, Any],
    state: BeltState,
    user_input: str,
) -> bool:
    """
    Handles this situation:

    BELT found a memory.
    User said yes.
    But BELT does not know which active user should own that memory.
    """

    if not state.waiting_for_name_to_save_memory or not state.pending_memory:
        return False

    detected_name = detect_name_from_message(user_input) or user_input
    activated_user = activate_user(memory, detected_name)

    if not activated_user:
        print("BELT: I still need a simple name to save that under, like: I'm Bella.")
        return True

    state.current_user = activated_user
    state.pending_memory["detected_user_name"] = state.current_user

    save_pending_memory(
        memory=memory,
        pending_memory=state.pending_memory,
        fallback_user=state.current_user,
    )

    state.pending_memory = None
    state.waiting_for_name_to_save_memory = False
    state.conversation = []

    print(f"BELT: Got it — I'll remember that for {state.current_user}.")
    return True


def handle_pending_memory_consent(
    memory: Dict[str, Any],
    state: BeltState,
    user_input: str,
) -> bool:
    """
    Handles yes/no after BELT asks:

    "Do you want me to remember that ...?"
    """

    if not state.pending_memory:
        return False

    consent = parse_consent(user_input)

    if consent is True:
        saved_user = save_pending_memory(
            memory=memory,
            pending_memory=state.pending_memory,
            fallback_user=state.current_user,
        )

        if saved_user:
            state.current_user = activate_user(memory, saved_user) or saved_user
            save_memory(memory)

            state.pending_memory = None

            print(f"BELT: Got it — I'll remember that for {state.current_user}.")
            return True

        state.waiting_for_name_to_save_memory = True

        print("BELT: Sure — what name should I save that under? You can say: I'm Bella.")
        return True

    if consent is False:
        state.pending_memory = None

        print("BELT: No worries — I won't save that.")
        return True

    print("BELT: Should I save that memory? Please say yes or no.")
    return True


# -----------------------------------------------------------------------------
# Intent handlers
# -----------------------------------------------------------------------------


def reply_with_llm(
    client: OpenAI,
    user_input: str,
    memory: Dict[str, Any],
    state: BeltState,
    intent: Intent,
) -> Optional[str]:
    """
    Chat, memory retrieval, and memory update all use the same LLM call.

    Why this is okay:
      - ask_llm already receives current_user
      - ask_llm already receives saved memory
      - ask_llm already receives recent conversation

    The difference between routes is what happens after the reply.
    """

    return ask_llm(
        client=client,
        user_message=user_input,
        current_user=state.current_user,
        memory=memory,
        conversation=state.conversation,
    )


def handle_movement() -> None:
    """
    Placeholder for future robot movement.

    Later, this is where you would call motor/action code after safety checks.
    """

    print("BELT: Movement intent detected, but movement is not implemented yet.")


def maybe_start_memory_save_flow(
    client: OpenAI,
    memory: Dict[str, Any],
    state: BeltState,
    user_input: str,
    answer: str,
) -> str:
    """
    Runs only after a memory_update route.

    This does NOT immediately save the memory.
    It detects a candidate memory, asks the user for consent,
    and stores the candidate in state.pending_memory.
    """

    memory_candidate = detect_memory_candidate(
        client=client,
        user_message=user_input,
        assistant_reply=answer,
        current_user=state.current_user,
    )

    if not memory_candidate:
        return answer

    detected_name = memory_candidate.get("detected_user_name")
    memory_note = memory_candidate.get("memory_note")
    consent_question = memory_candidate.get("consent_question")

    if detected_name:
        activated_user = activate_user(memory, detected_name)

        if activated_user:
            state.current_user = activated_user
            memory_candidate["detected_user_name"] = state.current_user
            save_memory(memory)

    state.pending_memory = memory_candidate

    if SHOW_MEMORY_DETECTION:
        display_user = state.current_user if state.current_user else "Unknown"

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

    if consent_question:
        print(f"BELT: {consent_question}\n")
        return f"{answer}\n{consent_question}"

    return answer


def handle_chat_like_intent(
    client: OpenAI,
    memory: Dict[str, Any],
    state: BeltState,
    user_input: str,
    intent: Intent,
) -> None:
    """
    Handles these routes:
      chat
      memory_retrieval
      memory_update

    Chat directly calls LLM
    memory_retrieval might use RAG
    memory_update has extra post-processing afterward.
    """

    answer = reply_with_llm(
        client=client,
        user_input=user_input,
        memory=memory,
        state=state,
        intent=intent,
    )

    if answer is None:
        return

    print(f"BELT: {answer}\n")

    assistant_history_text = answer

    if intent == "memory_update":
        assistant_history_text = maybe_start_memory_save_flow(
            client=client,
            memory=memory,
            state=state,
            user_input=user_input,
            answer=answer,
        )

    append_to_conversation(
        state=state,
        user_input=user_input,
        assistant_text=assistant_history_text,
    )


def handle_routed_input(
    client: OpenAI,
    intent_model,
    memory: Dict[str, Any],
    state: BeltState,
    user_input: str,
) -> None:
    """
    Main router for a single user message.
    """

    intent, confidence = route_intent(intent_model, user_input)
    if START_DEBUG:
        print(intent, confidence)

    if SHOW_INTENT_ROUTING:
        print(f"[router] intent={intent}, confidence={confidence:.2f}")

    if intent == "command":
        handle_command(user_input, memory, state.current_user)
        return

    if intent == "movement":
        handle_movement()
        return

    handle_chat_like_intent(
        client=client,
        memory=memory,
        state=state,
        user_input=user_input,
        intent=intent,
    )


# -----------------------------------------------------------------------------
# Main program loop
# -----------------------------------------------------------------------------


def main() -> None:
    client = make_deepseek_client()
    intent_model = load_intent_model()
    memory = load_memory()

    state = BeltState()

    print_startup_banner()
    warn_if_no_balance()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if is_quit_command(user_input):
            print("BELT: Goodbye! Powering down politely.")
            break

        # 1. Slash commands are deterministic and do not use the intent model.
        if is_command(user_input):
            handle_command(user_input, memory, state.current_user)
            continue

        # 2. Pending memory flows take priority over normal chat.
        #    Example: BELT asked "Should I remember that?" and now expects yes/no.
        if handle_pending_memory_owner(memory, state, user_input):
            continue

        if handle_pending_memory_consent(memory, state, user_input):
            continue

        # 3. Detect active user before replying, so BELT can use the name naturally.
        update_active_user_from_message(memory, state, user_input)

        # 4. Route the message into chat / memory_retrieval / memory_update / movement.
        handle_routed_input(
            client=client,
            intent_model=intent_model,
            memory=memory,
            state=state,
            user_input=user_input,
        )


if __name__ == "__main__":
    main()