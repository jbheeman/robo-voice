from __future__ import annotations

import json
from typing import Dict, List, Optional

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


def main() -> None:
    client = make_deepseek_client()
    memory = load_memory()

    current_user: Optional[str] = None
    pending_memory: Optional[Dict[str, Optional[str]]] = None
    waiting_for_name_to_save_memory = False

    conversation: List[Dict[str, str]] = []

    print("BELT v2 chatbot prototype online 🤖")
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
