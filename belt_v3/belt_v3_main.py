import os
import time

PROGRAM_START_TIME = time.perf_counter()
DEBUG = True
os.environ["BELT_DEBUG"] = "1" if DEBUG else "0"

# Choose which language-model backend BELT uses.
USE_DEEPSEEK_API = False

if USE_DEEPSEEK_API:
    from belt_v3_api import (
        ConversationMessage,
        call_llm,
        remember_conversation_turn,
    )
else:
    from belt_v3_new_api import (
        ConversationMessage,
        call_llm,
        remember_conversation_turn,
    )

from movement.belt_v3_simple_action_handle import (
    DEFAULT_COOLDOWN_SECONDS,
    simple_action_handle,
)
from speech.belt_v3_speech_handle import speech_handle, testing_speech_handle
from speech.belt_v3_qwen_tts import (
    preload_qwen_model,
    tts_configuration_summary,
)
from navigation.belt_v3_navigation_handle import navigation_handle
from belt_v3_helper import (
    combine_spoken_parts,
    compose_response,
    get_optional_cv_state,
    prepare_vision_context_for_llm,
    print_timing,
    print_timing_summary,
    speak_output,
)
from belt_v3_input import get_input, terminal_get_input
from comp_vision.belt_v3_cv import close_cv, get_cv_state

# Runtime configuration
USING_ROBOT = True
VOICE = "Vivian"
BELT_WAKE_WORD = False
START_HARNESS = True

# Holds the latest 4 user inputs and BELT's complete spoken responses.
conversation: list[ConversationMessage] = []


def generate_response(
    text_input: str,
    conversation: list[ConversationMessage],
    cv_state: dict | None = None,
    timing_metrics: dict[str, float] | None = None,
) -> dict:
    request_started_at = time.perf_counter()

    output, _rag_context = compose_response(
        text_input,
        conversation,
        vision_context=cv_state,
        debug=DEBUG,
        timing_metrics=timing_metrics,
        llm_caller=call_llm,
    )

    if DEBUG:
        print(
            f"[VALIDATED OUTPUT LIST] {output['output_list']}",
            flush=True,
        )

    print_timing(
        "Response generation total",
        request_started_at,
        debug=DEBUG,
    )
    return output


def execute_modules(
    response_output: dict,
    timing_metrics: dict[str, float],
) -> str:
    """Execute the validated output list from first event to last."""
    execution_started_at = time.perf_counter()
    pending_spoken_parts: list[str] = []
    spoken_sections: list[str] = []
    robot_action_was_executed = False

    for index, event in enumerate(
        response_output["output_list"],
        start=1,
    ):
        event_type = event["type"]

        if event_type == "speech":
            pending_spoken_parts.append(event["text"])
            continue

        if event_type == "navigation":
            navigation_started_at = time.perf_counter()
            navigation_directions = navigation_handle(
                event["location"]
            )
            if navigation_directions:
                pending_spoken_parts.append(
                    navigation_directions
                )
            print_timing(
                f"Navigation event {index}",
                navigation_started_at,
                debug=DEBUG,
            )
            continue

        if event_type == "action":
            pending_speech = combine_spoken_parts(
                pending_spoken_parts
            )
            if pending_speech:
                speak_output(
                    pending_speech,
                    timing_metrics,
                    using_robot=USING_ROBOT,
                    voice=VOICE,
                    speech_handler=speech_handle,
                    testing_speech_handler=testing_speech_handle,
                    debug=DEBUG,
                )
                spoken_sections.append(pending_speech)
                pending_spoken_parts.clear()

            action_started_at = time.perf_counter()
            if USING_ROBOT:
                if robot_action_was_executed:
                    time.sleep(DEFAULT_COOLDOWN_SECONDS)
                simple_action_handle([event["name"]])
                robot_action_was_executed = True
            else:
                print(
                    f"[SIMULATED ACTION] {event['name']}",
                    flush=True,
                )
            print_timing(
                f"Action event {index}",
                action_started_at,
                debug=DEBUG,
            )

    pending_speech = combine_spoken_parts(pending_spoken_parts)
    if pending_speech:
        speak_output(
            pending_speech,
            timing_metrics,
            using_robot=USING_ROBOT,
            voice=VOICE,
            speech_handler=speech_handle,
            testing_speech_handler=testing_speech_handle,
            debug=DEBUG,
        )
        spoken_sections.append(pending_speech)

    print_timing(
        "Module execution total",
        execution_started_at,
        debug=DEBUG,
    )
    return combine_spoken_parts(spoken_sections)


def main() -> None:
    try:
        if not USING_ROBOT:
            print(
                "[MODE] Terminal simulation: robot microphone, camera, "
                "speech output, and gestures are disabled.",
                flush=True,
            )
        print(tts_configuration_summary(VOICE))

        if USING_ROBOT:
            qwen_started_at = time.perf_counter()
            preload_qwen_model(VOICE)
            print_timing(
                "Qwen TTS model preload",
                qwen_started_at,
                debug=DEBUG,
            )

        print_timing(
            "Done starting up, including imports and model loading",
            PROGRAM_START_TIME,
            debug=DEBUG,
        )

        if USING_ROBOT and START_HARNESS:
            harness_started_at = time.perf_counter()
            simple_action_handle(["harness"])
            print_timing(
                "Startup harness movement",
                harness_started_at,
                debug=DEBUG,
            )

        while True:
            timing_metrics = {
                "input_handle": 0.0,
                "cv": 0.0,
                "rag": 0.0,
                "llm_response": 0.0,
                "output_audio": 0.0,
                "turn_total": 0.0,
            }
            turn_started_at = time.perf_counter()
            input_started_at = time.perf_counter()

            if not USING_ROBOT:
                text_input = terminal_get_input()
                cv_state = None
            else:
                text_input = get_input(
                    require_wake_word=BELT_WAKE_WORD,
                )

            timing_metrics["input_handle"] = (
                time.perf_counter() - input_started_at
            )
            input_source = "Audio" if USING_ROBOT else "Terminal"
            print_timing(
                f"{input_source} input wait",
                input_started_at,
                debug=DEBUG,
            )

            processing_started_at = time.perf_counter()
            if USING_ROBOT:
                cv_state = get_optional_cv_state(
                    timing_metrics,
                    get_cv_state,
                    debug=DEBUG,
                )

            if DEBUG:
                print("[CV_STATE]", flush=True)
                print(
                    prepare_vision_context_for_llm(cv_state),
                    flush=True,
                )

            response_output = generate_response(
                text_input,
                conversation,
                cv_state=cv_state,
                timing_metrics=timing_metrics,
            )

            spoken_response = execute_modules(
                response_output,
                timing_metrics,
            )
            remember_conversation_turn(
                conversation,
                text_input,
                spoken_response,
            )

            print_timing(
                "Turn processing after input",
                processing_started_at,
                debug=DEBUG,
            )
            print_timing(
                "Complete turn including input wait",
                turn_started_at,
                debug=DEBUG,
            )
            timing_metrics["turn_total"] = (
                time.perf_counter() - turn_started_at
            )
            print_timing_summary(timing_metrics, debug=DEBUG)
    except KeyboardInterrupt:
        print("\nBELT stopped.")
    finally:
        if USING_ROBOT:
            close_cv()


if __name__ == "__main__":
    main()
