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
from belt_v3_helper import compose_response, prepare_vision_context_for_llm
from belt_v3_input import get_input, terminal_get_input
from launch_streamlit import start_streamlit, stop_streamlit
from comp_vision.belt_v3_cv import close_cv, get_cv_state

# Runtime configuration
USING_ROBOT = False
LAUNCH_STREAMLIT = False
VOICE = "Vivian"
BELT_WAKE_WORD = False

# Holds the latest 4 user inputs and BELT's complete spoken responses.
conversation: list[ConversationMessage] = []


def print_timing(label: str, started_at: float) -> None:
    if DEBUG:
        elapsed = time.perf_counter() - started_at
        print(f"[TIMING] {label}: {elapsed:.3f}s", flush=True)


def print_timing_summary(timings: dict[str, float]) -> None:
    if not DEBUG:
        return

    measured_total = sum(
        timings.get(name, 0.0)
        for name in (
            "input_handle",
            "cv",
            "rag",
            "llm_response",
            "output_audio",
        )
    )
    print(
        "[TIMING SUMMARY] "
        f"input_handle={timings.get('input_handle', 0.0):.3f}s | "
        f"cv={timings.get('cv', 0.0):.3f}s | "
        f"rag={timings.get('rag', 0.0):.3f}s | "
        f"llm_response={timings.get('llm_response', 0.0):.3f}s | "
        f"output_audio={timings.get('output_audio', 0.0):.3f}s | "
        f"measured_total={measured_total:.3f}s | "
        f"turn_total={timings.get('turn_total', 0.0):.3f}s",
        flush=True,
    )


def get_optional_cv_state(
    timing_metrics: dict[str, float],
) -> dict | None:
    """Prevent any optional CV failure from stopping the conversation."""
    cv_started_at = time.perf_counter()

    try:
        return get_cv_state()
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        print(f"[CV ERROR] {error}", flush=True)
        print("CV state is not working, cv_state=None", flush=True)
        return None
    finally:
        timing_metrics["cv"] = time.perf_counter() - cv_started_at
        print_timing("CV request total", cv_started_at)


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

    print_timing("Response generation total", request_started_at)
    return output


def combine_spoken_response(
    speech: str,
    navigation_directions: str,
) -> str:
    """Join the planner response and directions into one natural utterance."""
    speech = " ".join(speech.split())
    navigation_directions = " ".join(
        navigation_directions.split()
    )

    if not navigation_directions:
        return speech
    if not speech:
        return navigation_directions

    separator = " " if speech.endswith((".", "!", "?")) else ". "
    return f"{speech}{separator}{navigation_directions}"


def speak_output(
    text: str,
    timing_metrics: dict[str, float],
) -> None:
    """Speak one buffered section and accumulate its audio-output time."""
    speech_started_at = time.perf_counter()
    if USING_ROBOT:
        speech_handle(text, VOICE)
    else:
        testing_speech_handle(text, VOICE)

    timing_metrics["output_audio"] = (
        timing_metrics.get("output_audio", 0.0)
        + time.perf_counter()
        - speech_started_at
    )
    print_timing("Speech output", speech_started_at)


def combine_spoken_parts(parts: list[str]) -> str:
    """Combine adjacent speech and navigation text into one utterance."""
    combined = ""
    for part in parts:
        combined = combine_spoken_response(combined, part)
    return combined


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
            )
            continue

        if event_type == "action":
            pending_speech = combine_spoken_parts(
                pending_spoken_parts
            )
            if pending_speech:
                speak_output(pending_speech, timing_metrics)
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
            )

    pending_speech = combine_spoken_parts(pending_spoken_parts)
    if pending_speech:
        speak_output(pending_speech, timing_metrics)
        spoken_sections.append(pending_speech)

    print_timing("Module execution total", execution_started_at)
    return combine_spoken_parts(spoken_sections)


def main() -> None:
    dashboard_process = None

    if LAUNCH_STREAMLIT:
        dashboard_started_at = time.perf_counter()
        try:
            dashboard_process = start_streamlit()
            print_timing("Streamlit dashboard startup", dashboard_started_at)
        except RuntimeError as error:
            print(f"BELT could not start the audio dashboard: {error}")
            return

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
            print_timing("Qwen TTS model preload", qwen_started_at)

        print_timing(
            "Done starting up, including imports and model loading",
            PROGRAM_START_TIME,
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
            print_timing(f"{input_source} input wait", input_started_at)

            processing_started_at = time.perf_counter()
            if USING_ROBOT:
                cv_state = get_optional_cv_state(timing_metrics)

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
            )
            print_timing(
                "Complete turn including input wait",
                turn_started_at,
            )
            timing_metrics["turn_total"] = (
                time.perf_counter() - turn_started_at
            )
            print_timing_summary(timing_metrics)
    except KeyboardInterrupt:
        print("\nBELT stopped.")
    finally:
        if USING_ROBOT:
            close_cv()
        stop_streamlit(dashboard_process)


if __name__ == "__main__":
    main()
