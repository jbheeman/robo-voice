import os
import time

PROGRAM_START_TIME = time.perf_counter()
DEBUG = True
os.environ["BELT_DEBUG"] = "1" if DEBUG else "0"

from movement.belt_v3_simple_action_handle import simple_action_handle
from speech.belt_v3_speech_handle import speech_handle, testing_speech_handle
from speech.belt_v3_qwen_tts import (
    preload_qwen_model,
    tts_configuration_summary,
)
from navigation.belt_v3_navigation_handle import navigation_handle
from belt_v3_api import ConversationMessage, remember_conversation_turn
from belt_v3_helper import compose_response, prepare_vision_context_for_llm
from belt_v3_input import get_input, terminal_get_input
from launch_streamlit import start_streamlit, stop_streamlit
from comp_vision.belt_v3_cv import close_cv, get_cv_state

#hyperparams? idk
USING_ROBOT = False
LAUNCH_STREAMLIT = False
VOICE = "Uncle_Fu"
BELT_WAKE_WORD = False

# Holds the latest 4 user inputs and BELT's corresponding speech responses.
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
):
    request_started_at = time.perf_counter()

    output, rag_context = compose_response(
        text_input,
        conversation,
        vision_context=cv_state,
        debug=DEBUG,
        timing_metrics=timing_metrics,
    )  # Python dictionary
    
    print_timing("Response generation total", request_started_at)
    #     print("Rag context:")
    #     print(rag_context)
    #     print("Structured response output:")
    #     print(output)
    
    return output


def main():
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

            if USING_ROBOT == False:
                text_input = terminal_get_input()
                cv_state = None
                timing_metrics["input_handle"] = (
                    time.perf_counter() - input_started_at
                )
                print_timing("Terminal input wait", input_started_at)
            else:
                text_input = get_input(
                    require_wake_word=BELT_WAKE_WORD,
                )
                timing_metrics["input_handle"] = (
                    time.perf_counter() - input_started_at
                )
                print_timing("Audio input wait", input_started_at)

                processing_started_at = time.perf_counter()
                cv_state = get_optional_cv_state(timing_metrics)

            if DEBUG:
                print("[CV_STATE]", flush=True)
                print(
                    prepare_vision_context_for_llm(cv_state),
                    flush=True,
                )

            stream_response, rag_context = compose_response_stream(
                text_input,
                conversation,
                debug=DEBUG,
                vision_context=cv_state,
                timing_metrics=timing_metrics,
            )

            full_response_text = ""
            llm_started_at = time.perf_counter()

            if stream_response is not None:
                for chunk in stream_response:
                    content = chunk.choices[0].delta.content if hasattr(chunk, "choices") else None
                    if content:
                        full_response_text += content
                    # Stream directly to audio / TTS buffer here:
                    # speech_handle_stream(content)

                timing_metrics["llm_response"] = time.perf_counter() - llm_started_at

                validated_response = _validated_llm_response(full_response_text)

                if validated_response["simple_action"]["requested"]:
                    simple_action_handle(validated_response["simple_action"]["actions"])

                if validated_response["navigation"]["requested"]:
                    navigation_handle(validated_response["navigation"]["locations"])

                remember_conversation_turn(
                    conversation,
                    text_input,
                    validated_response["speech"],
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
        close_cv()
        stop_streamlit(dashboard_process)


if __name__ == "__main__":
    main()
