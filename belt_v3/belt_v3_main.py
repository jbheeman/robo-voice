import time
from movement.belt_v3_simple_action_handle import simple_action_handle
from speech.belt_v3_speech_handle import speech_handle, testing_speech_handle
from navigation.belt_v3_navigation_handle import navigation_handle
from belt_v3_api import ConversationMessage, remember_conversation_turn
from belt_v3_helper import compose_response
from belt_v3_input import get_input, terminal_get_input
from launch_streamlit import start_streamlit, stop_streamlit

#hyperparams? idk
DEBUG = True
USING_ROBOT = False
LAUNCH_STREAMLIT = False
VOICE = "Aiden"

PROGRAM_START_TIME = time.perf_counter() if DEBUG else 0.0

# Holds the latest 10 user inputs and BELT's corresponding speech responses.
conversation: list[ConversationMessage] = []


def generate_response(
    text_input: str,
    conversation: list[ConversationMessage],
):
    if DEBUG:
        request_start = time.perf_counter()

    output, rag_context = compose_response(
        text_input,
        conversation,
        debug=DEBUG,
    )  # Python dictionary
    
    if DEBUG:
        total_time = time.perf_counter() - request_start
        print(f"Total response time ({total_time:.3f} seconds)")
    #     print("Rag context:")
    #     print(rag_context)
    #     print("Structured response output:")
    #     print(output)
    
    return output


def execute_modules(response_output: dict):
    if USING_ROBOT:
        speech_handle(response_output["speech"], VOICE)
    else:
        testing_speech_handle(response_output["speech"], VOICE)
        
    if response_output["simple_action"]["requested"]:
        simple_action_handle(response_output["simple_action"]["actions"])
        
    if response_output["navigation"]["requested"]:
        navigation_handle(response_output["navigation"]["locations"])


def main():
    dashboard_process = None

    if LAUNCH_STREAMLIT:
        try:
            dashboard_process = start_streamlit()
        except RuntimeError as error:
            print(f"BELT could not start the audio dashboard: {error}")
            return

    try:
        if DEBUG:
            startup_time = time.perf_counter() - PROGRAM_START_TIME
            print(f"Done starting up ({startup_time:.3f} seconds)")

        while True:
            if USING_ROBOT == False:
                text_input = terminal_get_input()
            else:
                text_input = get_input()

            # One LLM call returns speech, navigation, and simple actions.
            response_output = generate_response(
                text_input,
                conversation,
            )

            # Store the real user input and the response BELT will speak.
            remember_conversation_turn(
                conversation,
                text_input,
                response_output["speech"],
            )

            #execute modules based on request
            execute_modules(response_output)
    except KeyboardInterrupt:
        print("\nBELT stopped.")
    finally:
        stop_streamlit(dashboard_process)


if __name__ == "__main__":
    main()
