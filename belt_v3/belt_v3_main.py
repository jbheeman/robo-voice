import joblib
from movement.belt_v3_simple_action_handle import simple_action_handle
from speech.belt_v3_speech_handle import speech_handle
from navigation.belt_v3_navigation_handle import navigation_handle
from belt_v3_helper import extract_nav_action, compose_response

CHAT_CHECKER_MODEL = joblib.load(
    "chat_checker_model.joblib"
)

#hyperparams? idk
DEBUG = False
CHAT_THRESHOLD = 0.99

def get_input():
    return input("> ").strip()

def chat_checker(text_input: str):
    prob = CHAT_CHECKER_MODEL.predict_proba([text_input])[0][1]
    
    if (DEBUG):
        print("chat_checker probability: ", prob)

    return prob
    

def request_extractor(text_input: str, chat_prob: float):
    nav_action_dict = {
    "simple_action": {
        "requested": False,
        "actions": []
    },
    "navigation": {
        "requested": False,
        "locations": []
    }}
    
    if chat_prob < CHAT_THRESHOLD:
        nav_action_dict = extract_nav_action(text_input)
        
    output, rag_context = compose_response(nav_action_dict, text_input) #python dict 
    
    if DEBUG:
        print("Rag context:")
        print(rag_context)
        print("Request Extractor output:")
        print(output)
    
    return output


def execute_modules(extractor_output: dict):
    speech_handle(extractor_output["speech"])
    if extractor_output["simple_action"]["requested"]:
        simple_action_handle(extractor_output["simple_action"]["actions"])
        
    if extractor_output["navigation"]["requested"]:
        navigation_handle(extractor_output["navigation"]["locations"])


def main():
    while True:
        #for now this text input is just simple input from terminal
        #later we will change to handle robot audio
        text_input = get_input()
        
        #checks if navigation or simple_action or none
        #request is a python dictionary, ex: {"navigation": 0, "simple_action":1}
        chat_prob = chat_checker(text_input)
        
        #handles info based on request
        #returns speech text, navigation dict, simple_action dict
        extractor_output = request_extractor(text_input, chat_prob)
        
        #execute modules based on request
        execute_modules(extractor_output)


if __name__ == "__main__":
    main()