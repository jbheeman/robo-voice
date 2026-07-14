import joblib
from belt_v3_api import call_llm
from belt_v3_helper import safely_parse_json_to_python_dict, extract_nav_action

CHAT_CHECKER_MODEL = joblib.load(
    "chat_checker_model.joblib"
)

#hyperparams? idk
DEBUG = True
CHAT_THRESHOLD = 0.8

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
        
    output = response_composer(nav_action_dict, text_input) #python dict 
    return output


def execute_modules(extractor_output: dict):
    pass


def main():
    while True:
        #for now this text input is just simple input from terminal
        #later we will change to handle robot audio
        text_input = get_input()
        if (text_input == "/quit"): break #add this to break the code, will not be in actual robot tho
        
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