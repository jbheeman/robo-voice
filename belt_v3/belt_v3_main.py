import joblib

REQUEST_ROUTER_MODEL = joblib.load(
    "request_router_model.joblib"
)

DEBUG = False

def get_input():
    return input("> ").strip()

def request_router(text_input: str):
    prediction = REQUEST_ROUTER_MODEL.predict([text_input])[0]
    probabilities = REQUEST_ROUTER_MODEL.predict_proba([text_input])[0]
    output = {"navigation":prediction[0], "simple_action":prediction[1]}
    
    if (DEBUG):
        print("CURRENT FUNCTION: request_router")
        print(f"Prediction: navigation={prediction[0]}, simple_action={prediction[1]}")
        print(f"Confidence: navigation={probabilities[0]:.2%}, simple_action={probabilities[1]:.2%}")

    return output
    

def request_extractor(text_input: str, request: dict):
    pass

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
        request = request_router(text_input)
        
        #handles info based on request
        #returns speech text, navigation dict, simple_action dict
        extractor_output = request_extractor(text_input, request)
        
        #execute modules based on request
        execute_modules(extractor_output)


if __name__ == "__main__":
    main()