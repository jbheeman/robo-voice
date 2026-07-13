

def get_input():
    text_input = input()
    return text_input

def request_router(text_input: str):
    pass

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