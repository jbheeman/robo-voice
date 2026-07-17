from belt_v3_valid_movements import VALID_MOVEMENTS

def simple_action_handle(simple_action_list : list):
    for act in simple_action_list:
        if act not in VALID_MOVEMENTS:
            simple_action_list.remove(act)
            
    print(f"Simple action handle {simple_action_list}")
    
    
    
