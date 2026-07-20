from .belt_v3_valid_navigation import VALID_LOCATIONS

def navigation_handle(navigation_list : list):
    for nav in navigation_list:
        if nav.lower() not in VALID_LOCATIONS:
            navigation_list.remove(nav)
            
    print(f"Navigation handle: {navigation_list}")
