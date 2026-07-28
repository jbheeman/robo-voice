def nav(room_num):
    out = []
    if room_num in [2110, 2130, 2150]:
        return format(out)
    out.append("First, exit through the door closest to the building entrance.")
    if room_num in [2004]:
        return format(out)
    if room_num in [2005, 2007]:
        out.append("Go forward.")
        out.append("Turn right.")
        out.append("Go forward.")
        if room_num in [2005]:
            out.append("Turn left.")
        return format(out)
    out.append("Turn right.")
    if room_num in [2115, 2117, 2119, 2125, 2135, 2145, 2155, 2165, 2175]:
        out.append("Turn right.")
        if room_num in [2115, 2117, 2119]:
            out.append("Turn left.")
        else:
            out.append("Go down the hall until you see your room, which will be on your left.")
        return format(out)
    if room_num in [2013]:
        out.append("Go forward until you see your room, which will be on your right.")
        return format(out)
    if room_num in [2015, 2017, 2019]:
        out.append("Go forward until you see the bathrooms on your right.")
        return format(out)
    out.append("Go to the end of the hallway.")
    if room_num in [2204]:
        return format(out)
    if room_num in [2206, 2210, 2220, 2240, 2250, 2260]:
        out.append("Turn right.")
        if room_num in [2206]:
            out.append("Go down the hall until you see your room, which will be on your left.")
            return format(out)
        out.append("Go down the hall and turn right.")
        if room_num in [2260]:
            out.append("Go down the hall until you see your room, which will be on your left.")
        elif room_num in [2210, 2220]:
            out.append("Go forward until you see your room, which will be on your right.")
        else:
            out.append("Go down the hall until you see your room, which will be on your right.")
        return format(out)
    out.append("Turn left.")
    out.append("Turn right.")
    if room_num in [2255, 2300]:
        out.append("Go to the end of the hall.")
        if room_num in [2255]:
            out.append("Turn right.")
        else:
            out.append("Turn left.")
        out.append("Turn right.")
        return format(out)
    if room_num in [2231, 2227]:
        out.append("Go down the hall until you see your room, which will be on your right.")
    elif room_num in [2215, 2221]:
        out.append("Go forward until you see your room, which will be on your right.")
    elif room_num in [2205]:
        out.append("Turn left.")
    else:
        out.append("Go down the hall until you see your room, which will be on your left.")
    return format(out)

def format(out):
    out.append("You have arrived!")
    return " ".join(out)

# import random
# rooms = [2004, 2005, 2007, 2013, 2015, 2017, 2019, 2110, 2115, 2117, 2119, 2125, 2130, 2135, 2145, 2150, 2155, 2165, 2175, 2204, 2205, 2206, 2210, 2215, 2220, 2221, 2225, 2227, 2231, 2235, 2240, 2250, 2255, 2260, 2300]
# for _ in range(10):
#     room = random.choice(rooms)
#     print(room, nav(room))

# room = 2117
# print(room, nav(room))