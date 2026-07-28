# Every room corresponds to a character, which was made using ASCII.
# 2250: '?' and 2230: '<' were changed to '7', which 2210 maps to.
# This is to group the rooms together.
# Room 0 is the entrance/exit, and is not an actual room.
symbols = {0: '/', 2004: '0', 2005: '1', 2007: '2', 2013: '3', 2015: '4', 2017: '5', 2019: '6', 
           2110: '7', 2115: '8', 2117: '9', 2119: ':', 2125: ';', 2130: '7', 2135: '=', 
           2145: '>', 2150: '7', 2155: '@', 2165: 'A', 2175: 'B', 2204: 'C', 2205: 'D', 
           2206: 'E', 2210: 'F', 2215: 'G', 2220: 'H', 2221: 'I', 2225: 'J', 2227: 'K', 
           2231: 'L', 2235: 'M', 2240: 'N', 2250: 'O', 2255: 'P', 2260: 'Q', 2300: 'R'}

# This map is made based on the symbols above, with # representing a wall.
map = '''########################
#################0######
##############?<7    /##
################# ## ###
#                 ## ###
#B#A#@#>#=#;## ## ## ###
########### #: 8# ## ###
########### ## #3 ## ###
########### #9 ## ## ###
########### ##### #2   #
########### ###5# ## #1#
########### ##6   ## ###
########### ###4# ## ###
########### ##### ## ###
###########          ###
############ ##E#C# ####
###########F ###### D###
############ #####G ####
###########H ###### ####
############ #####I ####
############ ###### J###
###########N #####K ####
############ ###### ####
###########O #####L ####
############ Q##### M###
############ ####P# ####
############          ##
#####################R##
########################'''

# Turns map into 2D array
map = map.split("\n")
map = [list(row) for row in map]

# Represents what coords to increment by for each direction.
# Turning left and right are equivalent to adding 1 and -1 to to the index, respectively.
dirs = ((0, 1), (-1, 0), (0, -1), (1, 0))

# Calculates path to navigate from the start to the end.
# Output is a list of dirs.
def general_nav(start, end):
    start = symbols[start]
    end = symbols[end]
    if start == end:
        return general_format([])
    if start == '7':
        return nav(end)
    for x in range(29):
        row = map[x]
        if start in row:
            scoords = (x, row.index(start))
        if end in row:
            ecoords = (x, row.index(end))
    visited = {}
    queue = []
    qidx = 0
    visited[ecoords] = 0
    queue.append(ecoords)
    for depth in range(1, 999):
        while visited[queue[qidx]] < depth:
            node = queue[qidx]
            qidx += 1
            for d in range(4):
                new = (node[0] + dirs[d][0], node[1] + dirs[d][1])
                if map[new[0]][new[1]] != '#':
                    if new not in visited:
                        if new == scoords:
                            out = []
                            for depth2 in range(depth - 1, -1, -1):
                                for d2 in range(4):
                                    new2 = (new[0] + dirs[d2][0], new[1] + dirs[d2][1])
                                    if new2 in visited:
                                        if visited[new2] == depth2:
                                            out.append(d2)
                                            new = new2
                                            break
                            return general_format(start, out)
                        visited[new] = depth
                        queue.append(new)

# Takes in list of dirs and outputs text navigation instructions.
def general_format(start, dlist):
    if len(dlist) == 0:
        return "You have arrived!"
    out = []
    if start == '7':
        out.append("First, exit through the door closest to the building entrance.")
    elif start == 'D':
        out.append("First, exit through the door to the right of the whiteboard.")
    elif start == ';':
        if dlist[0] == 1:
            out.append("First, exit through the doorway to the left of the microwaves.")
        else:
            out.append("First, exit through the doorway to the right of the microwaves.")
    else:
        out.append("First, exit through the door.")
    # Convert to changes in direction.
    ddlist = []
    for i in range(len(dlist) - 1):
        ddlist.append((dlist[i + 1] - dlist[i]) % 4)
    # Convert to list of tuples of (direction change, # of units forward)
    i = 0
    j = 1
    while j < len(ddlist):
        if ddlist[j] != 0:
            rot = ddlist[i]
            dist = j - 1
            if rot == 1:
                out.append("Turn left.")
            elif rot == 3:
                out.append("Turn right.")
            if dist < 3:
                out.append("Go forward slightly.")
            else:
                if False: # Check if will reach end of hall.
                    out.append("Go to the end of the hall.")
                else:
                    out.append("Go forward.") # Edit to include how far to go.
            i = j
        j += 1
    rot = ddlist[i]
    dist = j - i
    if dist < 2:
        if rot == 1:
            out.append("Your destination will be on the left.")
        elif rot == 3:
            out.append("Your destination will be on the right.")
        else:
            out.append("You destination will be in front of you.")
    else:
        if rot == 1:
            out.append("Turn left.")
        elif rot == 3:
            out.append("Turn right.")
        out.append("Go all the way forward.")
        out.append("You destination will be in front of you.")
    return " ".join(out)

def nav(room_num):
    out = []
    if room_num in [2110, 2130, 2150]:
        return format(out)
    out.append("First, exit through the door closest to the building entrance.")
    if room_num in [2004]:
        return format(out)
    if room_num in [0, 2005, 2007]:
        out.append("Go forward.")
        if room_num in [0]:
            return format(out)
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
# rooms = list(symbols.keys())
# start = random.choice(rooms)
# end = random.choice(rooms)
# print(start, end, general_nav(start, end))
