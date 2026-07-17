import face_recognition
import cv2
import numpy as np
import pathlib
import joblib

# Known Gen AI faculty names
# https://genai.ucsc.edu/people/
names = {
  0: "Luca De Alfaro",
  1: "Pranav Anand",
  2: "Manel Camps",
  3: "Ashesh Kumar Chattopadhyay",
  4: "Jason Eshraghian",
  5: "Daniel J. Fremont",
  6: "Jeffrey M Flanigan",
  7: "Leilani H Gilpin",
  8: "David Haussler",
  9: "Minghui Hu",
  10: "Tae Myung Huh",
  11: "Ian Lane",
  12: "Bing Liu",
  13: "Alexander Ioannidis",
  14: "Nilah Ioannidis",
  15: "Heiner H Litz",
  16: "Razvan V Marinescu",
  17: "Jennifer A Parker",
  18: "Jose Renau",
  # "Sridhar Rao", # No image
  19: "Magy Seif El-Nasr",
  20: "Sagnik Nath",
  21: "Michael Tassio",
  22: "Chenguang Wang",
  23: "Xiao Wang",
  # "Zhu Wang", # No image
  24: "Cihang Xie",
  25: "Hao Yue",
  26: "Yi Zhang",
  27: "Yuyin Zhou",
  # "Zac Zimmer" # Image has no face
}

# Initializes encodings
def initEncodings():

    # Initializes empty encodings list
    encodings = []

    # Goes through images labeled 0-27 in faculty_images
    for i in range(28):

        # Creates path
        path = pathlib.Path("faculty_images") / f"{i}.jpg"

        # Loads image
        image = face_recognition.load_image_file(path)

        # Adds image encoding to encodings list
        encodings.append(face_recognition.face_encodings(image)[0])

    # Dumps encodings with joblib
    joblib.dump(encodings, "encodings.joblib")

# Runs encoding initializer (use once)
initEncodings()

# Gets image encodings
encodings = joblib.load("encodings.joblib")

# Returns list of recognized people and positions from cv2 image
def getPeople(frame):
    
    # Resizes frame
    frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)

    # Converts cv2 BGR format to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Finds faces and encodings in frame
    frame_locations = face_recognition.face_locations(frame)
    frame_encodings = face_recognition.face_encodings(frame, frame_locations)

    # Stores found names and recognized locations
    frame_names = []
    recognized_locations = []

    # Loops through encodings in frame to find matches
    for i in range(len(frame_encodings)):

        # Get frame encoding
        frame_encoding = frame_encodings[i]
        
        # Finds distances to known encodings
        distances = face_recognition.face_distance(encodings, frame_encoding)

        # Finds index of closest encoding
        index = np.argmin(distances)

        # Checks if encoding is within a threshold
        if distances[index] < 0.55:
            
            # Adds name to name list
            frame_names.append(names[index])

            # Scales location by scaling factor
            location = frame_locations[i]
            location = [coord * 4 for coord in location]

            # Adds location to location list
            recognized_locations.append(location)

    # Returns frame_names and recognized_locations
    return frame_names, recognized_locations
