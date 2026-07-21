import cv2
import numpy as np
import pathlib
import joblib
import os

from deepface import DeepFace

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
  19: "Magy Seif El-Nasr",
  20: "Sagnik Nath",
  21: "Michael Tassio",
  22: "Chenguang Wang",
  23: "Xiao Wang",
  24: "Cihang Xie",
  25: "Hao Yue",
  26: "Yi Zhang",
  27: "Yuyin Zhou",
}

ENCODINGS_PATH = "encodings.joblib"

MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"

MATCH_THRESHOLD = 0.40


def _cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def initEncodings():
    """Builds face embeddings from faculty_images/{i}.jpg and saves them."""
    encodings = []
    for i in range(28):
        path = str(pathlib.Path("faculty_images") / f"{i}.jpg")
        try:
            result = DeepFace.represent(
                img_path=path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
            )
            # DeepFace.represent returns a list (one entry per face found);
            # faculty photos should have exactly one face.
            encodings.append(result[0]["embedding"])
        except Exception as e:
            print(f"[WARN] Could not encode faculty_images/{i}.jpg ({names.get(i)}): {e}")
            encodings.append(None)

    joblib.dump(encodings, ENCODINGS_PATH)


# Only rebuild encodings if they don't already exist on disk.
if not os.path.exists(ENCODINGS_PATH):
    initEncodings()

encodings = joblib.load(ENCODINGS_PATH)


def getPeople(frame):
    frame_names = []
    recognized_locations = []

    try:
        faces = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend='retinaface',
            enforce_detection=False,
        )
    except Exception as e:
        print(f"[WARN] DeepFace.represent failed on frame: {e}")
        return frame_names, recognized_locations

    for face in faces:
        if face.get("face_confidence", 1.0) == 0:
            continue

        embedding = face["embedding"]

        best_index = None
        best_distance = None
        for i, known_embedding in enumerate(encodings):
            if known_embedding is None:
                continue
            distance = _cosine_distance(embedding, known_embedding)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = i

        if best_index is not None and best_distance < MATCH_THRESHOLD:
            frame_names.append(names[best_index])

            area = face["facial_area"]
            top = area["y"]
            left = area["x"]
            bottom = area["y"] + area["h"]
            right = area["x"] + area["w"]
            recognized_locations.append([top, right, bottom, left])

    return frame_names, recognized_locations
