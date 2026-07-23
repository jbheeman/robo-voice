import cv2
import numpy as np
import pathlib
import joblib
import os

from deepface import DeepFace

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

# Folder containing one subfolder per person
FACULTY_IMAGES_DIR = "faculty_images2"
ENCODINGS_PATH = "encodings.joblib"
MODEL_NAME = "Facenet"

DETECTOR_BACKEND = "retinaface"

MATCH_THRESHOLD = 0.40

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _folder_name_for(name):
    # "Jose Renau" -> "Jose_Renau" (matches the extracted zip's folder names)
    return name.replace(" ", "_")


def initEncodings():
    """
    Builds one embedding per training image (not per person) and stores them
    grouped by person index, so each person can have a variable number of
    reference photos (2, 3, or more).
    """
    print("[INFO] Building faculty encodings...")
    encodings = {}  # person_index -> list of embeddings

    for i, name in names.items():
        person_dir = pathlib.Path(FACULTY_IMAGES_DIR) / _folder_name_for(name)
        person_embeddings = []

        if not person_dir.is_dir():
            print(f"[WARN] No folder found for {name} at {person_dir}")
            encodings[i] = person_embeddings
            continue

        image_paths = sorted(
            p for p in person_dir.iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS
        )

        if not image_paths:
            print(f"[WARN] No images found in {person_dir}")

        for img_path in image_paths:
            try:
                result = DeepFace.represent(
                    img_path=str(img_path),
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                )
                if result and len(result) > 0:
                    person_embeddings.append(result[0]["embedding"])
                else:
                    print(f"[WARN] No face found in {img_path}")
            except Exception as e:
                print(f"[WARN] Could not encode {img_path} ({name}): {e}")

        encodings[i] = person_embeddings
        print(f"[INFO] {name}: {len(person_embeddings)} embedding(s) built")

    joblib.dump(encodings, ENCODINGS_PATH)
    print("[INFO] Encodings saved to disk.")


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
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
    except Exception as e:
        print(f"[WARN] DeepFace.represent failed on frame: {e}")
        return frame_names, recognized_locations

    for face in faces:
        # Ignore weak detections
        if face.get("face_confidence", 1.0) < 0.6:
            continue

        embedding = face["embedding"]

        best_index = None
        best_distance = None

        # Compare against every stored embedding for every person, and keep
        # whichever single reference photo is closest (nearest-neighbor 
        # matching across all training images)
        for i, known_embeddings in encodings.items():
            for known_embedding in known_embeddings:
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
