import cv2
import numpy as np
import pathlib
import joblib
import os

from deepface import DeepFace

names = {
  0: "Bianca",
  1: "Ethan",
  2: "Hao",
  3: "Jason",
  4: "Jesh",
  5: "Lionel",
  6: "Michelle",
  7: "Tina",
  8: "Yi",
}

FACULTY_IMAGES_DIR = "faculty_images2"
ENCODINGS_PATH = "encodings.joblib"
MODEL_NAME = "Facenet"

DETECTOR_BACKEND = "retinaface"

MATCH_THRESHOLD = 0.60

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _folder_name_for(name):
    return name.replace(" ", "_")


def initEncodings():
    print("[INFO] Building faculty encodings...")
    encodings = {} 

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
        if face.get("face_confidence", 1.0) < 0.6:
            continue

        embedding = face["embedding"]

        best_index = None
        best_distance = None

        for i, known_embeddings in encodings.items():
            for known_embedding in known_embeddings:
                distance = _cosine_distance(embedding, known_embedding)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = i

        if best_index is not None:
            print(f"[DEBUG] Closest match: {names[best_index]} (distance={best_distance:.3f}, threshold={MATCH_THRESHOLD})")

        if best_index is not None and best_distance < MATCH_THRESHOLD:
            frame_names.append(names[best_index])

            area = face["facial_area"]
            top = area["y"]
            left = area["x"]
            bottom = area["y"] + area["h"]
            right = area["x"] + area["w"]
            recognized_locations.append([top, right, bottom, left])

    return frame_names, recognized_locations
