import cv2
import flet as ft
import base64
from ultralytics import YOLO
import pyttsx3
import time
import threading
import face_recognition
import cv2
import numpy as np
import pathlib
import joblib
import os

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


def initEncodings():
    #Buids face encodings and saves them
    encodings = []
    for i in range(28):
        path = pathlib.Path("faculty_images") / f"{i}.jpg"
        image = face_recognition.load_image_file(path)
        encodings.append(face_recognition.face_encodings(image)[0])
    joblib.dump(encodings, ENCODINGS_PATH)


# Only rebuild encodings if they don't already exist
if not os.path.exists(ENCODINGS_PATH):
    initEncodings()

encodings = joblib.load(ENCODINGS_PATH)


def getPeople(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    frame_locations = face_recognition.face_locations(rgb_frame)
    frame_encodings = face_recognition.face_encodings(rgb_frame, frame_locations)

    frame_names = []
    recognized_locations = []

    for i in range(len(frame_encodings)):
        frame_encoding = frame_encodings[i]
        distances = face_recognition.face_distance(encodings, frame_encoding)
        index = np.argmin(distances)

        if distances[index] < 0.55:
            frame_names.append(names[index])
            location = frame_locations[i]
            location = [coord * 4 for coord in location]
            recognized_locations.append(location)

    return frame_names, recognized_locations


model = YOLO("yolov8n.pt")

GENERIC_GREETING = "Welcome to the UCSC silicon valley extension"
STAFF_GREETING_TEMPLATE = "Hello, {name}!"

STAFF_COOLDOWN = 3.0    # seconds a staff member must be absent before re-greeting
GENERIC_RESET_DELAY = 3.0 


def speak_phrase(text):
    def _speak():
        try:
            local_engine = pyttsx3.init()
            local_engine.setProperty('rate', 150)
            local_engine.say(text)
            local_engine.runAndWait()
            del local_engine
        except Exception as tts_err:
            print(f"[TTS ERROR] Failed to output audio: {tts_err}")

    threading.Thread(target=_speak, daemon=True).start()


def main(page: ft.Page):
    page.title = "BELT tour guide"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.theme_mode = ft.ThemeMode.DARK

    image_display = ft.Image(
        src_base64="",
        fit=ft.ImageFit.CONTAIN,
        width=640,
        height=480,
        gapless_playback=True
    )

    status_text = ft.Text("Camera Starting...", size=16, weight=ft.FontWeight.BOLD)

    page.add(
        ft.Text("Belt Tour Guide", size=24, weight=ft.FontWeight.BOLD),
        image_display,
        status_text
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        status_text.value = "Error: Could not open camera."
        page.update()
        return

    status_text.value = "Streaming Active"
    page.update()

    # unrecognized greeting
    already_greeted = False
    person_last_seen_time = 0.0

    # staff member greeting
    last_seen = {}            
    currently_greeted = set()  

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, stream=True, conf=0.5)

            human_detected = False
            annotated_frame = frame

            for result in results:
                annotated_frame = result.plot()
                if result.boxes is not None and len(result.boxes) > 0:
                    classes = result.boxes.cls.tolist()
                    if 0 in classes:
                        human_detected = True

            current_time = time.time()
            frame_names = []

            # Only run face recognition when a human is actually present
            if human_detected:
                frame_names, recognized_locations = getPeople(frame)

                # Draw name labels for recognized staff on top of bounding boxes
                for name, loc in zip(frame_names, recognized_locations):
                    top, right, bottom, left = loc
                    cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 200, 0), 2)
                    cv2.putText(annotated_frame, name, (left, max(top - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

            # Staff greeting
            for name in frame_names:
                last_seen[name] = current_time
                if name not in currently_greeted:
                    print(f"[STATE] Staff member detected: {name}. Greeting...")
                    speak_phrase(STAFF_GREETING_TEMPLATE.format(name=name))
                    currently_greeted.add(name)

            # Expire staff who've been gone longer than the cooldown
            for name in list(currently_greeted):
                if current_time - last_seen.get(name, 0) > STAFF_COOLDOWN:
                    currently_greeted.discard(name)

            # Greeting for people who arent identified staff members
            unrecognized_human_present = human_detected and len(frame_names) == 0

            if unrecognized_human_present:
                person_last_seen_time = current_time
                if not already_greeted:
                    print("Unrecognized")
                    speak_phrase(GENERIC_GREETING)
                    already_greeted = True
            else:
                if already_greeted:
                    time_since_last_seen = current_time - person_last_seen_time
                    if time_since_last_seen > GENERIC_RESET_DELAY:
                        already_greeted = False

            # Encode + display
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            image_display.src_base64 = base64_image
            page.update()
            time.sleep(0.05)

    except Exception as e:
        print(f"Encountered an error: {e}")
    finally:
        print("Releasing camera")
        cap.release()


ft.app(target=main)
