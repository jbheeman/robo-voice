import cv2
import flet as ft
import base64
from ultralytics import YOLO
import pyttsx3
import time
import threading

from staff_recognition import getPeople

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
