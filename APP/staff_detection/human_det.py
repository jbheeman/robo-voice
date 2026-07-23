import cv2
import time
import threading
from ultralytics import YOLO
import pyttsx3
import torch
from staff_recognition import getPeople

model = YOLO("yolov8n.pt")

# Automatically select the best available device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model.to(device)
GENERIC_GREETING = "Welcome to the UCSC silicon valley extension"
STAFF_GREETING_TEMPLATE = "Hello, {name}!"

STAFF_COOLDOWN = 3.0
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


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("[INFO] Streaming active. Press 'q' in the video window to quit.")

    already_greeted = False
    person_last_seen_time = 0.0

    last_seen = {}
    currently_greeted = set()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            small_frame = cv2.resize(frame, (640, 480))
            results = model(small_frame, stream=True, conf=0.5)

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

            if human_detected:
                frame_names, recognized_locations = getPeople(frame)

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

            for name in list(currently_greeted):
                if current_time - last_seen.get(name, 0) > STAFF_COOLDOWN:
                    currently_greeted.discard(name)

            # Generic greeting for unrecognized visitors
            unrecognized_human_present = human_detected and len(frame_names) == 0

            if unrecognized_human_present:
                person_last_seen_time = current_time
                if not already_greeted:
                    print("[STATE] Unrecognized visitor detected! Speaking generic greeting...")
                    speak_phrase(GENERIC_GREETING)
                    already_greeted = True
            else:
                if already_greeted:
                    time_since_last_seen = current_time - person_last_seen_time
                    if time_since_last_seen > GENERIC_RESET_DELAY:
                        already_greeted = False

            cv2.imshow("BELT", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Encountered an error: {e}")
    finally:
        print("[INFO] Releasing camera resource...")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
