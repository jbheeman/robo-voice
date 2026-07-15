#check IMPORTANT

import pyttsx3
import time
import threading 
import cv2
import flet as ft
import base64
from ultralytics import YOLO

# Automatically downloads on first run
model = YOLO("yolov8n.pt")

# Initialize TTS Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Moderate speaking pace


def speak_phrase(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()
    
    threading.Thread(target=_speak, daemon=True).start()

def main(page: ft.Page):
    page.title = "Real-Time YOLO Object Detector"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.theme_mode = ft.ThemeMode.DARK

    # UI Element to display the camera frames
    # Provide an empty string to src_base64 to prevent Flet from crashing at initialization
    image_display = ft.Image(
        src_base64="", 
        fit=ft.ImageFit.CONTAIN, 
        width=640, 
        height=480,
        gapless_playback=True  # Keeps the video stream playing smoothly without flickering
    )
    
    # Status text
    status_text = ft.Text("Camera Starting...", size=16, weight=ft.FontWeight.BOLD)

    page.add(
        ft.Text("YOLO Object Detection Camera", size=24, weight=ft.FontWeight.BOLD),
        image_display,
        status_text
    )

    # Opens the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        status_text.value = "Error: Could not open camera."
        page.update()
        return

    status_text.value = "Streaming Active • Detecting Objects"
    page.update()

    already_greeted = False
    person_last_seen_time = 0.0
    reset_delay = 3.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Run YOLO on current frame
            results = model(frame, stream=True)

            # Check if a human is detected in the current frame
            human_detected = False

            for result in results:
                # Draw bounding boxes
                annotated_frame = result.plot()
            
                if result.boxes is not None:
                    classes = result.boxes.cls.tolist()
                    if 0 in classes:  # 0 is 'person'
                        human_detected = True

            current_time = time.time()

            if human_detected:
                # Update the last seen timestamp
                person_last_seen_time = current_time
            
            # If we haven't greeted this person yet, do it now
                if not already_greeted:
                    speak_phrase("Welcome to the UCSC silicon valley extension")
                    already_greeted = True
            else:
            # If no human is in the frame, check if enough time has passed to reset
                if already_greeted and (current_time - person_last_seen_time > reset_delay):
                    already_greeted = False  # Ready to greet the next person who walks up
            
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            
            # Convert the buffer to base64 bytes to display in the Flet UI
            base64_image = base64.b64encode(buffer).decode('utf-8')
            image_display.src_base64 = base64_image
            
            # Refresh the UI frame
            page.update()

    except Exception as e:
        print(f"Encountered an error: {e}")
    finally:
        # Safely release the camera when the app is closed
        cap.release()

# Run the app
ft.app(target=main)
