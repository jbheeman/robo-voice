import cv2
import flet as ft
import base64
from ultralytics import YOLO
import pyttsx3
import time
import threading

model = YOLO("yolov8n.pt")

def speak_phrase(text):
    """Speaks the text by initializing a fresh TTS engine instance inside a background thread."""
    def _speak():
        try:
            # Re-initializing locally guarantees the audio doesn't fail
            local_engine = pyttsx3.init()
            local_engine.setProperty('rate', 150) #Moderate Speaking Pace
            local_engine.say(text)
            local_engine.runAndWait()
            
            del local_engine
        except Exception as tts_err:
            print(f"[TTS ERROR] Failed to output audio: {tts_err}")
    
    threading.Thread(target=_speak, daemon=True).start()

def main(page: ft.Page):
    page.title = "UCSC Campus Tour Guide"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.theme_mode = ft.ThemeMode.DARK

    # UI Element to display the camera frames
    image_display = ft.Image(
        src_base64="", 
        fit=ft.ImageFit.CONTAIN, 
        width=640, 
        height=480, 
        gapless_playback=True
    )

    # Status text
    status_text = ft.Text("Camera Starting...", size=16, weight=ft.FontWeight.BOLD)

    page.add(
        ft.Text("UCSC Campus Tour Guide", size=24, weight=ft.FontWeight.BOLD),
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
    reset_delay = 3.0  # Time (seconds) the frame must be completely clear of people

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Run YOLO inference on the current frame
            results = model(frame, stream=True, conf=0.5)

            human_detected = False
            annotated_frame = frame 

            for result in results:
                # Draw the bounding boxes and labels directly onto the frame
                annotated_frame = result.plot()
                
                # Check if a human is detected (Class ID 0 is 'person' in YOLO)
                if result.boxes is not None and len(result.boxes) > 0:
                    classes = result.boxes.cls.tolist()
                    if 0 in classes:
                        human_detected = True

            current_time = time.time()

            # Greet Once
            if human_detected:
                # Keep renewing the timestamp as long as a person is visible
                person_last_seen_time = current_time
                
                if not already_greeted:
                    print("[STATE] Person detected! Speaking greeting...")
                    speak_phrase("Welcome to the UCSC silicon valley extension")
                    already_greeted = True
            else:
                # No human detected in this frame
                if already_greeted:
                    time_since_last_seen = current_time - person_last_seen_time
                    
                    if time_since_last_seen > reset_delay:
                        print(f"[STATE] Frame clear for {time_since_last_seen:.1f}s. Resetting trigger.")
                        already_greeted = False
                    else:
                        print(f"[STATE] No human detected. Resetting in {reset_delay - time_since_last_seen:.1f}s...")

            # Encode the frame into JPEG format
            _, buffer = cv2.imencode('.jpg', annotated_frame)

            # Display
            base64_image = base64.b64encode(buffer).decode('utf-8')
            image_display.src_base64 = base64_image

            # Refresh the UI frame
            page.update()
            time.sleep(0.05)

    except Exception as e:
        print(f"Encountered an error: {e}")
    finally:
        # Safely release the camera when the application closes
        print("[INFO] Releasing camera resource...")
        cap.release()

# Run the Flet app
ft.app(target=main)
