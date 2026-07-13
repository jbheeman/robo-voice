pip install flet opencv-python ultralytics

import cv2
import flet as ft
import base64
from ultralytics import YOLO

# Automatically downloads on first run
model = YOLO("yolov8n.pt")

def main(page: ft.Page):
    page.title = "Real-Time YOLO Object Detector"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.theme_mode = ft.ThemeMode.DARK

    # UI Element to display the camera frames
    image_display = ft.Image(fit=ft.ImageFit.CONTAIN, width=640, height=480)
    
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

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Run YOLO inference on the current frame
            # stream: efficient memory usage
            results = model(frame, stream=True)

            for result in results:
                # Draw the bounding boxes and labels directly onto the frame
                annotated_frame = result.plot()

            # Encode the frame into JPEG format
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
