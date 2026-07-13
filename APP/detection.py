import cv2
import torch
pip install ultralytics
#YOLO is a fairly reliable industry standard framework dedicated to RGB detection
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format = "engine")


def process_camera_frame(rgb_frame, depth_frame):
  results = model(rgb_frame, conf=0.5)

  for result in results:
    boxes = result.boxes
    for box in boxes:

      x1, y1, x2, y2 = map(int, box.xyxy[0])
      class_id = int(box.cls[0])
      label = model.names[class_id]

      #list of the items we want the robot to identify
      personal_items = ["cell phone", "backpack", "cup", "bottle", "handbag", "suitcase", "laptop"]
      detected_objects = []
      if label in personal_items:

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        object_depth = depth_frame[center_y, center_x] / 1000.0

        print(f"Detected {label} at pixels ({center_x}, {center_y}), Distance: {object_depth:.2f} meters")

        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb_frame, f"{label}: {object_depth:.2f}m", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #Appends all crucial spacial data of the detected objects
        detected_objects.append({
          "label": label,
          "center_x": center_x,
          "center_y": center_y,
          "depth": object_depth
        })
  #returns the annotated frame with the list of detected objects and the spacial data of the objects
  return rgb_frame, detected_objects

cap = cv2.VideoCapture(0)

if not cap.isOpened():
  print("Error: Could not open the camera.")
  exit()

print("Press 'q' to quit the live detection.")

while True:
    # 2. Capture frame-by-frame from the camera
  ret, bgr_frame = cap.read()
    
    # If the frame wasn't grabbed successfully, break the loop
  if not ret:
    print("Error: Failed to grab frame.")
    break
    
    # Convert from BGR to RGB format
  rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
  height, width, _ = rgb_frame.shape

    # Create a depth map
  depth_frame = np.full((height, width), 1000, dtype=np.uint16)
    
  detection_results = process_camera_frame(rgb_frame, depth_frame)
