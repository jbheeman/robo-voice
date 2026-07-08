import cv2
import torch
#YOLO is a fairly reliable industry standard framework dedicated to RGB detection
from ultralytics import YOLO
model = YOLO("yolov8n.engine")

def process_camera_frame(rgb_frame, depth_frame):
  results = model(rgb_frame, conf=0.5)
  
  for result in results:
      boxes = result.boxes
      for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        label = model.names[class_id]

          #list of the items we want the robot to identify
        personal_items = ["cell phone", "keys", "wallet", "backpack", "cup"]
        if label in personal_items:
                
          center_x = int((x1 + x2) / 2)
          center_y = int((y1 + y2) / 2)
          object_depth = depth_frame[center_y, center_x] / 1000.0 
                
          print(f"Detected {label} at pixels ({center_x}, {center_y}), Distance: {object_depth:.2f} meters")
              
          cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
          cv2.putText(rgb_frame, f"{label}: {object_depth:.2f}m", (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
              


