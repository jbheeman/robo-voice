import cv2
import torch
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
          
          personal_items = ["cell phone", "keys", "wallet", "backpack", "cup"]
            if label in personal_items:
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
