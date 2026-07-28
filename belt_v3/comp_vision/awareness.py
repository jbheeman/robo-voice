# CHECK IMPORTANT #
import cv2
import numpy as np
import time
from ultralytics import YOLO
# Import Unitree's SDK2 locomotion client
from unitree_sdk2py.go2.sport.sport_client import SportClient

model = YOLO("yolov8n.pt") 

# Target classes we specifically want to recognize and steer around
AVOID_CLASSES = ["person", "chair", "backpack", "suitcase", "fire hydrant"]

# Initialize G1
sport_client = SportClient()
sport_client.SetTimeout(10.0)
sport_client.Init()

# Configurable Thresholds
SAFE_DISTANCE_MM = 1000       # 1.0 meter wall detection cushion
TRIGGER_PIXELS_THRESHOLD = 500  # Minimum pixels to trigger a "wall" warning

# To avoid staircases / depth drops
EXPECTED_FLOOR_DISTANCE_MAX = 1600 
CLIFF_PIXEL_THRESHOLD = 300

def main():
    cap_color = cv2.VideoCapture(0)  # RGB
    cap_depth = cv2.VideoCapture(1)  # Depth 
    
    if not cap_color.isOpened() or not cap_depth.isOpened():
        print("Could not open camera")
        return

    print("System active")

    try:
        while True:
            ret_c, color_frame = cap_color.read()
            ret_d, depth_frame = cap_depth.read()
            
            if not ret_c or not ret_d:
                continue

            # Wall avoidance
            depth_data = np.asanyarray(depth_frame)
            height, width = depth_data.shape[:2]

            # Split depth into Left, Center, and Right zones
            third_width = width // 3
            left_zone = depth_data[:, 0:third_width]
            center_zone = depth_data[:, third_width:2*third_width]
            right_zone = depth_data[:, 2*third_width:width]

            # Count pixels closer than 1.0m (ignoring 0 values)
            left_close_pixels = np.sum((left_zone > 0) & (left_zone < SAFE_DISTANCE_MM))
            center_close_pixels = np.sum((center_zone > 0) & (center_zone < SAFE_DISTANCE_MM))
            right_close_pixels = np.sum((right_zone > 0) & (right_zone < SAFE_DISTANCE_MM))

            bottom_zone = depth_data[int(height * 0.75):height, :]
            
            # Count pixels that are too far away (floor dropped)
            cliff_pixels = np.sum((bottom_zone > EXPECTED_FLOOR_DISTANCE_MAX) | (bottom_zone == 0))

            # Object avoidance
            results = model(color_frame, verbose=False)
            boxes = results[0].boxes
            
            yolo_left_danger = False
            yolo_right_danger = False
            yolo_center_danger = False

            # Analyze YOLO bounding boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label in AVOID_CLASSES:
                    x_min, _, x_max, _ = box.xyxy[0].tolist()
                    box_center_x = (x_min + x_max) / 2.0
                    relative_x = box_center_x / color_frame.shape[1]

                    if relative_x < 0.35:
                        yolo_left_danger = True
                    elif relative_x > 0.65:
                        yolo_right_danger = True
                    else:
                        yolo_center_danger = True

            # Decision logic
            vx, vy, yaw = 0.1, 0.0, 0.0 
            
            # Critical priority: Dont fall down a staircase
            if cliff_pixels > CLIFF_PIXEL_THRESHOLD:
                print("Drop detected, redirecting")
                vx, vy, yaw = -0.15, 0.0, 0.5  # Back up slightly and spin to find flat ground

            # Priority: Wall/Object ahead
            elif (center_close_pixels > TRIGGER_PIXELS_THRESHOLD) or yolo_center_danger:
                print("Object detected, redirecting")
                vx, vy, yaw = 0.0, 0.0, 0.4  # Stop and spin left to find a clear path
            
            # 2nd Priority: Left side is blocked
            elif (left_close_pixels > TRIGGER_PIXELS_THRESHOLD) or yolo_left_danger:
                print("[⬅️ LEFT BLOCKED] Steering Right...")
                vx, vy, yaw = 0.08, 0.0, -0.3
            
            # 3rd Priority: Right side is blocked
            elif (right_close_pixels > TRIGGER_PIXELS_THRESHOLD) or yolo_right_danger:
                print("[➡️ RIGHT BLOCKED] Steering Left...")
                vx, vy, yaw = 0.08, 0.0, 0.3

            # Send the final evaluated movement command to the G1
            sport_client.Move(vx, vy, yaw)
            time.sleep(0.1) 

    except KeyboardInterrupt:
        print("\nEmergency Stop Triggered by User. Halting G1...")
        sport_client.Move(0.0, 0.0, 0.0)
    finally:
        cap_color.release()
        cap_depth.release()

if __name__ == "__main__":
    main()
