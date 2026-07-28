# Robot camera setup

`human_det.py` now reads the robot's RealSense color stream through ROS 2 by
default. It does not try to open `/dev/video0`.

The data path is:

```text
robot RealSense -> ROS 2 Image topic -> CvBridge -> OpenCV frame
                 -> YOLO and staff recognition
```

Only the color stream is needed for human and face recognition. The depth
stream documented in `HOW_TO_ACCESS_CAMERA.txt` is not used by this program.

## 1. Verify the robot camera stream

On a computer connected to the robot's network:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic info /camera/camera/color/image_raw
ros2 topic hz /camera/camera/color/image_raw
```

The topic must report at least one publisher and a changing frame rate.

If the robot camera driver is running locally and the topic is absent, start
the standard RealSense ROS driver:

```bash
ros2 launch realsense2_camera rs_launch.py
```

If the driver runs on another computer, both computers must be on the same
network and use compatible ROS domain and DDS settings.

## 2. Use ROS's Python version

ROS Jazzy on the robot computer uses Python 3.12. Do not run the detector from
a Python 3.13 Conda environment.

Source ROS before activating the Python environment:

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
cd ~/robo-voice
source ros_venv/bin/activate
```

Verify the important imports:

```bash
python -c "import rclpy, cv2, cv_bridge; print('camera imports OK')"
```

If this reports that a module built with NumPy 1.x cannot run with NumPy 2.x,
the environment needs a NumPy version below 2:

```bash
python -m pip install --force-reinstall "numpy<2"
```

The detector also requires the packages listed in the repository's main
`README.md`, including Ultralytics, InsightFace, ONNX Runtime, Joblib, and
pyttsx3.

## 3. Run the detector

Run from the computer-vision directory so its model, encodings, and faculty
image paths resolve correctly:

```bash
cd ~/robo-voice/belt_v3/comp_vision
python human_det.py
```

When using SSH without graphical forwarding:

```bash
python human_det.py --no-display
```

Stop a headless run with `Ctrl+C`.

The expected startup messages are:

```text
[INFO] Waiting for camera frames from ROS 2 topic /camera/camera/color/image_raw...
[INFO] Camera connected: ROS 2 topic /camera/camera/color/image_raw (1280x720).
```

## Other camera inputs

Use a different ROS image topic:

```bash
python human_det.py --camera-topic /another/color/image_raw
```

Use a local webcam instead of the robot:

```bash
python human_det.py --camera-source webcam --camera-index 0
```

If the expected ROS topic has no publisher, the program waits 15 seconds and
then prints the image topics that are actually available.
