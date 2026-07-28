# Robot camera setup

`human_det.py` reads the computer's local webcam through OpenCV by default.
Robot ROS 2 support remains available for later, but it is not selected unless
`--camera-source ros` is passed explicitly.

When the program runs inside WSL, the local-camera data path is:

```text
Windows camera -> usbipd -> WSL /dev/video0 -> OpenCV
               -> YOLO and staff recognition
```

## 1. Attach the Windows camera to WSL

Open **PowerShell as Administrator**. The `usbipd` program is installed at:

```text
C:\Program Files\usbipd-win\usbipd.exe
```

List the USB devices:

```powershell
& "C:\Program Files\usbipd-win\usbipd.exe" list
```

On this computer the built-in camera is BUSID `2-1`. Share it:

```powershell
& "C:\Program Files\usbipd-win\usbipd.exe" bind --busid 2-1
```

Then attach it to WSL:

```powershell
& "C:\Program Files\usbipd-win\usbipd.exe" attach --wsl --busid 2-1
```

Close Windows Camera, Zoom, Teams, and other programs that may be using the
camera before attaching it.

Back in the WSL terminal, verify that the device exists:

```bash
ls -l /dev/video*
```

## 2. Activate the existing vision environment

From the workspace root:

```bash
cd "/root/git/UCSC AI SUMMER CAMP"
source venv/bin/activate
```

This environment already contains OpenCV, Ultralytics, InsightFace, ONNX
Runtime, Joblib, NumPy, and pyttsx3.

Test whether OpenCV can open the camera:

```bash
python -c "import cv2; c=cv2.VideoCapture(0); print('Camera opened:', c.isOpened()); c.release()"
```

## 3. Run the detector

Run from the computer-vision directory so its model, encodings, and faculty
image paths resolve correctly:

```bash
cd "/root/git/UCSC AI SUMMER CAMP/robo-voice/belt_v3/comp_vision"
python human_det.py
```

When using SSH without graphical forwarding:

```bash
python human_det.py --no-display
```

Stop a headless run with `Ctrl+C`.

The expected camera message is similar to:

```text
[INFO] Waiting for camera frames from local webcam index 0...
[INFO] Camera connected: local webcam index 0 (1280x720).
```

## Robot camera later

Robot support must be requested explicitly:

```bash
source /opt/ros/jazzy/setup.bash
python human_det.py \
  --camera-source ros \
  --camera-topic /camera/camera/color/image_raw
```
