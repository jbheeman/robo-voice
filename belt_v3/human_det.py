"""
BELT greeter: YOLO watches for people, DeepFace identifies them, TTS greets

  * Recognized staff  -> "Hello, {name}!"
  * Unrecognized face -> "Welcome"

Design notes:
  * YOLO runs every frame (cheap). Face recognition runs at most once every
    RECOGNITION_INTERVAL seconds and only when YOLO sees a person -- otherwise
    retinaface + Facenet would drag the feed down to a few FPS. Between passes
    the last known boxes are drawn, so the overlay still looks live
  * Greetings are driven ONLY by fresh recognition results. "Welcome" fires on
    a detected *face* that matched nobody -- not merely on a YOLO person box --
    so a person's back or an arm in frame won't trigger it
  * All speech goes through one worker thread + queue. Spawning a pyttsx3
    engine per utterance (the old approach) deadlocks or drops audio when two
    greetings overlap

Press 'q' to quit, 'd' to toggle debug distances, 'r' to reset greeting state
"""

import queue
import threading
import time

import cv2
import pyttsx3
from ultralytics import YOLO

from staff_recognition import KNOWN_NAMES, detect_faces

CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.5
PERSON_CLASS_ID = 0

GENERIC_GREETING = "Welcome"
STAFF_GREETING_TEMPLATE = "Hello, {name}!"

# Lower = snappier greetings
# Higher = smoother video.
RECOGNITION_INTERVAL = 0.60

# How long a recognition result stays on screen before it is considered stale.
RESULT_TTL = 1.5

# Someone must be gone this long before they can be greeted again.
PRESENCE_TIMEOUT = 8.0

# Hard floor between two greetings of the same person, even if they flicker
# in and out of frame.
GREETING_COOLDOWN = 15.0

SHOW_YOLO_BOXES = False  # the person boxes are noisy; face boxes are enough

COLOR_KNOWN = (0, 200, 0)
COLOR_UNKNOWN = (0, 165, 255)


class Speaker:
    def __init__(self, rate=165):
        self._queue = queue.Queue()
        self._rate = rate
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
        except Exception as exc:
            print(f"[TTS ERROR] Could not initialize speech engine: {exc}")
            return

        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"[TTS ERROR] {exc}")
            finally:
                self._queue.task_done()

        try:
            engine.stop()
        except Exception:
            pass

    def say(self, text):
        if self._queue.qsize() < 3:
            self._queue.put(text)

    def shutdown(self):
        self._stop.set()
        self._queue.put(None)



class GreetingTracker:
    """
    Tracks who is currently 'present'. A subject is greeted when they appear
    after being absent for PRESENCE_TIMEOUT, and never more often than
    GREETING_COOLDOWN.
    """

    def __init__(self):
        self.last_seen = {}
        self.last_greeted = {}
        self.present = set()

    def observe(self, subject, now):
        was_present = subject in self.present
        self.last_seen[subject] = now
        self.present.add(subject)

        if was_present:
            return False
        if now - self.last_greeted.get(subject, -1e9) < GREETING_COOLDOWN:
            return False

        self.last_greeted[subject] = now
        return True

    def expire(self, now):
        for subject in list(self.present):
            if now - self.last_seen.get(subject, 0) > PRESENCE_TIMEOUT:
                self.present.discard(subject)

    def reset(self):
        self.last_seen.clear()
        self.last_greeted.clear()
        self.present.clear()


def draw_faces(frame, faces, show_debug):
    for face in faces:
        top, right, bottom, left = face["box"]
        known = face["name"] is not None
        color = COLOR_KNOWN if known else COLOR_UNKNOWN
        label = face["name"] if known else "Visitor"
        if show_debug:
            label += f" ({face['distance']:.2f}/m{face['margin']:.2f})"

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = max(top - 8, th + 4)
        cv2.rectangle(frame, (left, ly - th - 6), (left + tw + 8, ly + 4), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def draw_hud(frame, fps, person_count, stale):
    lines = [
        f"FPS {fps:4.1f} | people {person_count} | enrolled {len(KNOWN_NAMES)}",
        "q quit   d debug   r reset",
    ]
    if stale:
        lines[0] += "  [scanning...]"
    for i, text in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame



def main():
    print(f"[INFO] Enrolled staff ({len(KNOWN_NAMES)}): {', '.join(KNOWN_NAMES) or 'none'}")
    if not KNOWN_NAMES:
        print("[WARN] Nobody is enrolled -- everyone will get the generic greeting.")

    model = YOLO(YOLO_MODEL)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    speaker = Speaker()
    staff = GreetingTracker()
    visitors = GreetingTracker()

    cached_faces = []
    last_recognition_at = 0.0
    last_result_at = 0.0
    show_debug = False
    fps, fps_frames, fps_since = 0.0, 0, time.time()

    print("[INFO] Streaming. Press 'q' in the video window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Dropped frame from camera.")
                continue

            now = time.time()

            # Is there a person at all?
            results = model.predict(frame, conf=YOLO_CONF, classes=[PERSON_CLASS_ID], verbose=False)
            person_count = sum(len(r.boxes) for r in results if r.boxes is not None)
            human_detected = person_count > 0

            display = frame.copy()
            if SHOW_YOLO_BOXES and human_detected:
                for r in results:
                    for box in r.boxes.xyxy.cpu().numpy().astype(int):
                        cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (200, 200, 200), 1)

            # Who is it?
            fresh_faces = None
            if human_detected and (now - last_recognition_at) >= RECOGNITION_INTERVAL:
                last_recognition_at = now
                fresh_faces = detect_faces(frame, verbose=show_debug)
                cached_faces = fresh_faces
                last_result_at = now
            elif not human_detected:
                # Nobody in frame: clear the overlay immediately.
                cached_faces = []

            stale = bool(cached_faces) and (now - last_result_at) > RESULT_TTL
            if stale:
                cached_faces = []

            display = draw_faces(display, cached_faces, show_debug)

            # Greetings
            if fresh_faces is not None:
                named = [f["name"] for f in fresh_faces if f["name"]]
                unknown_faces = [f for f in fresh_faces if f["name"] is None]

                for name in dict.fromkeys(named):  
                    if staff.observe(name, now):
                        print(f"[GREET] Staff recognized: {name}")
                        speaker.say(STAFF_GREETING_TEMPLATE.format(name=name))

                # One generic greeting regardless of how many unkown faces are in the shot.
                if unknown_faces:
                    if visitors.observe("__visitor__", now):
                        print(f"[GREET] Unrecognized visitor(s): {len(unknown_faces)}")
                        speaker.say(GENERIC_GREETING)

            staff.expire(now)
            visitors.expire(now)

            fps_frames += 1
            if now - fps_since >= 0.5:
                fps = fps_frames / (now - fps_since)
                fps_frames, fps_since = 0, now

            display = draw_hud(display, fps, person_count, human_detected and not cached_faces)
            cv2.imshow("BELT", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_debug = not show_debug
                print(f"[INFO] Debug overlay {'on' if show_debug else 'off'}")
            if key == ord("r"):
                staff.reset()
                visitors.reset()
                print("[INFO] Greeting state reset.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    except Exception as exc:
        import traceback
        print(f"[ERROR] {exc}")
        traceback.print_exc()
    finally:
        print("[INFO] Releasing camera...")
        speaker.shutdown()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
