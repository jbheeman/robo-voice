"""
Face enrollment + recognition for the BELT greeter.

Key ideas:
  * People are discovered by scanning FACULTY_IMAGES_DIR -- one subfolder per
    person. Adding someone new = adding a folder. No code edits.
  * Each person is reduced to a single L2-normalized *centroid* embedding
    (plus their individual embeddings for a secondary check). Centroids are
    far more stable than nearest-single-image matching.
  * A match must (a) beat an absolute distance threshold and (b) beat the
    runner-up person by a margin. The margin is what stops strangers from
    getting greeted as whoever they happen to resemble most.
  * The encoding cache auto-invalidates when the image folder, the model, or
    the detector changes -- no more stale encodings.joblib.

    python staff_recognition.py --rebuild        # force re-enrollment
    python staff_recognition.py --list           # show enrolled people
    python staff_recognition.py --test img.jpg   # identify faces in an image
    python staff_recognition.py --audit          # leave-one-out self check
"""

import argparse
import hashlib
import os
import pathlib
import sys

import joblib
import numpy as np
from deepface import DeepFace


FACULTY_IMAGES_DIR = "faculty_images2"
ENCODINGS_PATH = "encodings.joblib"

MODEL_NAME = "Facenet"

# Detector used when building the gallery. Slow but accurate is correct here
ENROLL_DETECTOR_BACKEND = "retinaface"
# Detector used on live webcam frames. Set to "yunet" or "opencv" if the video
# feed is too choppy -- see human_det.py's RECOGNITION_INTERVAL first tho.
LIVE_DETECTOR_BACKEND = "retinaface"

MODEL_THRESHOLDS = {
    "Facenet": 0.40,
    "Facenet512": 0.30,
    "ArcFace": 0.68,
    "VGG-Face": 0.40,
    "SFace": 0.59,
    "OpenFace": 0.10,
    "DeepFace": 0.23,
    "GhostFaceNet": 0.65,
}
MATCH_THRESHOLD = MODEL_THRESHOLDS.get(MODEL_NAME, 0.40)

# The best candidate must be at least this much closer than the runner-up
# Raise it if two staff members get confused for each other
MATCH_MARGIN = 0.05

# Minimum detector confidence for a face crop to be considered at all
MIN_FACE_CONFIDENCE = 0.80

# Enrollment images further than this from their own person's centroid are
# treated as mislabeled / bad crops and dropped
OUTLIER_DISTANCE = 0.55

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Cache format version -- bump to force everyone to re-enroll
CACHE_VERSION = 3



def _l2_normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _cosine_distance(a, b):
    return float(1.0 - np.dot(a, b))


def _display_name(folder_name):
    #faculty_images2/Mary_Jane -> 'Mary Jane'
    return folder_name.replace("_", " ").strip()


def _resolve_root(root):
    root = pathlib.Path(root)
    if not root.is_dir():
        return root

    subdirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "__"))]
    has_images = any(
        p.suffix.lower() in VALID_EXTENSIONS for p in root.iterdir() if p.is_file()
    )
    if len(subdirs) == 1 and not has_images:
        inner = subdirs[0]
        inner_subdirs = [
            p for p in inner.iterdir() if p.is_dir() and not p.name.startswith((".", "__"))
        ]
        if inner_subdirs:
            print(f"[INFO] Using nested image root: {inner}")
            return inner
    return root


def discover_people(root=FACULTY_IMAGES_DIR):
  
    # Returns {display_name: [image Path, ...]} for every subfolder that has at
    # least one usable image.
    
    root = _resolve_root(root)
    people = {}

    if not root.is_dir():
        print(f"[ERROR] Image directory not found: {root.resolve()}")
        return people

    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if person_dir.name.startswith((".", "__")):
            continue

        images = sorted(
            p for p in person_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in VALID_EXTENSIONS
            and not p.name.startswith(".")
        )
        if not images:
            print(f"[WARN] No images in {person_dir} -- skipping")
            continue

        people[_display_name(person_dir.name)] = images

    return people


def _fingerprint(people):
    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}|{MODEL_NAME}|{ENROLL_DETECTOR_BACKEND}".encode())
    for name in sorted(people):
        h.update(name.encode())
        for path in people[name]:
            try:
                st = path.stat()
                h.update(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode())
            except OSError:
                h.update(str(path).encode())
    return h.hexdigest()


def _represent(img, detector_backend, enforce_detection):
    return DeepFace.represent(
        img_path=img,
        model_name=MODEL_NAME,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=True,
    )


def _embed_enrollment_image(path):
   
    # Returns a normalized embedding for the largest/most confident face in the
    # image, or None. 
    try:
        faces = _represent(str(path), ENROLL_DETECTOR_BACKEND, enforce_detection=True)
    except Exception as exc:
        print(f"[WARN] No face detected in {path.name}: {exc}")
        return None

    if not faces:
        return None

    # Prefer the biggest face -- enrollment photos occasionally include
    # bystanders in the background
    def area(face):
        fa = face.get("facial_area", {})
        return fa.get("w", 0) * fa.get("h", 0)

    best = max(faces, key=area)
    if len(faces) > 1:
        print(f"[WARN] {len(faces)} faces in {path.name}; using the largest one")

    return _l2_normalize(best["embedding"])


def build_encodings(root=FACULTY_IMAGES_DIR, save_path=ENCODINGS_PATH):
    people = discover_people(root)
    if not people:
        raise RuntimeError(
            f"No enrollable people found under '{root}'. Expected one subfolder per person."
        )

    print(f"[INFO] Enrolling {len(people)} people with {MODEL_NAME} / {ENROLL_DETECTOR_BACKEND}...")

    gallery = {}
    for name, image_paths in people.items():
        embeddings = []
        for path in image_paths:
            emb = _embed_enrollment_image(path)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            print(f"[WARN] {name}: 0 usable images -- NOT enrolled")
            continue

        embeddings = np.vstack(embeddings)
        centroid = _l2_normalize(embeddings.mean(axis=0))

        # Drop outliers 
        if len(embeddings) >= 3:
            dists = np.array([_cosine_distance(centroid, e) for e in embeddings])
            keep = dists <= OUTLIER_DISTANCE
            dropped = int((~keep).sum())
            if dropped and keep.sum() >= 2:
                print(f"[INFO] {name}: dropped {dropped} outlier image(s)")
                embeddings = embeddings[keep]
                centroid = _l2_normalize(embeddings.mean(axis=0))

        spread = float(np.mean([_cosine_distance(centroid, e) for e in embeddings]))

        gallery[name] = {
            "centroid": centroid,
            "embeddings": embeddings,
            "count": int(len(embeddings)),
            "spread": spread,
        }
        flag = "  <-- inconsistent photos" if spread > 0.35 else ""
        print(f"[INFO] {name}: {len(embeddings)} embedding(s), spread={spread:.3f}{flag}")

    if not gallery:
        raise RuntimeError("Enrollment produced no usable embeddings.")

    _warn_about_lookalikes(gallery)

    payload = {
        "version": CACHE_VERSION,
        "model": MODEL_NAME,
        "detector": ENROLL_DETECTOR_BACKEND,
        "fingerprint": _fingerprint(people),
        "gallery": gallery,
    }
    joblib.dump(payload, save_path)
    print(f"[INFO] Saved {len(gallery)} people to {save_path}")
    return payload


def _warn_about_lookalikes(gallery):
    """Flag pairs whose centroids sit closer than the match threshold."""
    names = list(gallery)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = _cosine_distance(gallery[names[i]]["centroid"], gallery[names[j]]["centroid"])
            if d < MATCH_THRESHOLD:
                print(
                    f"[WARN] {names[i]} and {names[j]} are only {d:.3f} apart "
                    f"(threshold {MATCH_THRESHOLD}). Add more varied photos for both."
                )


def load_encodings(root=FACULTY_IMAGES_DIR, path=ENCODINGS_PATH, force_rebuild=False):
    """Load the cache, rebuilding automatically if it is missing or stale."""
    if force_rebuild or not os.path.exists(path):
        return build_encodings(root, path)

    try:
        payload = joblib.load(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path} ({exc}); rebuilding.")
        return build_encodings(root, path)

    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        print("[INFO] Encoding cache is from an older version; rebuilding.")
        return build_encodings(root, path)

    current = _fingerprint(discover_people(root))
    if payload.get("fingerprint") != current:
        print("[INFO] Image folder or config changed; rebuilding encodings.")
        return build_encodings(root, path)

    print(f"[INFO] Loaded {len(payload['gallery'])} enrolled people from {path}")
    return payload


_ENCODINGS = load_encodings(force_rebuild="--rebuild" in sys.argv)
GALLERY = _ENCODINGS["gallery"]
KNOWN_NAMES = sorted(GALLERY)


def _identify(embedding):
    #Returns (name_or_None, best_distance, margin).
   
    scored = []
    for name, entry in GALLERY.items():
        d_centroid = _cosine_distance(embedding, entry["centroid"])
        # Best individual photo, as a safety net for people whose photos vary
        # a lot (glasses on/off, big lighting differences)
        d_best_shot = float(np.min(1.0 - entry["embeddings"] @ embedding))
        scored.append((min(d_centroid, d_best_shot), name))

    scored.sort()
    best_distance, best_name = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else float("inf")
    margin = runner_up - best_distance

    if best_distance <= MATCH_THRESHOLD and margin >= MATCH_MARGIN:
        return best_name, best_distance, margin
    return None, best_distance, margin


def detect_faces(frame, verbose=False):
    
    results = []

    try:
        faces = _represent(frame, LIVE_DETECTOR_BACKEND, enforce_detection=False)
    except Exception as exc:
        if verbose:
            print(f"[WARN] DeepFace.represent failed on frame: {exc}")
        return results

    for face in faces or []:
        confidence = float(face.get("face_confidence", 1.0))
        if confidence < MIN_FACE_CONFIDENCE:
            continue

        area = face.get("facial_area") or {}
        w, h = area.get("w", 0), area.get("h", 0)
        # enforce_detection=False can return the whole frame as a "face"
        if w < 40 or h < 40:
            continue

        embedding = _l2_normalize(face["embedding"])
        name, distance, margin = _identify(embedding)

        if verbose:
            label = name or "UNKNOWN"
            print(f"[DEBUG] {label:<12} distance={distance:.3f} margin={margin:.3f} conf={confidence:.2f}")

        top, left = int(area["y"]), int(area["x"])
        results.append({
            "name": name,
            "distance": distance,
            "margin": margin,
            "box": (top, left + int(w), top + int(h), left),
            "confidence": confidence,
        })

    return results


def getPeople(frame):
    faces = [f for f in detect_faces(frame) if f["name"]]
    return [f["name"] for f in faces], [list(f["box"]) for f in faces]



def _audit():
    # does each enrollment photo match its own person?
    print("\n=== Leave-one-out audit ===")
    total = correct = 0
    for name, entry in GALLERY.items():
        embeddings = entry["embeddings"]
        if len(embeddings) < 2:
            print(f"{name}: only {len(embeddings)} photo(s) -- add more for a reliable match")
            continue

        for idx in range(len(embeddings)):
            probe = embeddings[idx]
            best_name, best_d = None, float("inf")
            for other, other_entry in GALLERY.items():
                pool = np.delete(other_entry["embeddings"], idx, axis=0) if other == name else other_entry["embeddings"]
                if len(pool) == 0:
                    continue
                d = _cosine_distance(_l2_normalize(pool.mean(axis=0)), probe)
                if d < best_d:
                    best_d, best_name = d, other
            total += 1
            if best_name == name:
                correct += 1
            else:
                print(f"  MISS: a photo of {name} looked most like {best_name} ({best_d:.3f})")

    if total:
        print(f"Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
    print(f"Threshold={MATCH_THRESHOLD}  margin={MATCH_MARGIN}")


def main():
    parser = argparse.ArgumentParser(description="Staff face enrollment / recognition")
    parser.add_argument("--rebuild", action="store_true", help="force re-enrollment")
    parser.add_argument("--list", action="store_true", help="list enrolled people")
    parser.add_argument("--test", metavar="IMAGE", help="identify faces in an image")
    parser.add_argument("--audit", action="store_true", help="leave-one-out accuracy check")
    args = parser.parse_args()

    if args.list or args.rebuild or not any([args.test, args.audit]):
        print(f"\nEnrolled ({len(GALLERY)}) using {MODEL_NAME}, threshold {MATCH_THRESHOLD}:")
        for name in KNOWN_NAMES:
            e = GALLERY[name]
            print(f"  {name:<20} {e['count']:>2} photo(s)  spread={e['spread']:.3f}")

    if args.test:
        import cv2
        image = cv2.imread(args.test)
        if image is None:
            print(f"[ERROR] Could not read {args.test}")
            sys.exit(1)
        found = detect_faces(image, verbose=True)
        if not found:
            print("No faces detected.")
        for face in found:
            print(f"  -> {face['name'] or 'UNKNOWN'} (distance {face['distance']:.3f})")

    if args.audit:
        _audit()


if __name__ == "__main__":
    main()
