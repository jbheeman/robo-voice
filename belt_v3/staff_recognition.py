"""
staff_recognition.py
--------------------
Face enrollment + recognition for the BELT greeter

Backend: InsightFace installs on any Python 3.9-3.13, runs fast on CPU

Setup:
    pip install insightface onnxruntime opencv-python joblib numpy
    # if insightface fails to build:  pip install cython numpy  first


Key ideas:
  * People are discovered by scanning FACULTY_IMAGES_DIR -- one subfolder per
    person. Adding someone new = adding a folder. No code edits
  * Each person is reduced to a single L2-normalized *centroid* embedding
    (plus their individual embeddings for a secondary check). Centroids are
    far more stable than nearest-single-image matching
  * A match must (a) beat an absolute distance threshold and (b) beat the
    runner-up person by a margin. The margin is what stops strangers from
    getting greeted as whoever they happen to resemble most
  * The encoding cache auto-invalidates when the image folder or the model
    config changes -- no more stale encodings.joblib

CLI:
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

import cv2
import joblib
import numpy as np
from insightface.app import FaceAnalysis


FACULTY_IMAGES_DIR = "faculty_images2"
ENCODINGS_PATH = "encodings.joblib"


MODEL_PACK = "buffalo_l"


PROVIDERS = ["CPUExecutionProvider"]

# Detector input size. Bigger = detects smaller/further faces, slower
DET_SIZE = (640, 640)
DET_THRESHOLD = 0.5

MATCH_THRESHOLD = 0.60

# The best candidate must be at least this much closer than the runner-up
# Raise it if two staff members get confused for each other
MATCH_MARGIN = 0.05

# Minimum detector confidence for a face to be considered at all
MIN_FACE_CONFIDENCE = 0.60

# Faces smaller than this (pixels, on the long side) are too low-res to
# identify reliably -- treated as "present but unknown" rather than matched
MIN_FACE_SIZE = 50

# Enrollment images further than this from their own person's centroid are
# treated as mislabeled / bad crops and dropped
OUTLIER_DISTANCE = 0.85

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

CACHE_VERSION = 4



_APP = None


def get_app():
    global _APP
    if _APP is None:
        print(f"[INFO] Loading InsightFace '{MODEL_PACK}' (first run downloads ~275MB)...")
        app = FaceAnalysis(
            name=MODEL_PACK,
            providers=PROVIDERS,
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=DET_SIZE, det_thresh=DET_THRESHOLD)
        _APP = app
        print("[INFO] Model ready.")
    return _APP



def _l2_normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _cosine_distance(a, b):
    """Cosine distance for vectors that are ALREADY L2-normalized."""
    return float(1.0 - np.dot(a, b))


def _display_name(folder_name):
    """faculty_images2/Mary_Jane -> 'Mary Jane'."""
    return folder_name.replace("_", " ").strip()


def _imread(path):
    """cv2.imread that survives non-ASCII paths."""
    image = cv2.imread(str(path))
    if image is not None:
        return image
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


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
    # least one usable image
    
    root = _resolve_root(root)
    people = {}

    if not root.is_dir():
        print(f"[ERROR] Image directory not found: {pathlib.Path(root).resolve()}")
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
    h.update(f"v{CACHE_VERSION}|{MODEL_PACK}|{DET_SIZE}|{DET_THRESHOLD}".encode())
    for name in sorted(people):
        h.update(name.encode())
        for path in people[name]:
            try:
                st = path.stat()
                h.update(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode())
            except OSError:
                h.update(str(path).encode())
    return h.hexdigest()


def _face_area(face):
    x1, y1, x2, y2 = face.bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _embed_enrollment_image(path):
    
    image = _imread(path)
    if image is None:
        print(f"[WARN] Could not read {path.name}")
        return None

    try:
        faces = get_app().get(image)
    except Exception as exc:
        print(f"[WARN] Detection failed on {path.name}: {exc}")
        return None

    faces = [f for f in faces if f.det_score >= MIN_FACE_CONFIDENCE]
    if not faces:
        print(f"[WARN] No face detected in {path.name}")
        return None

    # Prefer the biggest face
    best = max(faces, key=_face_area)
    if len(faces) > 1:
        print(f"[WARN] {len(faces)} faces in {path.name}; using the largest one")

    return _l2_normalize(best.normed_embedding)


def build_encodings(root=FACULTY_IMAGES_DIR, save_path=ENCODINGS_PATH):
    people = discover_people(root)
    if not people:
        raise RuntimeError(
            f"No enrollable people found under '{root}'. Expected one subfolder per person."
        )

    print(f"[INFO] Enrolling {len(people)} people with {MODEL_PACK}...")

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

        # Intra-person spread: high means inconsistent photos.
        spread = float(np.mean([_cosine_distance(centroid, e) for e in embeddings]))

        gallery[name] = {
            "centroid": centroid,
            "embeddings": embeddings,
            "count": int(len(embeddings)),
            "spread": spread,
        }
        flag = "  <-- inconsistent photos" if spread > 0.50 else ""
        print(f"[INFO] {name}: {len(embeddings)} embedding(s), spread={spread:.3f}{flag}")

    if not gallery:
        raise RuntimeError("Enrollment produced no usable embeddings.")

    _warn_about_lookalikes(gallery)

    payload = {
        "version": CACHE_VERSION,
        "model": MODEL_PACK,
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

    if payload.get("fingerprint") != _fingerprint(discover_people(root)):
        print("[INFO] Image folder or config changed; rebuilding encodings.")
        return build_encodings(root, path)

    print(f"[INFO] Loaded {len(payload['gallery'])} enrolled people from {path}")
    return payload



_ENCODINGS = load_encodings(force_rebuild="--rebuild" in sys.argv)
GALLERY = _ENCODINGS["gallery"]
KNOWN_NAMES = sorted(GALLERY)

_CENTROIDS = np.vstack([GALLERY[n]["centroid"] for n in KNOWN_NAMES]) if KNOWN_NAMES else None


def _identify(embedding):
   
    if _CENTROIDS is None:
        return None, 1.0, 0.0

    centroid_d = 1.0 - _CENTROIDS @ embedding
    best_shot_d = np.array([
        float(np.min(1.0 - GALLERY[n]["embeddings"] @ embedding)) for n in KNOWN_NAMES
    ])
    distances = np.minimum(centroid_d, best_shot_d)

    order = np.argsort(distances)
    best_distance = float(distances[order[0]])
    best_name = KNOWN_NAMES[order[0]]
    runner_up = float(distances[order[1]]) if len(order) > 1 else float("inf")
    margin = runner_up - best_distance

    if best_distance <= MATCH_THRESHOLD and margin >= MATCH_MARGIN:
        return best_name, best_distance, margin
    return None, best_distance, margin


def detect_faces(frame, verbose=False):
    
    results = []

    try:
        faces = get_app().get(frame)
    except Exception as exc:
        if verbose:
            print(f"[WARN] Face detection failed on frame: {exc}")
        return results

    height, width = frame.shape[:2]

    for face in faces or []:
        confidence = float(face.det_score)
        if confidence < MIN_FACE_CONFIDENCE:
            continue

        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        left, top = max(0, x1), max(0, y1)
        right, bottom = min(width, x2), min(height, y2)
        if (right - left) < MIN_FACE_SIZE or (bottom - top) < MIN_FACE_SIZE:
            # Too far away to identify -- report as an unknown face so the
            # greeter still says "Welcome" instead of guessing a name
            results.append({
                "name": None, "distance": 1.0, "margin": 0.0,
                "box": (top, right, bottom, left), "confidence": confidence,
            })
            continue

        embedding = _l2_normalize(face.normed_embedding)
        name, distance, margin = _identify(embedding)

        if verbose:
            print(f"[DEBUG] {name or 'UNKNOWN':<12} distance={distance:.3f} "
                  f"margin={margin:.3f} conf={confidence:.2f}")

        results.append({
            "name": name,
            "distance": distance,
            "margin": margin,
            "box": (top, right, bottom, left),
            "confidence": confidence,
        })

    return results


def getPeople(frame):

    faces = [f for f in detect_faces(frame) if f["name"]]
    return [f["name"] for f in faces], [list(f["box"]) for f in faces]



def _audit():

    print("\n=== Leave-one-out audit ===")
    total = correct = confident = 0
    same_person, diff_person = [], []

    for name, entry in GALLERY.items():
        embeddings = entry["embeddings"]
        if len(embeddings) < 2:
            print(f"  {name}: only {len(embeddings)} photo(s) -- add more, this cannot be validated")
            continue

        for idx in range(len(embeddings)):
            probe = embeddings[idx]
            scored = []
            for other, other_entry in GALLERY.items():
                pool = (np.delete(other_entry["embeddings"], idx, axis=0)
                        if other == name else other_entry["embeddings"])
                if len(pool) == 0:
                    continue
                d = _cosine_distance(_l2_normalize(pool.mean(axis=0)), probe)
                scored.append((d, other))
                (same_person if other == name else diff_person).append(d)

            scored.sort()
            total += 1
            best_d, best_name = scored[0]
            margin = scored[1][0] - best_d if len(scored) > 1 else float("inf")

            if best_name == name:
                correct += 1
                if best_d <= MATCH_THRESHOLD and margin >= MATCH_MARGIN:
                    confident += 1
                else:
                    print(f"  WEAK: a photo of {name} matched correctly but "
                          f"below threshold (d={best_d:.3f}, margin={margin:.3f})")
            else:
                print(f"  MISS: a photo of {name} looked most like {best_name} (d={best_d:.3f})")

    if not total:
        print("Nothing to audit.")
        return

    print(f"\nCorrect person is nearest: {correct}/{total} ({100 * correct / total:.1f}%)")
    print(f"...and would actually be greeted: {confident}/{total} ({100 * confident / total:.1f}%)")

    if same_person and diff_person:
        s, d = np.array(same_person), np.array(diff_person)
        print(f"\nSame-person distances : mean {s.mean():.3f}  max {s.max():.3f}")
        print(f"Different-person dists: mean {d.mean():.3f}  min {d.min():.3f}")
        print(f"Current threshold {MATCH_THRESHOLD}, margin {MATCH_MARGIN}")
        gap = d.min() - s.max()
        if gap > 0:
            print(f"Clean separation (gap {gap:.3f}). "
                  f"A threshold anywhere in {s.max():.2f}-{d.min():.2f} works.")
        else:
            print("Same-person and different-person distances OVERLAP. "
                  "Lower MATCH_THRESHOLD to stay safe, and add more varied photos.")


def main():
    parser = argparse.ArgumentParser(description="Staff face enrollment / recognition")
    parser.add_argument("--rebuild", action="store_true", help="force re-enrollment")
    parser.add_argument("--list", action="store_true", help="list enrolled people")
    parser.add_argument("--test", metavar="IMAGE", help="identify faces in an image")
    parser.add_argument("--audit", action="store_true", help="leave-one-out accuracy check")
    args = parser.parse_args()

    
    if args.list or args.rebuild or not any([args.test, args.audit]):
        print(f"\nEnrolled ({len(GALLERY)}) using {MODEL_PACK}, threshold {MATCH_THRESHOLD}:")
        for name in KNOWN_NAMES:
            e = GALLERY[name]
            print(f"  {name:<20} {e['count']:>2} photo(s)  spread={e['spread']:.3f}")

    if args.test:
        image = _imread(args.test)
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
