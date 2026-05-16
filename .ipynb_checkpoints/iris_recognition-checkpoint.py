"""
Iris Recognition System — PC Version
======================================
Hardware : PC webcam (camera index 0)
Algorithm: Haar eye detection → Hough iris circle → Gabor IrisCode → Hamming match
Database : Local JSON file (iris_db.json)

Usage:
    python iris_recognition.py

Controls:
    Q  — quit
    R  — reset/clear current result and scan again
"""

import cv2
import numpy as np
import json
import os
import time
from scipy.spatial.distance import hamming

# ──────────────────────────────────────────────────────────
#  SETTINGS  (tweak these if detection is poor)
# ──────────────────────────────────────────────────────────
CAMERA_INDEX     = 0        # 0 = default webcam
DB_FILE          = "iris_db.json"
MATCH_THRESHOLD  = 0.38     # Hamming distance; lower = stricter match
STABLE_SECONDS   = 2.0      # hold iris still for this long before capture
MIN_IRIS_RADIUS  = 18       # pixels — ignore circles smaller than this
MAX_IRIS_RADIUS  = 80       # pixels — ignore circles larger than this


# ──────────────────────────────────────────────────────────
#  DATABASE
# ──────────────────────────────────────────────────────────
def db_load():
    if os.path.exists(DB_FILE):
        with open(DB_FILE) as f:
            return json.load(f)
    return {}

def db_save(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

def db_register(db, name, code_hex):
    db.setdefault(name, []).append(code_hex)
    db_save(db)
    print(f"[DB] Saved iris for '{name}'.  Total identities: {len(db)}")


# ──────────────────────────────────────────────────────────
#  IRIS DETECTION
# ──────────────────────────────────────────────────────────
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def detect_iris(frame):
    """
    1. Grayscale + CLAHE (contrast boost)
    2. Haar cascade → eye bounding boxes
    3. Hough circles inside each eye box → pick the iris circle
    Returns: (64×64 gray patch, (cx,cy) in frame, radius) or (None,None,None)
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)

    eyes  = EYE_CASCADE.detectMultiScale(
        eq, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    best = None  # (patch, center, radius)

    for (ex, ey, ew, eh) in eyes:
        roi = eq[ey:ey+eh, ex:ex+ew]

        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1, minDist=30,
            param1=50, param2=22,
            minRadius=MIN_IRIS_RADIUS,
            maxRadius=MAX_IRIS_RADIUS,
        )

        if circles is None:
            continue

        circles = np.round(circles[0]).astype(int)

        for (cx, cy, r) in circles:
            abs_cx = ex + cx
            abs_cy = ey + cy

            x1 = max(abs_cx - r, 0);  y1 = max(abs_cy - r, 0)
            x2 = min(abs_cx + r, frame.shape[1])
            y2 = min(abs_cy + r, frame.shape[0])

            patch = gray[y1:y2, x1:x2]
            if patch.size == 0 or patch.shape[0] < 10 or patch.shape[1] < 10:
                continue

            patch = cv2.resize(patch, (64, 64))

            # Prefer the largest radius (iris over pupil)
            if best is None or r > best[2]:
                best = (patch, (abs_cx, abs_cy), r)

    if best:
        return best
    return None, None, None


# ──────────────────────────────────────────────────────────
#  FEATURE EXTRACTION  — simplified IrisCode
# ──────────────────────────────────────────────────────────
# Gabor filter bank: 4 orientations × 2 phases = 8 filters
_GABOR_KERNELS = []
for _theta in [0, 45, 90, 135]:
    for _psi in [0, 90]:
        k = cv2.getGaborKernel(
            (11, 11), sigma=3.0,
            theta=np.deg2rad(_theta),
            lambd=8.0, gamma=0.5,
            psi=np.deg2rad(_psi),
            ktype=cv2.CV_32F
        )
        _GABOR_KERNELS.append(k)

def extract_code(patch):
    """
    Rubber-sheet polar transform → Gabor filter bank → binarize → hex string.
    Returns a hex string representing the IrisCode bits.
    """
    # Polar (rubber-sheet) unwrap: maps circular iris to a rectangle
    polar = cv2.linearPolar(
        patch.astype(np.float32),
        (32.0, 32.0), 32.0,
        cv2.WARP_FILL_OUTLIERS
    )
    polar = cv2.resize(polar, (64, 8))  # 64 angular bins × 8 radial strips

    bits = []
    for kernel in _GABOR_KERNELS:
        response = cv2.filter2D(polar, cv2.CV_32F, kernel)
        bits.extend((response.flatten() > 0).astype(np.uint8).tolist())

    packed = np.packbits(bits)
    return packed.tobytes().hex()


# ──────────────────────────────────────────────────────────
#  MATCHING
# ──────────────────────────────────────────────────────────
def _to_bits(hex_code):
    return np.unpackbits(np.frombuffer(bytes.fromhex(hex_code), dtype=np.uint8)).astype(np.float64)

def match(query_code, db):
    """
    Compare query IrisCode against every stored code using Hamming distance.
    Returns (matched_name, distance) or (None, best_distance).
    """
    q = _to_bits(query_code)
    best_name, best_dist = None, 1.0

    for name, codes in db.items():
        for c in codes:
            d_bits = _to_bits(c)
            n = min(len(q), len(d_bits))
            dist = hamming(q[:n], d_bits[:n])
            if dist < best_dist:
                best_dist, best_name = dist, name

    if best_dist <= MATCH_THRESHOLD:
        return best_name, best_dist
    return None, best_dist


# ──────────────────────────────────────────────────────────
#  UI HELPERS
# ──────────────────────────────────────────────────────────
GREEN  = (60, 220, 60)
YELLOW = (0, 200, 255)
RED    = (60, 60, 220)
CYAN   = (255, 220, 0)
WHITE  = (255, 255, 255)
DARK   = (20, 20, 20)

def draw_iris_ring(frame, center, radius, progress, color):
    """Animated sweep ring showing capture countdown."""
    sweep = int(360 * progress)
    cv2.ellipse(frame, center, (radius+10, radius+10), -90, 0, sweep, color, 3)
    cv2.circle(frame, center, radius, color, 1)

def draw_label(frame, text, center, radius, color):
    x = center[0] - radius
    y = center[1] - radius - 14
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    cv2.rectangle(frame, (x-4, y-h-4), (x+w+4, y+4), DARK, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)

def draw_hud(frame, db, state_text, color):
    h, w = frame.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    cv2.putText(bar, state_text, (10, 27), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA)
    ids = f"Identities in DB: {len(db)}   |   Q=quit  R=reset"
    cv2.putText(bar, ids, (w-380, 27), cv2.FONT_HERSHEY_DUPLEX, 0.55, (160,160,160), 1, cv2.LINE_AA)
    frame[:40] = bar


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────
def main():
    print("\n=== Iris Recognition System ===")
    print(f"Camera: index {CAMERA_INDEX}")
    print(f"DB    : {DB_FILE}")
    print("================================\n")

    db  = db_load()
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Could not open camera. Check CAMERA_INDEX setting.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    stable_start = None   # time when iris became stable
    result_until = 0      # show result label until this time
    result_text  = ""
    result_color = WHITE
    frozen_center = None
    frozen_radius = None

    STATE = "scanning"    # scanning | stable | result

    print("Look into the camera. Hold still when an iris is found.")
    print("A name prompt will appear in the terminal for new irises.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now   = time.time()
        patch, center, radius = detect_iris(frame)

        # ── State machine ──────────────────────────────
        if STATE == "scanning":
            if patch is not None:
                if stable_start is None:
                    stable_start = now
                elapsed  = now - stable_start
                progress = min(elapsed / STABLE_SECONDS, 1.0)
                draw_iris_ring(frame, center, radius, progress, YELLOW)
                draw_label(frame, "Hold still...", center, radius, YELLOW)
                frozen_center = center
                frozen_radius = radius

                if elapsed >= STABLE_SECONDS:
                    # ── Capture & match ────────────────
                    code       = extract_code(patch)
                    name, dist = match(code, db)

                    if name:
                        result_text  = f"MATCH: {name}  (d={dist:.3f})"
                        result_color = GREEN
                        print(f"[MATCH]  {name}   distance={dist:.4f}")
                        STATE        = "result"
                        result_until = now + 4.0
                    else:
                        # Unknown — freeze display, ask in terminal
                        print(f"\n[UNKNOWN] No match found (closest distance={dist:.4f})")
                        print("  Enter a name to register this iris,")
                        new_name = input("  or press Enter to skip: ").strip()

                        if new_name:
                            db_register(db, new_name, code)
                            result_text  = f"Registered: {new_name}"
                            result_color = CYAN
                        else:
                            result_text  = "Skipped."
                            result_color = (140, 140, 140)

                        STATE        = "result"
                        result_until = now + 3.0

                    stable_start = None

            else:
                stable_start = None  # iris lost, reset timer

        elif STATE == "result":
            # Show result overlay for a few seconds, then go back to scanning
            if frozen_center and frozen_radius:
                cv2.circle(frame, frozen_center, frozen_radius, result_color, 2)
                draw_label(frame, result_text, frozen_center, frozen_radius, result_color)

            if now > result_until:
                STATE         = "scanning"
                frozen_center = None
                frozen_radius = None
                result_text   = ""

        # ── HUD bar ───────────────────────────────────
        state_labels = {
            "scanning": "Scanning for iris...",
            "stable"  : "Hold still...",
            "result"  : result_text,
        }
        draw_hud(frame, db, state_labels.get(STATE, ""), result_color if STATE=="result" else YELLOW)

        cv2.imshow("Iris Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            STATE        = "scanning"
            stable_start = None
            frozen_center = None
            frozen_radius = None
            result_text  = ""
            print("[RESET] Scanning again.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nExited. Goodbye.")


if __name__ == "__main__":
    main()
