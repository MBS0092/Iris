"""
Iris Recognition System — PC Version
======================================
Hardware : PC webcam (camera index 0)
Algorithm: Haar eye detection → Hough iris circle → Gabor IrisCode → Hamming match
Database : Local JSON file (iris_db.json)

Controls (OpenCV window must be focused):
    Q  — quit
    R  — cancel current prompt, scan again

After every capture the terminal asks you what to do:
    • If a match is found   → confirm it's correct, or register as someone else
    • If no match is found  → enter a name to register, or skip
"""

import cv2
import numpy as np
import json
import os
import time

def hamming(a, b):
    """Hamming distance between two equal-length bit arrays (0.0 – 1.0)."""
    return np.mean(a != b)

# ──────────────────────────────────────────────────────────
#  SETTINGS
# ──────────────────────────────────────────────────────────
CAMERA_INDEX    = 0        # 0 = default webcam; try 1 or 2 if wrong camera
DB_FILE         = "iris_db.json"
MATCH_THRESHOLD = 0.18     # Hamming distance; lower = stricter
STABLE_SECONDS  = 2.0      # hold still before capture
MIN_IRIS_RADIUS = 18
MAX_IRIS_RADIUS = 80


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

MAX_SAMPLES = 3  # max iris samples stored per person

def db_register(db, name, code_hex):
    db.setdefault(name, [])
    if len(db[name]) >= MAX_SAMPLES:
        print(f"[DB] '{name}' already has {MAX_SAMPLES} samples (max). Skipping save.")
        return False
    db[name].append(code_hex)
    db_save(db)
    count = len(db[name])
    remaining = MAX_SAMPLES - count
    print(f"[DB] Saved iris sample #{count}/{MAX_SAMPLES} for '{name}'.  "
          f"({remaining} slot{'s' if remaining!=1 else ''} remaining)  "
          f"Total identities in DB: {len(db)}")
    return True

def db_list_names(db):
    return sorted(db.keys())


# ──────────────────────────────────────────────────────────
#  IRIS DETECTION
# ──────────────────────────────────────────────────────────
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def detect_iris(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    eyes = EYE_CASCADE.detectMultiScale(
        enhanced,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40)
    )

    if len(eyes) == 0:
        return (None, None, None)

    # ---------------------------------
    # ALWAYS USE SAME EYE SIDE
    # ---------------------------------

    eyes = sorted(
        eyes,
        key=lambda e: e[0]
    )

    # LEFT EYE
    ex, ey, ew, eh = eyes[0]

    # ---------------------------------
    # PADDING
    # ---------------------------------

    pad = 15

    x1_eye = max(ex - pad, 0)
    y1_eye = max(ey - pad, 0)

    x2_eye = min(
        ex + ew + pad,
        frame.shape[1]
    )

    y2_eye = min(
        ey + eh + pad,
        frame.shape[0]
    )

    eye_roi = frame[
        y1_eye:y2_eye,
        x1_eye:x2_eye
    ]

    if eye_roi.size == 0:
        return (None, None, None)

    # ---------------------------------
    # RESIZE FOR PROCESSING
    # ---------------------------------

    eye_input = cv2.resize(
        eye_roi,
        (90, 90)
    )

    gray_eye = cv2.cvtColor(
        eye_input,
        cv2.COLOR_BGR2GRAY
    )

    gray_eye = clahe.apply(gray_eye)

    # ---------------------------------
    # HOUGH IRIS DETECTION
    # ---------------------------------

    circles = cv2.HoughCircles(
        gray_eye,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=18,
        minRadius=15,
        maxRadius=40
    )

    if circles is None:
        return (None, None, None)

    circles = np.round(
        circles[0]
    ).astype(int)

    cx, cy, r = max(
        circles,
        key=lambda c: c[2]
    )

    # ---------------------------------
    # CIRCULAR CROP
    # ---------------------------------

    x1 = max(cx - r, 0)
    y1 = max(cy - r, 0)

    x2 = min(cx + r, gray_eye.shape[1])
    y2 = min(cy + r, gray_eye.shape[0])

    iris = gray_eye[y1:y2, x1:x2]

    if iris.size == 0:
        return (None, None, None)

    # ---------------------------------
    # NORMALIZE SIZE
    # ---------------------------------

    iris = cv2.resize(
        iris,
        (64, 64)
    )

    # ---------------------------------
    # CIRCULAR MASK
    # ---------------------------------

    mask_circle = np.zeros(
        iris.shape[:2],
        dtype=np.uint8
    )

    circle_center = (
        iris.shape[1] // 2,
        iris.shape[0] // 2
    )

    circle_radius = min(
        iris.shape[:2]
    ) // 2

    cv2.circle(
        mask_circle,
        circle_center,
        circle_radius,
        255,
        -1
    )

    iris = cv2.bitwise_and(
        iris,
        iris,
        mask=mask_circle
    )

    # ---------------------------------
    # FINAL CLAHE
    # ---------------------------------

    iris = clahe.apply(iris)


    return iris, (cx, cy), r


# ──────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────
_GABOR_KERNELS = [
    cv2.getGaborKernel(
        (11, 11), sigma=3.0,
        theta=np.deg2rad(t), lambd=8.0,
        gamma=0.5, psi=np.deg2rad(p),
        ktype=cv2.CV_32F
    )
    for t in [0, 45, 90, 135]
    for p in [0, 90]
]

def extract_code(patch):
    polar = cv2.linearPolar(
        patch.astype(np.float32), (32.0, 32.0), 32.0, cv2.WARP_FILL_OUTLIERS
    )
    polar = cv2.resize(polar, (128, 32))
    bits = []
    for k in _GABOR_KERNELS:
        response = cv2.filter2D(
            polar,
            cv2.CV_32F,
            k
        )
        
        bits.extend(
            (
                response.flatten()
                > np.mean(response)
            )
            .astype(np.uint8)
            .tolist()
)
return np.packbits(bits).tobytes().hex()


# ──────────────────────────────────────────────────────────
#  MATCHING
# ──────────────────────────────────────────────────────────
def _to_bits(h):
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)).astype(np.float64)

def match_all(query_code, db):
    """
    Returns:
      candidates — list of (name, dist) within threshold, sorted best-first
      all_scores — dict of {name: best_dist} for everyone in DB
    """
    q = _to_bits(query_code)
    all_scores = {}

    for name, codes in db.items():
        best = 1.0
        for c in codes:
            d = _to_bits(c)
            n = min(len(q), len(d))
            dist = hamming(q[:n], d[:n])
            if dist < best:
                best = dist
        all_scores[name] = best

    candidates = sorted(
        [(n, d) for n, d in all_scores.items() if d <= MATCH_THRESHOLD],
        key=lambda x: x[1]
    )
    return candidates, all_scores


# ──────────────────────────────────────────────────────────
#  TERMINAL PROMPT
# ──────────────────────────────────────────────────────────
# Sentinel returned by prompt_user when user types Q to quit
QUIT_SENTINEL = "__QUIT__"

def _ask(prompt_str):
    """Input wrapper — returns 'Q' on quit, uppercased answer otherwise."""
    raw = input(prompt_str).strip().upper()
    return raw  # caller checks for "Q"

def prompt_user(candidates, all_scores, db, code):
    """
    Interactive terminal prompt.
    Returns (display_text, color), or (QUIT_SENTINEL, None) if user typed Q.
    """
    names = db_list_names(db)
    print("  (type Q at any prompt to quit the program)")

    # ── Case 1: match found ───────────────────────────────
    if candidates:
        best_name, best_dist = candidates[0]
        confidence = int((1 - best_dist / MATCH_THRESHOLD) * 100)

        print(f"\n{'─'*52}")
        print(f"  MATCH:  {best_name}  "
              f"(distance={best_dist:.4f}, confidence≈{confidence}%)")
        if len(candidates) > 1:
            others = ", ".join(f"{n} ({d:.3f})" for n, d in candidates[1:3])
            print(f"  Also close: {others}")
        print(f"{'─'*52}")
        print(f"  [Y]  Yes, this is {best_name}  (adds extra sample → better accuracy)")
        print( "  [N]  No — register as a NEW person")
        if len(names) > 1:
            print("  [A]  Assign to a DIFFERENT existing person")
        print( "  [S]  Skip (do nothing)")
        print( "  [Q]  Quit")
        print()

        while True:
            raw = _ask("  Your choice: ")

            if raw == "Q":
                return QUIT_SENTINEL, None

            elif raw == "Y":
                db_register(db, best_name, code)
                return f"Confirmed: {best_name}", (60, 220, 60)

            elif raw == "N":
                new_name = input("  New person's name (or Q to quit): ").strip()
                if new_name.upper() == "Q":
                    return QUIT_SENTINEL, None
                if new_name:
                    db_register(db, new_name, code)
                    return f"Registered new: {new_name}", (255, 220, 0)
                print("  (no name entered — skipped)")
                return "Skipped.", (140, 140, 140)

            elif raw == "A" and len(names) > 1:
                result = _pick_existing(names, db, code)
                return result

            elif raw == "S":
                return "Skipped.", (140, 140, 140)

            else:
                opts = "Y/N/A/S/Q" if len(names) > 1 else "Y/N/S/Q"
                print(f"  Please enter {opts}.")

    # ── Case 2: no match ──────────────────────────────────
    else:
        closest = min(all_scores.items(), key=lambda x: x[1]) if all_scores else None
        print(f"\n{'─'*52}")
        print( "  NO MATCH FOUND")
        if closest:
            print(f"  Closest in DB: '{closest[0]}'  "
                  f"(distance={closest[1]:.4f} — above threshold {MATCH_THRESHOLD})")
        print(f"{'─'*52}")
        print("  [N]  Register as a NEW person")
        if names:
            print("  [A]  Add this iris sample to an EXISTING person")
        print("  [S]  Skip")
        print("  [Q]  Quit")
        print()

        while True:
            raw = _ask("  Your choice: ")

            if raw == "Q":
                return QUIT_SENTINEL, None

            elif raw == "N":
                new_name = input("  Name (or Q to quit): ").strip()
                if new_name.upper() == "Q":
                    return QUIT_SENTINEL, None
                if new_name:
                    db_register(db, new_name, code)
                    return f"Registered: {new_name}", (255, 220, 0)
                print("  (no name entered — skipped)")
                return "Skipped.", (140, 140, 140)

            elif raw == "A" and names:
                return _pick_existing(names, db, code)

            elif raw == "S":
                return "Skipped.", (140, 140, 140)

            else:
                opts = "N/A/S/Q" if names else "N/S/Q"
                print(f"  Please enter {opts}.")


def _pick_existing(names, db, code):
    """Helper: show numbered list, let user pick a person."""
    print("\n  People in DB:")
    for i, n in enumerate(names, 1):
        samples = len(db.get(n, []))
        print(f"    [{i}] {n}  ({samples} sample{'s' if samples!=1 else ''})")
    raw = input("  Enter number (0 to cancel, Q to quit): ").strip()
    if raw.upper() == "Q":
        return QUIT_SENTINEL, None
    try:
        idx = int(raw) - 1
        if idx < 0:
            return "Cancelled.", (140, 140, 140)
        chosen = names[idx]
        db_register(db, chosen, code)
        return f"Added to: {chosen}", (255, 220, 0)
    except (ValueError, IndexError):
        print("  Invalid — skipped.")
        return "Skipped.", (140, 140, 140)


# ──────────────────────────────────────────────────────────
#  UI HELPERS
# ──────────────────────────────────────────────────────────
YELLOW = (0, 200, 255)
DARK   = (20, 20, 20)

def draw_ring(frame, center, radius, progress, color):
    cv2.ellipse(frame, center, (radius+10, radius+10),
                -90, 0, int(360*progress), color, 3)
    cv2.circle(frame, center, radius, color, 1)

def draw_label(frame, text, center, radius, color):
    x = center[0] - radius
    y = center[1] - radius - 14
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    cv2.rectangle(frame, (x-4, y-h-4), (x+w+4, y+4), DARK, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)

def draw_hud(frame, db, text, color):
    w = frame.shape[1]
    bar = np.full((44, w, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(bar, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA)
    right = f"DB: {len(db)} {'person' if len(db)==1 else 'people'}   |   Q=quit  R=reset"
    cv2.putText(bar, right, (w-370, 30), cv2.FONT_HERSHEY_DUPLEX, 0.52, (150,150,150), 1, cv2.LINE_AA)
    frame[:44] = bar


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────
def main():
    print("\n=== Iris Recognition System ===")
    print(f"Camera : index {CAMERA_INDEX}")
    print(f"DB file: {DB_FILE}")
    print("================================")
    print("Look into the camera. Hold still when an iris is detected.")
    print("After capture, answer the prompts in THIS terminal.\n")

    db  = db_load()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in settings.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    STATE         = "scanning"
    stable_start  = None
    frozen_frame  = None
    frozen_center = None
    frozen_radius = None
    captured_code = None
    result_text   = "Scanning for iris..."
    result_color  = YELLOW

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r') and STATE != "prompting":
            STATE        = "scanning"
            stable_start = None
            result_text  = "Scanning for iris..."
            result_color = YELLOW
            print("[RESET] Scanning again.")

        # ── Grab live frame only when not prompting ───────
        if STATE != "prompting":
            ret, frame = cap.read()
            if not ret:
                continue
            display = frame.copy()
        else:
            display = frozen_frame.copy() if frozen_frame is not None \
                      else np.zeros((480, 640, 3), np.uint8)

        now = time.time()

        # ── SCANNING ──────────────────────────────────────
        if STATE == "scanning":
            patch, center, radius = detect_iris(frame)

            if patch is not None:
                if stable_start is None:
                    stable_start = now
                elapsed  = now - stable_start
                progress = min(elapsed / STABLE_SECONDS, 1.0)

                draw_ring(display, center, radius, progress, YELLOW)
                draw_label(display, "Hold still...", center, radius, YELLOW)
                frozen_center = center
                frozen_radius = radius
                result_text   = "Hold still..."
                result_color  = YELLOW

                if elapsed >= STABLE_SECONDS:
                    captured_code = extract_code(patch)
                    frozen_frame  = display.copy()
                    stable_start  = None
                    STATE         = "prompting"

            else:
                stable_start  = None
                result_text   = "Scanning for iris..."
                result_color  = YELLOW

        # ── PROMPTING (blocking terminal input) ───────────
        elif STATE == "prompting":
            if frozen_center:
                cv2.circle(display, frozen_center, frozen_radius, (200, 200, 0), 2)
            cv2.putText(display,
                        "Check the terminal window to answer →",
                        (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 0), 1, cv2.LINE_AA)
            draw_hud(display, db, "Waiting for your input in terminal...", (200, 200, 0))
            cv2.imshow("Iris Recognition", display)
            cv2.waitKey(1)

            # ── Blocking prompt ───────────────────────────
            candidates, all_scores = match_all(captured_code, db)
            result_text, result_color = prompt_user(candidates, all_scores, db, captured_code)

            if result_text == QUIT_SENTINEL:
                print("\n[QUIT] Exiting...")
                break

            print(f"  → {result_text}  — resuming scan...\n")

            # Go straight back to scanning, clear all frozen state
            STATE         = "scanning"
            stable_start  = None
            frozen_frame  = None
            frozen_center = None
            frozen_radius = None
            captured_code = None
            continue  # skip the imshow at the bottom this iteration

        # ── Draw HUD and show frame ────────────────────────
        draw_hud(display, db, result_text, result_color)
        cv2.imshow("Iris Recognition", display)

    cap.release()
    cv2.destroyAllWindows()
    print("\nExited. Goodbye.")


if __name__ == "__main__":
    main()
