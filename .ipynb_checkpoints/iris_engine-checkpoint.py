import cv2
import numpy as np
import json
import os
from tensorflow.keras.models import load_model

DB_FILE = "iris_db.json"
MATCH_THRESHOLD = 0.38
UNET_MODEL = load_model("iris_unet.h5", compile=False)

# ---------------- DB ----------------
def db_load():
    if os.path.exists(DB_FILE):
        with open(DB_FILE) as f:
            return json.load(f)
    return {}

def db_save(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

# ---------------- HAMMING ----------------
def hamming(a, b):
    return np.mean(a != b)

def _to_bits(h):
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)).astype(np.float64)

# ---------------- YOUR FEATURE FUNCTION ----------------
_GABOR_KERNELS = [
    cv2.getGaborKernel((11,11),3.0,np.deg2rad(t),8.0,0.5,np.deg2rad(p),cv2.CV_32F)
    for t in [0,45,90,135]
    for p in [0,90]
]

# ---------------- IRIS SEGMENTATION ----------------

EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

    # ---------- DETECT EYE ----------
def segment_iris(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    eyes = EYE_CASCADE.detectMultiScale(
        enhanced,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(eyes) == 0:
        return None

    # use largest eye
    ex, ey, ew, eh = max(
        eyes,
        key=lambda e: e[2] * e[3]
    )

    # crop eye from ORIGINAL COLOR FRAME
    eye_roi = frame[ey:ey+eh, ex:ex+ew]

    if eye_roi.size == 0:
        return None

    # ---------- PREPARE FOR MODEL ----------

    eye_input = cv2.resize(eye_roi, (90, 90))

    eye_input = eye_input.astype(np.float32) / 255.0

    # shape becomes (1,90,90,3)
    x = np.expand_dims(eye_input, axis=0)

    # ---------- PREDICT MASK ----------

    pred = UNET_MODEL.predict(x, verbose=0)[0, :, :, 0]

    mask = (pred > 0.5).astype(np.uint8) * 255

    cv2.imwrite(
        "debug_eye.jpg",
        (eye_input * 255).astype(np.uint8)
    )

    cv2.imwrite(
        "debug_mask.jpg",
        mask
    )
    

    # ---------- FIND IRIS ----------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # largest contour = iris
    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    iris = eye_input[y:y+h, x:x+w]

    if iris.size == 0:
        return None

    # convert to uint8 grayscale for Gabor
    iris = (iris * 255).astype(np.uint8)

    iris = cv2.cvtColor(
        iris,
        cv2.COLOR_BGR2GRAY
    )

    iris = cv2.resize(iris, (64, 64))
    
    cv2.imwrite(
        "debug_iris.jpg",
        iris
    )
    return iris
    
    
    
def extract_code(patch):

    polar = cv2.linearPolar(
        patch.astype(np.float32),
        (32.0, 32.0),
        32.0,
        cv2.WARP_FILL_OUTLIERS
    )

    polar = cv2.resize(polar, (64, 8))

    bits = []

    for k in _GABOR_KERNELS:

        response = cv2.filter2D(
            polar,
            cv2.CV_32F,
            k
        )

        bits.extend(
            (response.flatten() > 0)
            .astype(np.uint8)
            .tolist()
        )

    return np.packbits(bits).tobytes().hex()

    
# ---------------- MATCHING ----------------
def match(query_code):
    db = db_load()
    q = _to_bits(query_code)

    best_name = None
    best_dist = 1.0

    for name, codes in db.items():
        for c in codes:
            d = _to_bits(c)
            n = min(len(q), len(d))
            dist = hamming(q[:n], d[:n])
            if dist < best_dist:
                best_dist = dist
                best_name = name

    if best_name and best_dist <= MATCH_THRESHOLD:
        return {
            "status": "known",
            "name": best_name,
            "confidence": float(1 - best_dist)
        }

    return {
        "status": "unknown",
        "confidence": float(1 - best_dist)
    }

# ---------------- ENROLL ----------------
def enroll(name, code):
    db = db_load()
    db.setdefault(name, [])

    db[name].append(code)
    db_save(db)

    return {"status": "enrolled", "name": name}