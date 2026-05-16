from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
import time
import json
import threading
import serial
import serial.tools.list_ports

from iris_engine import match, enroll, extract_code, segment_iris

app = Flask(__name__)

# ── CHANGE THIS to your Arduino COM port ──
SERIAL_PORT = "COM4"      # Windows: COM3, COM4 ...
SERIAL_BAUD = 115200        # Linux: /dev/ttyACM0 or /dev/ttyUSB0

ser = None

LATEST_RESULT = {
    "status": "waiting",
    "name": None,
    "confidence": 0.0,
    "message": "Waiting for scan..."
}

SYSTEM_MODE       = "recognize"
LOG_FILE          = "match_log.json"
PENDING_IRIS_CODE = None          # iris code waiting for TFT name confirmation
_lock             = threading.Lock()   # protect PENDING_IRIS_CODE + LATEST_RESULT


# ════════════════════════════════════════
# SERIAL HELPERS
# ════════════════════════════════════════

def serial_send(msg: str):
    global ser
    if ser and ser.is_open:
        try:
            ser.write((msg + "\n").encode())
            print(f"[TFT <-] {msg}")
        except Exception as e:
            print(f"[Serial TX error] {e}")


def push_result_to_tft(result: dict):
    """Send all four UI fields to TFT so the home screen refreshes."""
    serial_send(f"UI|STATUS|{result.get('status', 'waiting')}")
    serial_send(f"UI|NAME|{result.get('name') or ''}")
    serial_send(f"UI|CONF|{float(result.get('confidence', 0)):.3f}")
    serial_send(f"UI|MESSAGE|{result.get('message', '')}")


def push_enroll_prompt():
    """Tell TFT: unknown iris detected, show confirm screen."""
    serial_send("UI|ENROLL_PROMPT|")


def push_enroll_ok(name: str):
    serial_send(f"UI|ENROLL_OK|{name}")


def push_enroll_fail(reason: str):
    serial_send(f"UI|ENROLL_FAIL|{reason}")


# ════════════════════════════════════════
# SERIAL READER (background thread)
# ════════════════════════════════════════

def serial_reader():
    global ser
    buf = ""
    while True:
        try:
            if ser and ser.is_open and ser.in_waiting:
                c = ser.read().decode(errors="ignore")
                if c == "\n":
                    line = buf.strip()
                    buf = ""
                    if line:
                        handle_arduino_command(line)
                else:
                    buf += c
        except Exception as e:
            print(f"[Serial RX error] {e}")
        time.sleep(0.01)


def handle_arduino_command(line: str):
    global PENDING_IRIS_CODE, LATEST_RESULT
    print(f"[TFT ->] {line}")

    if line.startswith("ENROLL:"):
        name = line[7:].strip()
        if not name:
            push_enroll_fail("Empty name")
            return

        with _lock:
            code = PENDING_IRIS_CODE
            PENDING_IRIS_CODE = None

        if code is not None:
            result = enroll(name, code)
        else:
            # Fallback: re-process last saved frame
            result = enroll_from_last_frame(name)

        if result.get("status") in ("enrolled", "ok", "success"):
            enrolled_name = result.get("name", name)
            new_result = {
                "status":     "enrolled",
                "name":       enrolled_name,
                "confidence": 1.0,
                "message":    f"Enrolled: {enrolled_name}"
            }
            with _lock:
                LATEST_RESULT.update(new_result)

            # Push fresh home data THEN the OK banner (banner draws over header)
            push_result_to_tft(new_result)
            push_enroll_ok(enrolled_name)
            save_log(new_result)
        else:
            msg = result.get("message", "Failed")
            push_enroll_fail(msg)

    elif line == "IGNORE":
        with _lock:
            PENDING_IRIS_CODE = None
        print("[TFT] User ignored enroll prompt.")

    elif line == "GETUI":
        with _lock:
            r = dict(LATEST_RESULT)
        push_result_to_tft(r)


def enroll_from_last_frame(name: str) -> dict:
    if not os.path.exists("debug_last_frame.jpg"):
        return {"status": "error", "message": "No frame captured yet"}

    img = cv2.imread("debug_last_frame.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"status": "error", "message": "Cannot read frame"}

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    patch = segment_iris(img_color)
    if patch is None:
        return {"status": "error", "message": "Iris not detected in frame"}

    cv2.imwrite("debug_iris.jpg", patch)
    code = extract_code(patch)
    return enroll(name, code)


# ════════════════════════════════════════
# START SERIAL
# ════════════════════════════════════════

def start_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)
        print(f"[Serial] Connected: {SERIAL_PORT}")
        threading.Thread(target=serial_reader, daemon=True).start()
    except Exception as e:
        print(f"[Serial] Cannot open {SERIAL_PORT}: {e}")
        err = str(e).lower()
        if "access is denied" in err or "permissionerror" in err or "permission denied" in err:
            print(
                "[Serial] Port is probably in use: close Arduino Serial Monitor / "
                "second Serial Plotter, PuTTY, or another python server.py on this COM."
            )
        ports = serial.tools.list_ports.comports()
        if ports:
            print("[Serial] Available ports:")
            for p in ports:
                print(f"  {p.device} — {p.description}")
        else:
            print("[Serial] No serial ports found.")
        print("[Serial] Continuing without TFT.")


# ════════════════════════════════════════
# LOG
# ════════════════════════════════════════

def save_log(result):
    log = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                log = json.load(f)
            except Exception:
                pass
    log.append({
        "time":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "status":     result.get("status"),
        "name":       result.get("name"),
        "confidence": float(result.get("confidence", 0))
    })
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


# ════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════

@app.route("/mode", methods=["GET", "POST"])
def mode():
    global SYSTEM_MODE
    if request.method == "POST":
        SYSTEM_MODE = request.get_json().get("mode", "recognize")
    return jsonify({"mode": SYSTEM_MODE})


@app.route("/upload", methods=["POST"])
def upload():
    global LATEST_RESULT

    if SYSTEM_MODE == "enroll":
        return jsonify({"status": "paused", "message": "Enroll mode active"})

    img_data = request.data
    if not img_data:
        return jsonify({"status": "error", "message": "no image"}), 400

    nparr = np.frombuffer(img_data, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({"status": "error", "message": "decode failed"}), 400

    cv2.imwrite("debug_last_frame.jpg", img)

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    patch     = segment_iris(img_color)

    if patch is None:
        r = {"status": "error", "name": None,
             "confidence": 0.0, "message": "Iris not detected"}
        with _lock:
            LATEST_RESULT = r
        push_result_to_tft(r)
        return jsonify(r), 400

    cv2.imwrite("debug_iris.jpg", patch)
    code   = extract_code(patch)
    result = match(code)

    if result["status"] == "known":
        r = {
            "status":     "known",
            "name":       result["name"],
            "confidence": float(result["confidence"]),
            "message":    "Access Granted"
        }
        with _lock:
            LATEST_RESULT = r
        push_result_to_tft(r)

    else:
        r = {
            "status":     "unknown",
            "name":       None,
            "confidence": float(result["confidence"]),
            "message":    "Not Enrolled"
        }
        with _lock:
            LATEST_RESULT = r
        # Update home screen first, then pop up the confirm screen
        push_result_to_tft(r)
        push_enroll_prompt()

    save_log(LATEST_RESULT)
    return jsonify(LATEST_RESULT)


@app.route("/enroll_frame", methods=["POST"])
def enroll_frame_route():
    """Web UI enroll — re-uses last saved frame, asks TFT for name."""
    global PENDING_IRIS_CODE

    if not os.path.exists("debug_last_frame.jpg"):
        return jsonify({"status": "error", "message": "No frame captured yet"}), 400

    img = cv2.imread("debug_last_frame.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({"status": "error", "message": "Cannot read frame"}), 400

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    patch = segment_iris(img_color)
    if patch is None:
        return jsonify({"status": "error", "message": "Iris not detected in frame"}), 400

    cv2.imwrite("debug_iris.jpg", patch)
    with _lock:
        PENDING_IRIS_CODE = extract_code(patch)

    push_enroll_prompt()
    return jsonify({"status": "pending", "message": "Waiting for name on TFT screen"})


@app.route("/enroll", methods=["POST"])
def api_enroll():
    """
    Web UI enroll — uploaded image.
    Segments iris, stores code, then asks TFT for name confirmation.
    Returns immediately with status='pending'.
    The TFT keyboard result comes back via serial → handle_arduino_command.
    """
    global PENDING_IRIS_CODE

    if "image" not in request.files:
        return jsonify({"status": "error", "message": "missing image"}), 400

    img = cv2.imdecode(
        np.frombuffer(request.files["image"].read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )
    if img is None:
        return jsonify({"status": "error", "message": "invalid image"}), 400

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    patch     = segment_iris(img_color)
    if patch is None:
        return jsonify({"status": "error", "message": "iris not detected"}), 400

    cv2.imwrite("debug_last_frame.jpg", img)
    cv2.imwrite("debug_iris.jpg", patch)

    with _lock:
        PENDING_IRIS_CODE = extract_code(patch)

    push_enroll_prompt()

    return jsonify({
        "status":  "pending",
        "message": "Waiting for name confirmation on TFT screen"
    })


@app.route("/debug/frame")
def debug_frame():
    if not os.path.exists("debug_last_frame.jpg"):
        return "No frame", 404
    return send_file("debug_last_frame.jpg", mimetype="image/jpeg")


@app.route("/debug/iris")
def debug_iris():
    if not os.path.exists("debug_iris.jpg"):
        return "No iris", 404
    return send_file("debug_iris.jpg", mimetype="image/jpeg")


@app.route("/latest")
def latest():
    with _lock:
        return jsonify(LATEST_RESULT)


@app.route("/")
def home():
    path = "web/index.html"
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    return "Web UI not found"


if __name__ == "__main__":
    start_serial()
    print("Server running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)