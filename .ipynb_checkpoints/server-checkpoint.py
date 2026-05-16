from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import time
import json

from iris_engine import (
    match,
    enroll,
    extract_code,
    segment_iris
)

app = Flask(__name__)

# ---------------- GLOBAL STATE ----------------
LATEST_RESULT = {
    "status": "waiting",
    "name": None,
    "confidence": 0.0,
    "message": "No data yet"
}

LOG_FILE = "match_log.json"


# ---------------- SAVE LOG ----------------
def save_log(result):
    log = []

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                log = json.load(f)
            except:
                log = []

    log.append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": result.get("status"),
        "name": result.get("name"),
        "confidence": float(result.get("confidence", 0))
    })

    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


# ---------------- ESP32 UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload():
    global LATEST_RESULT

    img_data = request.data

    if not img_data:
        return jsonify({"status": "error", "message": "no image"}), 400

    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return jsonify({"status": "error", "message": "decode failed"}), 400

    # convert grayscale upload to BGR
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    patch = segment_iris(img_color)
    
    if patch is None:
        return jsonify({
            "status": "error",
            "message": "iris not detected"
        }), 400
    
    code = extract_code(patch)

    result = match(code)

    # ---------------- FORMAT RESULT ----------------
    if result["status"] == "known":
        LATEST_RESULT = {
            "status": "known",
            "name": result["name"],
            "confidence": float(result["confidence"]),
            "message": "Access Granted"
        }
    else:
        LATEST_RESULT = {
            "status": "unknown",
            "name": None,
            "confidence": float(result["confidence"]),
            "message": "Not Enrolled"
        }

    # ---------------- SAVE LOG ----------------
    save_log(LATEST_RESULT)

    return jsonify(LATEST_RESULT)


# ---------------- LIVE RESULT ----------------
@app.route("/latest")
def latest():
    return jsonify(LATEST_RESULT)


# ---------------- ENROLL USER ----------------
@app.route("/enroll", methods=["POST"])
def api_enroll():
    if "name" not in request.form or "image" not in request.files:
        return jsonify({"status": "error", "message": "missing data"}), 400

    name = request.form["name"]
    file = request.files["image"]

    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return jsonify({"status": "error", "message": "invalid image"}), 400
    
    # convert grayscale upload to BGR
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    patch = segment_iris(img_color)
    
    if patch is None:
        return jsonify({
            "status": "error",
            "message": "iris not detected"
        }), 400
    
    code = extract_code(patch)

    return jsonify(enroll(name, code))


# ---------------- WEB PAGE ----------------
@app.route("/")
def home():
    path = "web/index.html"
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    return "Web UI not found"


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    print("Server running on http://0.0.0.0:5000")
    app.run(
    host="0.0.0.0",
    port=5000,
    debug=False
)



