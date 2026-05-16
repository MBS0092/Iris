# Iris Recognition System — PC Version

Simple iris recognition that runs entirely on your PC webcam.
No cloud, no server — just Python + OpenCV.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
python iris_recognition.py
```

---

## How It Works

```
Webcam frame
    │
    ▼
Grayscale + CLAHE (contrast enhance)
    │
    ▼
Haar Cascade → finds eye region
    │
    ▼
Hough Circle Transform → finds iris circle inside eye
    │
    ▼
Rubber-sheet polar transform (normalizes iris size/rotation)
    │
    ▼
Gabor filter bank (8 filters, 4 orientations × 2 phases)
    │
    ▼
Binarize responses → IrisCode (hex string)
    │
    ▼
Hamming distance vs every code in iris_db.json
    │
    ├─ distance ≤ 0.38 → MATCH found → show name
    └─ distance  > 0.38 → UNKNOWN → ask name → register
```

---

## Usage

| Action | What happens |
|--------|-------------|
| Look at camera | System scans for your iris |
| Hold still for 2 seconds | Countdown ring fills up, then captures |
| First time = unknown | Terminal prompts for your name |
| Next time = known | Shows your name instantly |
| Press `R` | Reset, scan again |
| Press `Q` | Quit |

---

## Settings (top of iris_recognition.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_INDEX` | `0` | Webcam index. Try `1` or `2` if wrong camera opens |
| `MATCH_THRESHOLD` | `0.38` | Lower = stricter. Raise if you get too many unknowns |
| `STABLE_SECONDS` | `2.0` | How long to hold still before capture |
| `MIN_IRIS_RADIUS` | `18` | Tune if small circles are being detected as iris |
| `MAX_IRIS_RADIUS` | `80` | Tune if iris isn't being detected |

---

## Database

Iris codes are stored in `iris_db.json` next to the script.
Each person can have multiple registered samples — the system picks the best match across all of them.

To reset the database, just delete `iris_db.json`.

---

## Future: Porting to ESP32

The algorithm is intentionally lightweight for this reason:
- No ML models, no large libraries
- Fixed-size feature vectors (Gabor IrisCodes)
- Simple Hamming distance comparison

When porting to ESP32:
- Replace OpenCV with ESP-IDF camera + a lightweight circle detector
- Implement Gabor filters as fixed integer kernels
- Store DB in flash (SPIFFS/LittleFS)
- Replace the terminal name prompt with a small OLED/TFT keyboard UI
