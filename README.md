# Human Detection & Restricted Zone Alert

Real-time person detection from a network video stream with polygon-based restricted zones, on-screen overlays, alert beeps, and automatic alert snapshots.

## Current Setup

- Entry point: `main.py`
- Detection engine: YOLOv8 via `ultralytics`
- Default model: `yolov8n.pt`
- Stream URL source: `.env` via `STREAM_URL`
- Restricted zones: configured in `config.py`
- Alert snapshots: saved to `snapshots/`

The app currently opens a live OpenCV display window and lets you edit zones while the stream is running.

## Project Files

```text
human detection/
├── main.py
├── detector.py
├── config.py
├── alert_engine.py
├── zone_manager.py
├── utils.py
├── requirements.txt
├── yolov8n.pt
└── snapshots/
```

## Requirements

- Python 3.10+
- A reachable camera/video stream
- Desktop environment access for the OpenCV display window

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` currently includes:

- `opencv-python`
- `numpy`
- `ultralytics`
- `aiortc`
- `av`
- `aiohttp`

## Configuration

Edit [`config.py`](/Downloads/human%20detection/config.py) to match your camera and alert preferences.

Create a local `.env` file for the camera stream:

```env
STREAM_URL="rtsp://username:password@camera-ip:554/stream1"
```

Important settings:

```python
stream_url = os.getenv("STREAM_URL", "")
model_path = "yolov8n.pt"
confidence_threshold = 0.50
process_every_n_frames = 2
alert_cooldown_seconds = 5.0
sound_alert = True
save_snapshot_on_alert = True
snapshot_dir = "snapshots"
```

Restricted zones are defined as polygons:

```python
alert_zones = [
    [(50, 50), (600, 50), (600, 400), (50, 400)],
]
```

If `alert_zones` is empty, alerts trigger anywhere in the frame.

## Running

Start with the config defaults:

```bash
python main.py
```

The app reads `STREAM_URL` from `.env` automatically.

Override the stream URL:

```bash
python main.py --url http://192.168.1.100:8080/video
```

Override confidence:

```bash
python main.py --confidence 0.6
```

Override the YOLO model:

```bash
python main.py --model yolov8s.pt
```

Open the zone painter before detection starts:

```bash
python main.py --paint-zones
```

## Supported Stream Behavior

The code currently tries the stream in this order:

1. OpenCV `VideoCapture`
2. WebRTC/WHEP fallback when the URL looks like a MediaMTX-style WebRTC viewer page and the async WebRTC dependencies are installed

Typical stream examples:

- `http://192.168.1.100:8080/video`
- `rtsp://admin:password@192.168.1.50/stream1`
- `http://camera-host:8889/stream1/`

## Runtime Controls

While the live window is open:

- `Q`: quit
- `S`: save a manual snapshot
- `Z`: toggle zone edit mode
- `Enter`: save the current draft zone
- `C`: clear the draft zone
- `X`: clear all configured zones
- Left mouse click: add points to the draft zone when zone edit mode is enabled

## Alerts

When a detected person enters a configured restricted zone, the system:

- marks the detection in red
- shows an intrusion banner on screen
- logs the alert in the console
- plays a beep when enabled
- saves a snapshot when enabled

Alert snapshots are written to `snapshots/` using names like:

```text
ALERT_YYYYMMDD_HHMMSS_microseconds.jpg
```

## Notes

- Keep real credentials in `.env`. The repo includes `.env.example` as the template and `.gitignore` excludes `.env`.
- `--no-display` is defined in `main.py`, but the current runtime still opens the OpenCV display window and does not run fully headless yet.
- `--paint-zones` reads the first frame with OpenCV. If your stream only works through the WebRTC fallback path, the painter may not load that first frame successfully.
- Zone labels in `config.py` are not currently used by the renderer; zones are displayed generically as `Zone A`, `Zone B`, and so on.
