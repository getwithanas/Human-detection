"""
config.py — All tuneable parameters in one place.
Loads secrets such as the stream URL from a local .env file when present.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Zone polygon type:  list of (x, y) pixel coordinates
Polygon = List[Tuple[int, int]]


def _load_local_env() -> None:
    """Load simple KEY=VALUE pairs from .env into os.environ once."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ[key] = value


_load_local_env()


@dataclass
class DetectionConfig:
    # ── Stream ─────────────────────────────────────────────
    stream_url: str = os.getenv("STREAM_URL", "")
    # Examples:
    #   "http://192.168.1.100:8080/video"   (IP camera / DroidCam)
    #   "rtsp://admin:pass@192.168.1.50/stream1"
    #   "http://192.168.1.100/mjpeg"

    # ── Model ──────────────────────────────────────────────
    model_path: str = "yolov8n.pt"          # nano=fastest, yolov8s/m/l/x = more accurate
    confidence_threshold: float = 0.50      # 0.0 – 1.0

    # ── Performance ────────────────────────────────────────
    process_every_n_frames: int = 2         # 1=every frame, 2=skip 1, etc.
    display_margin: int = 120               # keep some space around the window
    min_display_width: int = 640
    min_display_height: int = 360

    # ── Alert zones ────────────────────────────────────────
    # Define as many polygons as you need (pixel coords on the stream frame).
    # Leave empty [] to trigger alerts anywhere in the frame.
    alert_zones: List[Polygon] = field(default_factory=lambda: [
        # Example: a rectangular restricted zone (top-left quadrant)
        [(50, 50), (600, 50), (600, 400), (50, 400)],
        # Add more polygons for multiple zones:
        # [(700, 50), (1200, 50), (1200, 400), (700, 400)],
    ])

    zone_labels: List[str] = field(default_factory=lambda: [
        "Zone A — Restricted",
        # "Zone B — Entry",
    ])

    # ── Alert behaviour ────────────────────────────────────
    alert_cooldown_seconds: float = 5.0     # min gap between repeat alerts
    sound_alert: bool = True                # beep on intrusion
    save_snapshot_on_alert: bool = True     # auto-save frame when alert fires
    snapshot_dir: str = "snapshots"

    # ── Overlay colours (BGR) ──────────────────────────────
    color_person_normal: tuple = (0, 220, 0)       # green
    color_person_alert:  tuple = (0, 0, 255)       # red
    color_zone_safe:     tuple = (0, 255, 255)     # yellow
    color_zone_alert:    tuple = (0, 0, 255)       # red
    color_hud:           tuple = (255, 255, 255)   # white
