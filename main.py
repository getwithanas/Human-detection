"""
main.py — Entry point for the Human Detection & Alert System.

Usage:
    python main.py                          # use .env / config defaults
    python main.py --url http://IP/video    # override stream URL
    python main.py --paint-zones            # interactive zone drawing tool
    python main.py --no-display             # headless / server mode
"""

import argparse
import logging
import sys
import cv2
import numpy as np
from pathlib import Path

from config import DetectionConfig
from detector import DetectionSystem

log = logging.getLogger("Main")


# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Human Detection & Proximity Alert System")
    p.add_argument("--url",         type=str,  help="Override stream URL from config")
    p.add_argument("--confidence",  type=float,help="Detection confidence threshold (0–1)")
    p.add_argument("--model",       type=str,  help="Path/name of YOLO model")
    p.add_argument("--paint-zones", action="store_true",
                   help="Launch interactive zone drawing tool first")
    p.add_argument("--no-display",  action="store_true",
                   help="Run without GUI (log + snapshot only)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
def interactive_zone_painter(stream_url: str) -> list:
    """
    Click to define polygon vertices on the first frame of the stream.
    Press ENTER to close polygon, C to clear, Q to finish.
    Returns list of polygons (each a list of (x,y) tuples).
    """
    cap = cv2.VideoCapture(stream_url)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        log.error("Could not read frame for zone painting.")
        return []

    clone = frame.copy()
    zones, current = [], []

    COLORS = [(0, 220, 255), (0, 255, 128), (255, 128, 0), (200, 0, 255)]

    def mouse_cb(event, x, y, flags, param):
        nonlocal frame
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
            frame = clone.copy()
            _render()

    def _render():
        for zi, zone in enumerate(zones):
            pts = np.array(zone, np.int32)
            c = COLORS[zi % len(COLORS)]
            cv2.polylines(frame, [pts.reshape(-1,1,2)], True, c, 2)
            cv2.fillPoly(frame.copy(), [pts.reshape(-1,1,2)], c)
        if len(current) > 1:
            pts = np.array(current, np.int32)
            cv2.polylines(frame, [pts.reshape(-1,1,2)], False, (255,255,255), 2)
        for pt in current:
            cv2.circle(frame, pt, 5, (0,200,255), -1)
        cv2.putText(frame,
            "Click = add point | ENTER = close zone | C = clear | Q = done",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow("Zone Painter", frame)

    cv2.namedWindow("Zone Painter")
    cv2.setMouseCallback("Zone Painter", mouse_cb)
    _render()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13 and len(current) >= 3:   # ENTER
            zones.append(list(current))
            current.clear()
            clone2 = frame.copy()
            frame = clone2
            _render()
        elif key == ord("c"):
            current.clear()
            frame = clone.copy()
            _render()
        elif key == ord("q"):
            break

    cv2.destroyWindow("Zone Painter")
    log.info(f"Defined {len(zones)} zone(s).")
    return zones


# ─────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    cfg = DetectionConfig()

    # Apply CLI overrides
    if args.url:
        cfg.stream_url = args.url
    if args.confidence:
        cfg.confidence_threshold = args.confidence
    if args.model:
        cfg.model_path = args.model

    print(
        "\n"
        "╔══════════════════════════════════════════════════╗\n"
        "║   Human Detection & Proximity Alert System       ║\n"
        "║   Stream : {:<38}║\n"
        "║   Model  : {:<38}║\n"
        "║   Conf   : {:<38}║\n"
        "╚══════════════════════════════════════════════════╝\n"
        .format(cfg.stream_url[:38], cfg.model_path[:38],
                str(cfg.confidence_threshold)[:38])
    )

    # Optional interactive zone painting
    if args.paint_zones:
        painted = interactive_zone_painter(cfg.stream_url)
        if painted:
            cfg.alert_zones = painted

    # Validate stream URL
    if "YOUR_CAMERA_IP" in cfg.stream_url:
        log.error(
            "Stream URL not configured!\n"
            "Set STREAM_URL in .env  OR  pass --url http://your-camera/video"
        )
        sys.exit(1)

    if not cfg.stream_url.strip():
        log.error(
            "Stream URL not configured!\n"
            "Set STREAM_URL in .env  OR  pass --url http://your-camera/video"
        )
        sys.exit(1)

    system = DetectionSystem(cfg)
    system.run()


if __name__ == "__main__":
    main()
