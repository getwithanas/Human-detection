"""
utils.py — OpenCV drawing helpers and snapshot utility.
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List


# ─────────────────────────────────────────────────────────────
def draw_zones(frame: np.ndarray, zones: list) -> None:
    """Draw semi-transparent filled zone polygons + outlines."""
    overlay = frame.copy()
    for poly in zones:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 80, 255))      # orange-red fill
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

    for poly in zones:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 140, 255), thickness=2)

        # Zone label at centroid
        cx = int(np.mean([p[0] for p in poly]))
        cy = int(np.mean([p[1] for p in poly]))
        _label(frame, "RESTRICTED ZONE", (cx, cy), (0, 140, 255))


# ─────────────────────────────────────────────────────────────
def draw_detections(frame: np.ndarray, detections: list) -> None:
    """Draw bounding boxes, confidence, and status for each detection."""
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        color = (0, 0, 240) if d.in_zone else (30, 220, 30)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents (stylish brackets)
        _draw_corners(frame, x1, y1, x2, y2, color)

        # Label
        label = f"PERSON  {d.confidence:.0%}"
        if d.in_zone:
            label += f"  ⚠ {d.zone_name}"
        _label(frame, label, (x1, y1 - 8), color, bg=True)

        # Centre dot
        cv2.circle(frame, d.center, 4, color, -1)


# ─────────────────────────────────────────────────────────────
def draw_hud(
    frame: np.ndarray,
    fps: float,
    person_count: int,
    alert_active: bool,
) -> None:
    """Draw heads-up display: FPS, count, alert banner."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 44), (15, 15, 15), -1)
    cv2.addWeighted(bar, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    status = f"PERSONS DETECTED: {person_count}"
    cv2.putText(frame, status, (w // 2 - 110, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w - 100, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    # Flashing alert banner at bottom
    if alert_active:
        flash = int(time.time() * 4) % 2 == 0
        if flash:
            cv2.rectangle(frame, (0, h - 48), (w, h), (0, 0, 200), -1)
            cv2.putText(frame, "🚨  INTRUSION ALERT — PERSON IN RESTRICTED ZONE",
                        (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        (255, 255, 255), 2)


# ── Private helpers ───────────────────────────────────────────
def _label(
    frame: np.ndarray,
    text: str,
    pos: tuple,
    color: tuple,
    bg: bool = False,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.55, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    if bg:
        cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + 2), (15, 15, 15), -1)
    cv2.putText(frame, text, (x + 2, y), font, scale, color, thick)


def _draw_corners(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    length: int = 14,
    thickness: int = 3,
) -> None:
    corners = [
        ((x1, y1), (x1 + length, y1), (x1, y1 + length)),
        ((x2, y1), (x2 - length, y1), (x2, y1 + length)),
        ((x1, y2), (x1 + length, y2), (x1, y2 - length)),
        ((x2, y2), (x2 - length, y2), (x2, y2 - length)),
    ]
    for origin, p1, p2 in corners:
        cv2.line(frame, origin, p1, color, thickness)
        cv2.line(frame, origin, p2, color, thickness)


# ─────────────────────────────────────────────────────────────
def save_snapshot(
    frame: np.ndarray,
    directory: Path,
    prefix: str = "snapshot",
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"{prefix}_{ts}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


# time is used inside draw_hud; import here to avoid circular issues
import time
