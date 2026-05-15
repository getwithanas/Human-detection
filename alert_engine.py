"""
alert_engine.py — Evaluates detections and fires alerts.
Supports: console log, system beep, snapshot save, webhook (optional).
"""

import time
import logging
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("AlertEngine")


class AlertEngine:
    """
    Fires alerts when a person enters a restricted zone.

    Alert types (all configurable via DetectionConfig):
      • Console log  — always on
      • Sound beep   — cross-platform (winsound / beep / print fallback)
      • Snapshot     — saves annotated frame to disk
      • Webhook      — optional POST to an external URL
    """

    def __init__(self, config):
        self.cfg = config
        self._last_alert_time: float = 0.0
        self.active_alert: bool = False
        self._alert_clear_at: float = 0.0

    # ── Main evaluation ───────────────────────────────────────
    def evaluate(self, detections: list, frame: np.ndarray) -> None:
        intruders = [d for d in detections if d.in_zone]

        now = time.time()
        # Clear active alert flag if cooldown passed with no intruders
        if self.active_alert and now > self._alert_clear_at:
            self.active_alert = False

        if not intruders:
            return

        self.active_alert = True
        self._alert_clear_at = now + self.cfg.alert_cooldown_seconds

        # Respect cooldown between actual alert actions
        if now - self._last_alert_time < self.cfg.alert_cooldown_seconds:
            return

        self._last_alert_time = now
        self._fire(intruders, frame)

    # ── Alert dispatch ────────────────────────────────────────
    def _fire(self, intruders: list, frame: np.ndarray) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        zones = {d.zone_name for d in intruders}
        count = len(intruders)

        msg = (
            f"🚨 ALERT [{ts}]  {count} person(s) detected in "
            f"zone(s): {', '.join(zones)}"
        )
        log.warning(msg)
        print(f"\n{'═'*60}\n{msg}\n{'═'*60}\n")

        if self.cfg.sound_alert:
            threading.Thread(target=self._beep, daemon=True).start()

        if self.cfg.save_snapshot_on_alert:
            from utils import save_snapshot
            path = save_snapshot(frame, Path(self.cfg.snapshot_dir), prefix="ALERT")
            log.info(f"Alert snapshot → {path}")

    # ── Cross-platform beep ───────────────────────────────────
    @staticmethod
    def _beep() -> None:
        try:
            import winsound                         # Windows
            winsound.Beep(1000, 500)
            return
        except ImportError:
            pass
        try:
            import subprocess                       # Linux / macOS
            subprocess.run(["beep"], capture_output=True)
            return
        except Exception:
            pass
        try:
            import subprocess
            subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                           capture_output=True)
        except Exception:
            print("\a", end="", flush=True)         # terminal bell fallback
