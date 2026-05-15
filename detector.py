"""
╔══════════════════════════════════════════════════════════════╗
║         HUMAN DETECTION & PROXIMITY ALERT SYSTEM            ║
║         Real-time analysis from HTTP Video Stream            ║
╚══════════════════════════════════════════════════════════════╝

Author  : AI Engineer
Version : 2.0.0
Engine  : YOLOv8 + OpenCV
"""

import asyncio
import re
import urllib.request
import cv2
import time
import logging
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

# ─────────────────────────────────────────────
#  Optional: install with  pip install ultralytics
# ─────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import aiohttp
    import av
    from aiortc import RTCPeerConnection, RTCSessionDescription
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False

from config import DetectionConfig
from alert_engine import AlertEngine
from zone_manager import ZoneManager
from utils import draw_detections, draw_zones, draw_hud, save_snapshot

# ── Logger Setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("HumanDetector")


# ─────────────────────────────────────────────────────────────
@dataclass
class Detection:
    """Single person detection result."""
    bbox: tuple          # (x1, y1, x2, y2)
    confidence: float
    center: tuple        # (cx, cy)
    area: int
    in_zone: bool = False
    zone_name: str = ""
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────
class OpenCVStreamWrapper:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()

    def is_opened(self) -> bool:
        return self.cap.isOpened()


class WebRTCStreamReader:
    def __init__(self, stream_url: str, reconnect_delay: float = 3.0):
        self.url = stream_url
        self.whep_url = self._resolve_whep_url(stream_url)
        self.reconnect_delay = reconnect_delay
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pc = None
        self._session_url: Optional[str] = None
        self._offer_data: Optional[dict] = None
        self._candidate_queue: list = []

    def start(self) -> "WebRTCStreamReader":
        self._running = True
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        log.info(f"WebRTC stream reader started → {self.whep_url}")
        return self

    def stop(self) -> None:
        self._running = False
        if self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(self._close_pc_async(), self._loop)
                future.result(timeout=5)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        log.info("WebRTC stream reader stopped.")

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_loop())
        finally:
            self._loop.close()

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._connect_and_receive()
            except Exception as exc:
                log.error(f"WebRTC stream error: {exc}")
            if self._running:
                await asyncio.sleep(self.reconnect_delay)

    async def _connect_and_receive(self) -> None:
        self._session_url = None
        self._offer_data = None
        self._candidate_queue = []

        ice_servers = await self._request_ice_servers()
        self._pc = RTCPeerConnection(configuration={"iceServers": ice_servers})
        self._pc.on("track", self._on_track)
        self._pc.on("icecandidate", self._on_ice_candidate)
        self._pc.on("connectionstatechange", self._on_connection_state_change)
        self._pc.on("iceconnectionstatechange", self._on_connection_state_change)

        self._pc.addTransceiver("video", direction="recvonly")
        self._pc.addTransceiver("audio", direction="recvonly")

        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        self._offer_data = self._parse_offer(offer.sdp)

        answer = await self._send_offer(offer.sdp)
        await self._pc.setRemoteDescription(RTCSessionDescription(answer, "answer"))

        if self._candidate_queue:
            await self._send_local_candidates(self._candidate_queue)
            self._candidate_queue = []

        log.info("✓ WebRTC stream connected.")
        while self._running and self._pc.connectionState not in {"failed", "closed"}:
            await asyncio.sleep(0.2)

        await self._close_pc_async()

    async def _request_ice_servers(self) -> list[dict]:
        headers = self._auth_header()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.options(self.whep_url, headers=headers, timeout=5) as resp:
                    link = resp.headers.get("Link")
                    return self._link_to_ice_servers(link)
        except Exception:
            return []

    async def _send_offer(self, offer_sdp: str) -> str:
        headers = {
            "Content-Type": "application/sdp",
            **self._auth_header(),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.whep_url, data=offer_sdp.encode(), headers=headers) as resp:
                if resp.status != 201:
                    raise RuntimeError(f"WHEP offer failed: {resp.status}")
                location = resp.headers.get("location")
                if not location:
                    raise RuntimeError("WHEP response missing location header")
                self._session_url = urljoin(self.whep_url, location)
                return await resp.text()

    async def _send_local_candidates(self, candidates: list) -> None:
        if self._session_url is None or self._offer_data is None:
            return
        body = self._generate_sdp_fragment(self._offer_data, candidates)
        headers = {
            "Content-Type": "application/trickle-ice-sdpfrag",
            "If-Match": "*",
        }
        async with aiohttp.ClientSession() as session:
            async with session.patch(self._session_url, data=body.encode(), headers=headers) as resp:
                if resp.status != 204:
                    raise RuntimeError(f"WHEP ICE patch failed: {resp.status}")

    async def _close_pc_async(self) -> None:
        if self._pc is not None:
            try:
                await self._pc.close()
            except Exception:
                pass
            self._pc = None

    def _on_track(self, track) -> None:
        if track.kind != "video":
            return

        async def recv_video() -> None:
            while self._running:
                try:
                    frame = await track.recv()
                except Exception:
                    break
                image = frame.to_ndarray(format="bgr24")
                with self._lock:
                    self._frame = image

        asyncio.create_task(recv_video())

    def _on_ice_candidate(self, candidate) -> None:
        if candidate is None:
            return
        if self._session_url is None:
            self._candidate_queue.append(candidate)
        else:
            asyncio.create_task(self._send_local_candidates([candidate]))

    def _on_connection_state_change(self) -> None:
        if self._pc is None:
            return
        if self._pc.connectionState in {"failed", "closed"}:
            log.warning(f"WebRTC connection state changed to {self._pc.connectionState}")

    def _auth_header(self) -> dict:
        return {}

    def _resolve_whep_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.path.endswith("/whep"):
            return url

        path = parsed.path
        if not path.endswith("/"):
            path += "/"
        base = urljoin(parsed._replace(path=path, query="", fragment="").geturl(), "")
        whep = urljoin(base, "whep")
        if parsed.query:
            whep += "?" + parsed.query
        return whep

    def _link_to_ice_servers(self, link_header: Optional[str]) -> list[dict]:
        if not link_header:
            return []

        servers = []
        for part in link_header.split(","):
            match = re.match(
                r"^\s*<(.+?)>; rel=\"ice-server\"(?:; username=\"(.*?)\"; credential=\"(.*?)\"; credential-type=\"password\")?",
                part.strip(),
                re.I,
            )
            if not match:
                continue
            server = {"urls": [match.group(1)]}
            if match.group(2) is not None:
                server["username"] = match.group(2)
                server["credential"] = match.group(3)
            servers.append(server)
        return servers

    @staticmethod
    def _parse_offer(sdp: str) -> dict:
        ret = {"iceUfrag": "", "icePwd": "", "medias": []}
        for line in sdp.split("\r\n"):
            if line.startswith("m="):
                ret["medias"].append(line[2:])
            elif ret["iceUfrag"] == "" and line.startswith("a=ice-ufrag:"):
                ret["iceUfrag"] = line[len("a=ice-ufrag:"):]
            elif ret["icePwd"] == "" and line.startswith("a=ice-pwd:"):
                ret["icePwd"] = line[len("a=ice-pwd:"):]
        return ret

    @staticmethod
    def _generate_sdp_fragment(od: dict, candidates: list) -> str:
        candidates_by_media = {}
        for candidate in candidates:
            mid = candidate.sdpMLineIndex
            candidates_by_media.setdefault(mid, []).append(candidate)

        frag = f"a=ice-ufrag:{od['iceUfrag']}\r\n"
        frag += f"a=ice-pwd:{od['icePwd']}\r\n"

        mid = 0
        for media in od["medias"]:
            if mid in candidates_by_media:
                frag += f"m={media}\r\n"
                frag += f"a=mid:{mid}\r\n"
                for candidate in candidates_by_media[mid]:
                    frag += f"a={candidate.candidate}\r\n"
            mid += 1

        return frag


class StreamReader:
    """
    Thread-safe stream reader with OpenCV capture and WHEP fallback.
    """

    def __init__(self, stream_url: str, reconnect_delay: float = 3.0):
        self.url = stream_url
        self.reconnect_delay = reconnect_delay
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream = None

    # ── Public API ────────────────────────────────────────────
    def start(self) -> "StreamReader":
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info(f"Stream reader started → {self.url}")
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._stream is not None:
            if isinstance(self._stream, OpenCVStreamWrapper):
                self._stream.release()
            else:
                self._stream.stop()
        log.info("Stream reader stopped.")

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    # ── Internal loop ─────────────────────────────────────────
    def _capture_loop(self) -> None:
        self._stream = None
        while self._running:
            if self._stream is None:
                self._stream = self._connect()
                if self._stream is None:
                    time.sleep(self.reconnect_delay)
                    continue

            if isinstance(self._stream, OpenCVStreamWrapper):
                ret, frame = self._stream.read()
            else:
                ret, frame = self._stream.read()

            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                continue

            log.warning("Lost stream — reconnecting…")
            if isinstance(self._stream, OpenCVStreamWrapper):
                self._stream.release()
            else:
                self._stream.stop()
            self._stream = None
            time.sleep(self.reconnect_delay)

    def _connect(self):
        log.info(f"Connecting to stream: {self.url}")
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            log.info("✓ Stream connected via OpenCV.")
            return OpenCVStreamWrapper(cap)

        log.warning("OpenCV failed to open stream.")
        if AIORTC_AVAILABLE and self._is_webrtc_viewer(self.url):
            log.info("Attempting WebRTC/WHEP fallback.")
            reader = WebRTCStreamReader(self.url, self.reconnect_delay)
            reader.start()
            return reader

        log.error("✗ Failed to open stream.")
        return None

    def _is_webrtc_viewer(self, url: str) -> bool:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                content_type = resp.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type:
                    return False
                html = resp.read(4096).decode(errors="ignore").lower()
                return "mediamtxwebrtcreader" in html or "new mediamtxwebrtcreader" in html or "whep" in html
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────
class HumanDetector:
    """
    Core detection engine.
    Loads YOLOv8 (or HOG fallback), runs inference,
    filters by confidence, and returns Detection objects.
    """

    def __init__(self, config: DetectionConfig):
        self.cfg = config
        self.model = self._load_model()

    def _load_model(self):
        if YOLO_AVAILABLE:
            log.info(f"Loading YOLO model: {self.cfg.model_path}")
            model = YOLO(self.cfg.model_path)
            model.fuse()          # speed optimisation
            log.info("✓ YOLO model loaded.")
            return ("yolo", model)

        log.warning("ultralytics not found — using HOG fallback detector.")
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return ("hog", hog)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        kind, model = self.model
        if kind == "yolo":
            return self._yolo_detect(frame, model)
        return self._hog_detect(frame, model)

    # ── YOLO inference ────────────────────────────────────────
    def _yolo_detect(self, frame: np.ndarray, model) -> list[Detection]:
        results = model(
            frame,
            conf=self.cfg.confidence_threshold,
            classes=[0],          # class 0 = person in COCO
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    center=(cx, cy),
                    area=area,
                ))
        return detections

    # ── HOG fallback ─────────────────────────────────────────
    def _hog_detect(self, frame: np.ndarray, hog) -> list[Detection]:
        small = cv2.resize(frame, (640, 480))
        sx = frame.shape[1] / 640
        sy = frame.shape[0] / 480

        rects, weights = hog.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
        )
        detections = []
        for (x, y, w, h), conf in zip(rects, weights):
            x1, y1 = int(x * sx), int(y * sy)
            x2, y2 = int((x + w) * sx), int((y + h) * sy)
            if conf < self.cfg.confidence_threshold:
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                center=(cx, cy),
                area=area,
            ))
        return detections


# ─────────────────────────────────────────────────────────────
class DetectionSystem:
    """
    Orchestrates the full pipeline:
      StreamReader → HumanDetector → ZoneManager → AlertEngine → Display
    """

    def __init__(self, config: DetectionConfig):
        self.cfg = config
        self.stream = StreamReader(config.stream_url)
        self.detector = HumanDetector(config)
        self.zones = ZoneManager(config.alert_zones)
        self.alert_engine = AlertEngine(config)

        self._fps_buf: deque = deque(maxlen=30)
        self._running = False
        self._snapshot_dir = Path(config.snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._window_name = "Human Detection System"
        self._zone_edit_mode = False
        self._draft_zone: list[tuple[int, int]] = []
        self._mouse_callback_set = False
        self._window_ready = False
        self._screen_size: Optional[tuple[int, int]] = self._detect_screen_size()

    def _detect_screen_size(self) -> Optional[tuple[int, int]]:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            if width > 0 and height > 0:
                return width, height
        except Exception:
            return None
        return None

    def _fit_window_to_frame(self, frame: np.ndarray) -> None:
        if self._window_ready:
            return

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self._window_name,
            cv2.WND_PROP_ASPECT_RATIO,
            cv2.WINDOW_KEEPRATIO,
        )

        frame_h, frame_w = frame.shape[:2]
        target_w, target_h = frame_w, frame_h

        if self._screen_size is not None:
            screen_w, screen_h = self._screen_size
            max_w = max(self.cfg.min_display_width, screen_w - self.cfg.display_margin)
            max_h = max(self.cfg.min_display_height, screen_h - self.cfg.display_margin)

            scale = min(max_w / frame_w, max_h / frame_h, 1.0)
            target_w = max(self.cfg.min_display_width, int(frame_w * scale))
            target_h = max(self.cfg.min_display_height, int(frame_h * scale))

            if target_w > max_w:
                target_w = max_w
            if target_h > max_h:
                target_h = max_h

        cv2.resizeWindow(self._window_name, target_w, target_h)
        self._window_ready = True

    def _ensure_window_callback(self) -> None:
        if self._mouse_callback_set:
            return
        cv2.setMouseCallback(self._window_name, self._on_mouse)
        self._mouse_callback_set = True

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if not self._zone_edit_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._draft_zone.append((x, y))

    def _render_zone_editor(self, frame: np.ndarray) -> None:
        hud_lines = [
            "Z: toggle zone edit",
            "ENTER: save zone",
            "C: clear draft",
            "X: clear all zones",
            "Q: quit",
        ]

        panel_h = 24 + 22 * len(hud_lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 54), (260, 54 + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        title = "ZONE EDIT MODE" if self._zone_edit_mode else "ZONE CONTROLS"
        cv2.putText(frame, title, (22, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        for idx, line in enumerate(hud_lines, start=1):
            cv2.putText(
                frame,
                line,
                (22, 78 + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (235, 235, 235),
                1,
            )

        if self._draft_zone:
            pts = np.array(self._draft_zone, dtype=np.int32)
            for pt in self._draft_zone:
                cv2.circle(frame, pt, 5, (0, 220, 255), -1)
            if len(self._draft_zone) > 1:
                cv2.polylines(frame, [pts], False, (0, 220, 255), 2)

    def _handle_key(self, key: int, vis: np.ndarray) -> bool:
        if key == ord("q"):
            return False
        if key == ord("s"):
            path = save_snapshot(vis, self._snapshot_dir)
            log.info(f"Snapshot saved → {path}")
        elif key == ord("z"):
            self._zone_edit_mode = not self._zone_edit_mode
            self._draft_zone.clear()
            state = "enabled" if self._zone_edit_mode else "disabled"
            log.info(f"Zone edit mode {state}.")
        elif key == ord("c") and self._zone_edit_mode:
            self._draft_zone.clear()
            log.info("Cleared draft zone.")
        elif key == ord("x"):
            self.zones.clear()
            self._draft_zone.clear()
            log.info("Cleared all restricted zones.")
        elif key in (10, 13) and self._zone_edit_mode:
            if len(self._draft_zone) >= 3:
                self.zones.add_zone(list(self._draft_zone))
                log.info(f"Added restricted zone with {len(self._draft_zone)} points.")
                self._draft_zone.clear()
                self._zone_edit_mode = False
            else:
                log.warning("Need at least 3 points to create a zone.")
        return True

    # ── Entry point ───────────────────────────────────────────
    def run(self) -> None:
        self.stream.start()
        self._running = True
        log.info("Detection system running. Press Z to edit zones, S to save snapshot, Q to quit.")

        skip = 0   # frame-skip counter for performance
        while self._running:
            ok, frame = self.stream.read()
            if not ok:
                time.sleep(0.05)
                continue

            skip += 1
            if skip % self.cfg.process_every_n_frames != 0:
                continue

            t0 = time.perf_counter()

            # ① Detect
            detections = self.detector.detect(frame)

            # ② Zone check
            for d in detections:
                d.in_zone, d.zone_name = self.zones.check(d.center)

            # ③ Alert
            self.alert_engine.evaluate(detections, frame)

            # ④ FPS
            elapsed = time.perf_counter() - t0
            self._fps_buf.append(1.0 / elapsed if elapsed > 0 else 0)
            fps = sum(self._fps_buf) / len(self._fps_buf)

            # ⑤ Render
            vis = frame.copy()
            draw_zones(vis, self.zones.zone_polygons)
            draw_detections(vis, detections)
            draw_hud(vis, fps, len(detections), self.alert_engine.active_alert)
            self._render_zone_editor(vis)

            self._fit_window_to_frame(vis)
            cv2.imshow(self._window_name, vis)
            self._ensure_window_callback()
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key, vis):
                break

        self._shutdown()

    def _shutdown(self) -> None:
        log.info("Shutting down…")
        self.stream.stop()
        cv2.destroyAllWindows()
        log.info("Goodbye.")
