"""
╔══════════════════════════════════════════════════════════════════╗
║     CCTV FACE CROP + GOD-LEVEL UPSCALER                        ║
║     YOLOv8 Person Detection → Face Crop → AI Upscale           ║
╚══════════════════════════════════════════════════════════════════╝

PIPELINE:
  1. YOLOv8  → detect all persons in the frame
  2. Smart crop → extract face/head region from each person box
  3. Real-ESRGAN → upscale face to crystal-clear 4x
  4. Enhancement → sharpen, denoise, contrast boost
  5. Save each face as a separate output image

USAGE (CLI):
  python cctv_face_upscale.py --input cctv.png
  python cctv_face_upscale.py --input cctv.png --scale 4 --show
  python cctv_face_upscale.py --input folder/ --scale 4

USAGE (Python API):
  from cctv_face_upscale import CCTVFaceUpscaler
  upscaler = CCTVFaceUpscaler(scale=4)
  results = upscaler.process("cctv.png")
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
from typing import List, Tuple, Optional

# ── Device detection ──────────────────────────────────────────────────────────
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else \
             "mps"  if torch.backends.mps.is_available() else "cpu"
    print(f"[⚡] Device: {DEVICE.upper()}")
except ImportError:
    DEVICE = "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class CCTVFaceUpscaler:
    """
    Detects every person in a CCTV frame,
    crops their face/head region, and upscales to crystal clarity.
    """

    def __init__(
        self,
        scale: int = 4,
        face_padding: float = 0.35,   # extra % padding around face crop
        conf_threshold: float = 0.35, # YOLO confidence threshold
        min_person_height: int = 60,  # ignore tiny detections (px)
        verbose: bool = True,
    ):
        self.scale            = scale
        self.face_padding     = face_padding
        self.conf_threshold   = conf_threshold
        self.min_person_height = min_person_height
        self.verbose          = verbose
        self._yolo            = None
        self._esrgan          = None

        self._load_yolo()
        self._load_esrgan()

    # ── MODEL LOADING ─────────────────────────────────────────────────────────

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            self._log("[🧠] Loading YOLOv8 person detector...")
            self._yolo = YOLO("yolov8n.pt")   # auto-downloaded on first run
            self._log("[✅] YOLOv8 ready")
        except Exception as e:
            raise RuntimeError(f"YOLOv8 load failed: {e}")

    def _load_esrgan(self):
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            self._log("[🧠] Loading Real-ESRGAN upscaler...")
            model_path = self._download_model(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.1.0/RealESRGAN_x4plus.pth",
                "RealESRGAN_x4plus.pth"
            )
            arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)
            self._esrgan = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=arch,
                tile=256,
                tile_pad=10,
                pre_pad=0,
                half=(DEVICE == "cuda"),
                device=DEVICE,
            )
            self._log("[✅] Real-ESRGAN ready")
        except Exception as e:
            self._log(f"[⚠] Real-ESRGAN not available ({e}) — using Lanczos fallback")
            self._esrgan = None

    @staticmethod
    def _download_model(url: str, filename: str) -> str:
        import urllib.request
        cache_dir = Path.home() / ".cache" / "godupscaler"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / filename
        if not dest.exists():
            print(f"[📥] Downloading {filename}...")
            urllib.request.urlretrieve(url, dest)
            print(f"[✅] Saved → {dest}")
        return str(dest)

    # ═══════════════════════════════════════════════════════════════════════════
    #  MAIN PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════

    def process(
        self,
        source: str,
        output_dir: Optional[str] = None,
        show: bool = False,
    ) -> List[Image.Image]:
        """
        Full pipeline: detect → crop face → upscale → enhance → save.

        Returns list of upscaled face PIL Images (one per person found).
        """
        # ── Load image ────────────────────────────────────────────────────────
        img_path = Path(source)
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {source}")

        h, w = frame.shape[:2]
        self._log(f"\n[📸] Image: {img_path.name}  ({w}×{h}px)")

        # ── Output dir ────────────────────────────────────────────────────────
        if output_dir is None:
            output_dir = str(img_path.parent / (img_path.stem + "_faces"))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # ── Step 1: Detect persons ────────────────────────────────────────────
        boxes = self._detect_persons(frame)
        if not boxes:
            self._log("[⚠] No persons detected in this frame.")
            return []

        self._log(f"[👤] Detected {len(boxes)} person(s)")

        # ── Step 2: For each person → crop face → upscale ────────────────────
        results = []
        for i, box in enumerate(boxes):
            self._log(f"\n[🔍] Person {i+1}/{len(boxes)}")

            # Crop the face/head region
            face_crop = self._crop_face_region(frame, box)
            if face_crop is None:
                self._log("   [skip] Face region too small")
                continue

            fh, fw = face_crop.shape[:2]
            self._log(f"   Face crop: {fw}×{fh}px → upscaling ×{self.scale}...")

            # Upscale + enhance
            upscaled = self._upscale_and_enhance(face_crop)
            results.append(upscaled)

            # Save
            out_file = Path(output_dir) / f"{img_path.stem}_person{i+1}_face_{self.scale}x.png"
            upscaled.save(str(out_file), quality=99)
            self._log(f"   [💾] Saved → {out_file}")

            if show:
                upscaled.show(title=f"Person {i+1} — Crystal Clear")

        # ── Step 3: Save annotated overview ──────────────────────────────────
        annotated = self._draw_annotations(frame.copy(), boxes)
        overview_path = Path(output_dir) / f"{img_path.stem}_overview.jpg"
        cv2.imwrite(str(overview_path), annotated)
        self._log(f"\n[🗺] Overview saved → {overview_path}")
        self._log(f"[🏁] Done! {len(results)} face(s) upscaled → {output_dir}")

        return results

    # ── PERSON DETECTION ──────────────────────────────────────────────────────

    def _detect_persons(self, frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
        """Run YOLOv8 and return bounding boxes for class=person only."""
        results = self._yolo(
            frame,
            conf=self.conf_threshold,
            classes=[0],   # class 0 = person
            verbose=False,
        )
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                person_h = y2 - y1
                if person_h >= self.min_person_height:
                    boxes.append((x1, y1, x2, y2))
        return boxes

    # ── SMART FACE CROP ───────────────────────────────────────────────────────

    def _crop_face_region(
        self,
        frame: np.ndarray,
        box: Tuple[int,int,int,int],
        face_fraction: float = 0.32,  # top ~32% of person box = head/face
    ) -> Optional[np.ndarray]:
        """
        From a full-body person bounding box, extract the head/face region.
        Takes the top `face_fraction` of the box height, with padding.
        """
        x1, y1, x2, y2 = box
        person_h = y2 - y1
        person_w = x2 - x1

        # Head is roughly top 30% of the person box
        face_h = int(person_h * face_fraction)
        face_cx = (x1 + x2) // 2

        # Square crop centered on head
        half = max(face_h, int(person_w * 0.5)) // 2
        pad  = int(half * self.face_padding)

        # Face region with padding
        fx1 = max(0, face_cx - half - pad)
        fy1 = max(0, y1 - pad)
        fx2 = min(frame.shape[1], face_cx + half + pad)
        fy2 = min(frame.shape[0], y1 + face_h + pad * 2)

        crop = frame[fy1:fy2, fx1:fx2]

        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        return crop

    # ── UPSCALE + ENHANCE ────────────────────────────────────────────────────

    def _upscale_and_enhance(self, face_bgr: np.ndarray) -> Image.Image:
        """AI upscale + full crystal-clear enhancement pipeline."""

        # ── Upscale ───────────────────────────────────────────────────────────
        if self._esrgan:
            upscaled_bgr, _ = self._esrgan.enhance(face_bgr, outscale=self.scale)
        else:
            h, w = face_bgr.shape[:2]
            upscaled_bgr = cv2.resize(
                face_bgr,
                (w * self.scale, h * self.scale),
                interpolation=cv2.INTER_LANCZOS4
            )

        # BGR → PIL RGB
        pil = Image.fromarray(cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB))

        # ── Enhancement pipeline ──────────────────────────────────────────────

        # 1. Unsharp mask — crispness
        pil = pil.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=2))

        # 2. Adaptive edge sharpening
        pil = self._adaptive_sharpen(pil)

        # 3. Denoise (gentle — preserves detail)
        arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        arr = cv2.fastNlMeansDenoisingColored(arr, None, 3, 3, 7, 21)
        pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

        # 4. Color + contrast + brightness
        pil = ImageEnhance.Color(pil).enhance(1.10)
        pil = ImageEnhance.Contrast(pil).enhance(1.08)
        pil = ImageEnhance.Brightness(pil).enhance(1.03)
        pil = ImageEnhance.Sharpness(pil).enhance(1.4)

        # 5. Final micro-clarity pass
        pil = pil.filter(ImageFilter.UnsharpMask(radius=0.5, percent=70, threshold=0))

        return pil

    @staticmethod
    def _adaptive_sharpen(pil: Image.Image) -> Image.Image:
        """Sharpen only edges, not flat/noisy areas."""
        img_np = np.array(pil).astype(np.float32)
        gray   = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        lap    = cv2.Laplacian(gray, cv2.CV_32F)
        mask   = np.clip(np.abs(lap) / 100.0, 0, 1)
        mask   = np.stack([mask] * 3, axis=-1)

        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        sharp  = cv2.filter2D(img_np, -1, kernel)

        out = img_np + mask * (sharp - img_np) * 0.45
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    # ── ANNOTATION ────────────────────────────────────────────────────────────

    def _draw_annotations(
        self, frame: np.ndarray, boxes: List[Tuple]
    ) -> np.ndarray:
        """Draw bounding boxes + face region on the overview frame."""
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            person_h = y2 - y1
            face_end_y = y1 + int(person_h * 0.32)

            # Full person box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 50), 2)

            # Face crop region highlight
            cv2.rectangle(frame, (x1, y1), (x2, face_end_y), (0, 120, 255), 2)

            # Label
            label = f"Person {i+1}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 50), 2)
            cv2.putText(frame, "FACE", (x1 + 4, y1 + int(person_h * 0.15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2)

        return frame

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ── BATCH MODE ────────────────────────────────────────────────────────────

    def batch_process(self, input_dir: str, output_dir: str):
        """Process every image in a folder."""
        exts   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files  = [f for f in Path(input_dir).iterdir()
                  if f.suffix.lower() in exts]
        self._log(f"\n[📂] {len(files)} images found in {input_dir}")

        for i, f in enumerate(files, 1):
            self._log(f"\n{'='*55}")
            self._log(f"[{i}/{len(files)}] {f.name}")
            try:
                out = str(Path(output_dir) / f.stem)
                self.process(str(f), output_dir=out)
            except Exception as e:
                self._log(f"[❌] Error: {e}")

        self._log(f"\n[🏁] Batch complete → {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="🎯 CCTV Face Crop + God-Level AI Upscaler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python cctv_face_upscale.py -i cctv.png
  python cctv_face_upscale.py -i cctv.png -s 4 --show
  python cctv_face_upscale.py -i ./cctv_frames/ -o ./faces/ -s 4
        """
    )
    p.add_argument("-i", "--input",   required=True,
                   help="Input image file or folder")
    p.add_argument("-o", "--output",  default=None,
                   help="Output folder (auto-created if omitted)")
    p.add_argument("-s", "--scale",   type=int, default=4, choices=[2, 4, 8],
                   help="Upscale factor (default: 4)")
    p.add_argument("--conf",          type=float, default=0.35,
                   help="YOLO confidence threshold (default: 0.35)")
    p.add_argument("--padding",       type=float, default=0.35,
                   help="Face crop padding ratio (default: 0.35)")
    p.add_argument("--show",          action="store_true",
                   help="Display each result after processing")
    p.add_argument("--quiet",         action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    upscaler = CCTVFaceUpscaler(
        scale=args.scale,
        face_padding=args.padding,
        conf_threshold=args.conf,
        verbose=not args.quiet,
    )

    in_path = Path(args.input)
    if in_path.is_dir():
        out = args.output or str(in_path.parent / (in_path.name + "_faces"))
        upscaler.batch_process(str(in_path), out)
    else:
        upscaler.process(str(in_path), output_dir=args.output, show=args.show)


if __name__ == "__main__":
    main()