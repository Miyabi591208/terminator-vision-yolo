import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any

import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Detection:
    xyxy: Tuple[int, int, int, int]
    cls_id: int
    cls_name: str
    conf: float

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) // 2, (y1 + y2) // 2


def _names_lookup(names: Any, cls_id: int) -> str:
    try:
        if isinstance(names, dict):
            return str(names.get(int(cls_id), str(cls_id)))
        return str(names[int(cls_id)])
    except Exception:
        return str(cls_id)


def extract_detections(result: Any) -> List[Detection]:
    dets: List[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return dets

    if len(boxes) == 0:
        return dets

    xyxy = boxes.xyxy
    conf = boxes.conf
    cls = boxes.cls

    xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
    conf_np = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
    cls_np = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls)

    names = getattr(result, "names", None) or {}

    for i in range(xyxy_np.shape[0]):
        x1, y1, x2, y2 = [int(v) for v in xyxy_np[i].tolist()]
        cls_id = int(cls_np[i])
        cls_name = _names_lookup(names, cls_id)
        c = float(conf_np[i])
        dets.append(Detection((x1, y1, x2, y2), cls_id, cls_name, c))
    return dets


def pick_target(dets: List[Detection], target_class: str) -> Optional[Detection]:
    if not dets:
        return None
    target_class = (target_class or "").strip().lower()
    candidates = [d for d in dets if d.cls_name.lower() == target_class] if target_class else []
    if candidates:
        return max(candidates, key=lambda d: (d.area * d.conf))
    return max(dets, key=lambda d: d.conf)


# -----------------------------
# Visual effect
# -----------------------------
def apply_terminator_grade(frame_bgr: np.ndarray, t: float, add_edges=True, add_scanlines=True) -> np.ndarray:
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    b = img[:, :, 0].astype(np.float32) * 0.25
    g = img[:, :, 1].astype(np.float32) * 0.45
    r = img[:, :, 2].astype(np.float32) * 1.25 + 15.0
    img = np.clip(np.stack([b, g, r], axis=2), 0, 255).astype(np.uint8)

    if add_edges:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 1.0, edges_bgr, 0.35, 0)

    if add_scanlines:
        spacing = 4
        img[::spacing, :, :] = (img[::spacing, :, :] * 0.55).astype(np.uint8)

        bar_y = int((t * 120) % max(1, h))
        y1 = max(0, bar_y - 8)
        y2 = min(h, bar_y + 8)
        if y2 > y1:
            bar = img[y1:y2].astype(np.float32)
            bar[:, :, 2] = np.clip(bar[:, :, 2] * 1.35 + 20, 0, 255)
            img[y1:y2] = bar.astype(np.uint8)

    noise = np.random.randint(-8, 9, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    vign = np.clip(1.0 - 0.35 * dist, 0.65, 1.0).astype(np.float32)
    img = (img.astype(np.float32) * vign[:, :, None]).astype(np.uint8)

    return img


def draw_hud(img: np.ndarray, dets: List[Detection], target: Optional[Detection], fps: float, frame_idx: int) -> np.ndarray:
    out = img
    h, w = out.shape[:2]

    # Grid
    for x in range(0, w, 80):
        cv2.line(out, (x, 0), (x, h), (0, 0, 50), 1)
    for y in range(0, h, 60):
        cv2.line(out, (0, y), (w, y), (0, 0, 50), 1)

    # Header
    cv2.putText(out, "CYBERDYNE VISION SYSTEM", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2, cv2.LINE_AA)
    cv2.putText(out, f"FPS:{fps:5.1f}  FRAME:{frame_idx}", (15, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2, cv2.LINE_AA)

    # Detections
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        is_target = (target is not None) and (d is target)

        color = (0, 0, 255) if is_target else (0, 0, 140)
        thickness = 3 if is_target else 1

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        tag = f"{d.cls_name.upper()} {d.conf:.2f}"
        if is_target:
            tag += "  [TERMINATE]"

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1 - 8)
        cv2.rectangle(out, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 4), (0, 0, 0), -1)
        cv2.putText(out, tag, (x1 + 3, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 230), 2, cv2.LINE_AA)

    # Crosshair + status
    if target is not None:
        cx, cy = target.center
        size = 18
        cv2.line(out, (cx - size, cy), (cx + size, cy), (0, 0, 255), 2)
        cv2.line(out, (cx, cy - size), (cx, cy + size), (0, 0, 255), 2)
        cv2.circle(out, (cx, cy), 28, (0, 0, 255), 2)

        status = f"TARGET:{target.cls_name.upper()}  CONF:{target.conf:.2f}"
        cv2.putText(out, status, (15, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2, cv2.LINE_AA)
        cv2.putText(out, "STATUS: LOCKED", (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2, cv2.LINE_AA)
    else:
        cv2.putText(out, "STATUS: SEARCHING", (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2, cv2.LINE_AA)

    return out


# -----------------------------
# Overlay video compositing
# -----------------------------
class OverlayPlayer:
    """
    overlay動画をループ再生し、輝度からマスクを作ってscreen合成します。
    黒に近いほど透明として扱います。
    """
    def __init__(self, overlay_path: str, alpha: float = 0.85, threshold: int = 10):
        self.overlay_path = overlay_path
        self.alpha = float(alpha)
        self.threshold = int(threshold)
        self.cap = cv2.VideoCapture(overlay_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open overlay video: {overlay_path}")

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("Overlay video read failed even after rewind.")
        return frame

    def release(self) -> None:
        self.cap.release()

    def composite_screen(self, base_bgr: np.ndarray, overlay_bgr: np.ndarray) -> np.ndarray:
        if overlay_bgr.shape[:2] != base_bgr.shape[:2]:
            overlay_bgr = cv2.resize(overlay_bgr, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mask = gray / 255.0

        th = self.threshold / 255.0
        mask = np.clip((mask - th) / (1.0 - th), 0.0, 1.0)
        mask = (mask * self.alpha).astype(np.float32)

        base = base_bgr.astype(np.float32) / 255.0
        ov = overlay_bgr.astype(np.float32) / 255.0

        screen = 1.0 - (1.0 - base) * (1.0 - ov)

        mask3 = np.repeat(mask[:, :, None], 3, axis=2)
        out = base * (1.0 - mask3) + screen * mask3
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return out


# -----------------------------
# Audio (pygame)
# -----------------------------
def start_audio_loop(audio_path: Optional[str], volume: float) -> Optional[object]:
    if not audio_path:
        return None
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.set_volume(max(0.0, min(1.0, float(volume))))
        pygame.mixer.music.play(-1)  # loop
        return pygame
    except Exception as e:
        print(f"[WARN] Audio disabled (pygame failed): {e}")
        return None


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Webcam YOLO Terminator HUD + overlay video + looping audio (local).")
    ap.add_argument("--cam", type=int, default=0, help="webcam index (0,1,2...)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--model", default="yolov8n.pt", help="e.g. yolov8n.pt / your custom .pt")
    ap.add_argument("--device", default=None, help="e.g. 'cpu', '0' (GPU). empty=auto")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)

    ap.add_argument("--target", default="person", help="class name to lock-on (default: person)")
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--no-scanlines", action="store_true")

    ap.add_argument("--overlay-video", default=None, help="optional overlay mp4 (e.g. overlay1.mp4)")
    ap.add_argument("--overlay-alpha", type=float, default=0.85)
    ap.add_argument("--overlay-threshold", type=int, default=10)

    ap.add_argument("--audio", default=None, help="optional audio mp3 (loop) (e.g. overlay1.mp3)")
    ap.add_argument("--volume", type=float, default=0.6)

    ap.add_argument("--out", default=None, help="optional output mp4 path (records)")
    ap.add_argument("--out-fps", type=float, default=30.0)

    args = ap.parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index: {args.cam}")

    # Try set camera resolution (not all cams honor this)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))

    overlay_player = None
    if args.overlay_video:
        overlay_player = OverlayPlayer(args.overlay_video, alpha=args.overlay_alpha, threshold=args.overlay_threshold)

    pygame_mod = start_audio_loop(args.audio, args.volume)

    writer = None
    if args.out:
        # We'll initialize writer after we know actual frame size
        writer = None

    t0 = time.perf_counter()
    last = t0
    fps = 0.0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            # YOLO
            res = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
                device=args.device,
            )[0]
            dets = extract_detections(res)
            target = pick_target(dets, args.target)

            # base effects + HUD
            stylized = apply_terminator_grade(
                frame, now - t0,
                add_edges=not args.no_edges,
                add_scanlines=not args.no_scanlines
            )
            hud = draw_hud(stylized, dets, target, fps, frame_idx)

            # overlay mp4
            if overlay_player is not None:
                ov = overlay_player.read()
                hud = overlay_player.composite_screen(hud, ov)

            # record if requested
            if args.out:
                if writer is None:
                    h, w = hud.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.out, fourcc, float(args.out_fps), (w, h))
                writer.write(hud)

            cv2.imshow("Terminator Vision (press q / ESC)", hud)

            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if overlay_player is not None:
            overlay_player.release()
        cv2.destroyAllWindows()

        # stop audio
        try:
            if pygame_mod is not None:
                pygame_mod.mixer.music.stop()
                pygame_mod.mixer.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
