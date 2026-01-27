import argparse
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Any, Union

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    xyxy: Tuple[int, int, int, int]
    cls_id: int
    cls_name: str
    conf: float
    track_id: Optional[int] = None

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) // 2, (y1 + y2) // 2


def _names_lookup(names: Any, cls_id: int) -> str:
    # Ultralytics may expose names as dict[int,str] or list[str]
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

    try:
        n = len(boxes)
    except Exception:
        return dets
    if n == 0:
        return dets

    xyxy = boxes.xyxy
    conf = boxes.conf
    cls = boxes.cls

    xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
    conf_np = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
    cls_np = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls)

    # track ids (optional)
    track_ids_np = None
    if getattr(boxes, "is_track", False) and getattr(boxes, "id", None) is not None:
        tid = boxes.id
        track_ids_np = tid.int().cpu().numpy() if hasattr(tid, "cpu") else np.asarray(tid)

    names = getattr(result, "names", None) or {}

    for i in range(xyxy_np.shape[0]):
        x1, y1, x2, y2 = [int(v) for v in xyxy_np[i].tolist()]
        cls_id = int(cls_np[i])
        cls_name = _names_lookup(names, cls_id)
        c = float(conf_np[i])
        tid = int(track_ids_np[i]) if track_ids_np is not None else None
        dets.append(Detection((x1, y1, x2, y2), cls_id, cls_name, c, tid))
    return dets


def pick_target(dets: List[Detection], target_class: str) -> Optional[Detection]:
    if not dets:
        return None
    target_class = (target_class or "").strip().lower()

    candidates = [d for d in dets if d.cls_name.lower() == target_class] if target_class else []
    if candidates:
        # Prefer large & confident objects (more stable visually)
        return max(candidates, key=lambda d: (d.area * d.conf))
    # Fallback: highest confidence
    return max(dets, key=lambda d: d.conf)


def apply_terminator_grade(frame_bgr: np.ndarray, t: float, *, add_edges: bool = True, add_scanlines: bool = True) -> np.ndarray:
    """Cheap 'Terminator-ish' grade: red tint + scanlines + light noise + optional edges."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    # Red tint (BGR)
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)

    b *= 0.25
    g *= 0.45
    r = r * 1.25 + 15.0

    img = np.stack([b, g, r], axis=2)
    img = np.clip(img, 0, 255).astype(np.uint8)

    if add_edges:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 1.0, edges_bgr, 0.35, 0)

    if add_scanlines:
        # Darken every N-th line (scanlines)
        spacing = 4
        img[::spacing, :, :] = (img[::spacing, :, :] * 0.55).astype(np.uint8)

        # Moving bright scan bar
        bar_y = int((t * 120) % max(1, h))
        y1 = max(0, bar_y - 8)
        y2 = min(h, bar_y + 8)
        if y2 > y1:
            bar = img[y1:y2, :, :].astype(np.float32)
            bar[:, :, 2] = np.clip(bar[:, :, 2] * 1.35 + 20, 0, 255)
            img[y1:y2, :, :] = bar.astype(np.uint8)

    # Subtle noise
    noise = np.random.randint(-8, 9, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Vignette-ish dark corners (very light)
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

    # Header text
    cv2.putText(out, "CYBERDYNE VISION SYSTEM", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2, cv2.LINE_AA)
    cv2.putText(out, f"FPS:{fps:5.1f}  FRAME:{frame_idx}", (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2, cv2.LINE_AA)

    # Draw detections
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        is_target = (target is not None) and (d is target)

        color = (0, 0, 255) if is_target else (0, 0, 140)
        thickness = 3 if is_target else 1

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        tag = f"{d.cls_name.upper()} {d.conf:.2f}"
        if d.track_id is not None:
            tag += f" ID:{d.track_id}"
        if is_target:
            tag += "  [TERMINATE]"

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1 - 8)
        cv2.rectangle(out, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 4), (0, 0, 0), -1)
        cv2.putText(out, tag, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 230), 2, cv2.LINE_AA)

    # Crosshair on target
    if target is not None:
        cx, cy = target.center
        size = 18
        cv2.line(out, (cx - size, cy), (cx + size, cy), (0, 0, 255), 2)
        cv2.line(out, (cx, cy - size), (cx, cy + size), (0, 0, 255), 2)
        cv2.circle(out, (cx, cy), 28, (0, 0, 255), 2)

        status = f"TARGET:{target.cls_name.upper()}  CONF:{target.conf:.2f}"
        cv2.putText(out, status, (15, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2, cv2.LINE_AA)
        cv2.putText(out, "STATUS: LOCKED", (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2, cv2.LINE_AA)
    else:
        cv2.putText(out, "STATUS: SEARCHING", (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2, cv2.LINE_AA)

    return out


def parse_source(source: str) -> Union[int, str]:
    try:
        return int(source)  # "0" -> webcam 0
    except ValueError:
        return source


def iter_frames_from_capture(source: Union[int, str]) -> Iterable[np.ndarray]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def iter_frames_from_screen(monitor_index: int = 1) -> Iterable[np.ndarray]:
    import mss
    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]
        while True:
            sct_img = sct.grab(mon)          # BGRA
            frame = np.array(sct_img)[:, :, :3]  # BGR
            yield frame


def main() -> None:
    ap = argparse.ArgumentParser(description="Terminator-style HUD + YOLO detection (visual effect).")
    ap.add_argument("--model", default="yolo26n.pt", help="e.g. yolo26n.pt / yolov8n.pt / path/to/best.pt")
    ap.add_argument("--source", default="0", help="webcam index (0) / video path / 'screen'")
    ap.add_argument("--monitor", type=int, default=1, help="mss monitor index when --source screen (1=primary)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default=None, help="e.g. 'cpu', '0' (GPU). empty=auto")
    ap.add_argument("--target", default="person", help="class name to mark as TERMINATE (e.g. person, car)")
    ap.add_argument("--track", action="store_true", help="enable tracking (IDs)")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml / botsort.yaml")
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--no-scanlines", action="store_true")
    ap.add_argument("--out", default=None, help="optional output video path (e.g. out.mp4)")
    args = ap.parse_args()

    # Load model (fallback)
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[WARN] Failed to load model '{args.model}': {e}")
        print("[WARN] Falling back to 'yolov8n.pt' ...")
        model = YOLO("yolov8n.pt")

    # Frame source
    if args.source.strip().lower() == "screen":
        frame_iter = iter_frames_from_screen(args.monitor)
        display_name = f"Terminator HUD (screen monitor={args.monitor})"
    else:
        src = parse_source(args.source)
        frame_iter = iter_frames_from_capture(src)
        display_name = f"Terminator HUD (source={args.source})"

    writer = None
    out_path = args.out

    t0 = time.perf_counter()
    last = t0
    fps = 0.0
    frame_idx = 0

    for frame in frame_iter:
        now = time.perf_counter()
        dt = now - last
        last = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        # Inference
        if args.track:
            res = model.track(
                frame,
                persist=True,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                tracker=args.tracker,
                verbose=False,
                device=args.device,
            )[0]
        else:
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

        stylized = apply_terminator_grade(
            frame, now - t0,
            add_edges=not args.no_edges,
            add_scanlines=not args.no_scanlines
        )
        hud = draw_hud(stylized, dets, target, fps, frame_idx)

        cv2.imshow(display_name, hud)

        if out_path:
            if writer is None:
                h, w = hud.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
            writer.write(hud)

        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
