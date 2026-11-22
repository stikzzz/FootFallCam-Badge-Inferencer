"""Draw tracked person boxes with crop scores, badge locations, or badged labels."""

import argparse
from collections import deque
import csv
from contextlib import ExitStack, contextmanager
import math
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import cv2
from PIL import Image
import torch

from staff_crop_model import build_model, to_tensor


ROOT = Path(__file__).resolve().parent
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = {"A": (235, 180, 66), "B": (205, 122, 233), "other": (145, 145, 145)}


class CropClassifier:
    def __init__(self, checkpoint, backend):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.classes = data["classes"]
        self.staff_classes = set(data["staff_classes"])
        self.can_score_staff = bool(set(self.classes) - self.staff_classes)
        self.size = data["image_size"]
        self.device = torch.device("cpu")
        if backend == "directml":
            import torch_directml

            if not torch_directml.is_available():
                raise RuntimeError("DirectML is unavailable. Use --classifier-device cpu.")
            self.device = torch_directml.device()
            print(f"Crop classifier: {torch_directml.device_name(0).strip(chr(0))} (DirectML)")
        else:
            print("Crop classifier: CPU")
        self.model = build_model(len(self.classes))
        self.model.load_state_dict(data["model_state"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def classify(self, crops):
        if not crops:
            return []
        tensors = [
            to_tensor(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), self.size)
            for crop in crops
        ]
        probabilities = self.model(torch.stack(tensors).to(self.device)).cpu().softmax(dim=1)
        return [dict(zip(self.classes, row)) for row in probabilities.tolist()]


class BadgeDetector:
    def __init__(self, weights, size, confidence):
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="detect")
        if list(self.model.names.values()) != ["badge"]:
            raise ValueError("--badge-detector requires a single-class model trained on 'badge'.")
        self.size = size
        self.confidence = confidence

    def detect(self, crops):
        if not crops:
            return []
        results = self.model.predict(crops, imgsz=self.size, conf=self.confidence,
                                     max_det=1, device="cpu", verbose=False)
        return [
            next(iter(bounded_detections(result, crop.shape[1], crop.shape[0])), None)
            for result, crop in zip(results, crops)
        ]


def badge_locations(detections, badges):
    for ((x1, y1, _, _), _, track_id), badge in zip(detections, badges):
        if badge is not None:
            (bx1, by1, bx2, by2), confidence, _ = badge
            yield (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2), confidence, track_id


def badge_overlap(first, second):
    x1, y1, x2, y2 = first
    a1, b1, a2, b2 = second
    intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
    return intersection / union


def assign_badges(detections, badges):
    tracked = [(bounds, track_id) for bounds, _, track_id in detections if track_id is not None]
    assigned = {}
    used = []
    for bounds, score, _ in sorted(badges, key=lambda badge: badge[1], reverse=True):
        if any(badge_overlap(bounds, box) >= 0.4 for box in used):
            continue
        bx1, by1, bx2, by2 = bounds
        center_x, center_y = (bx1 + bx2) / 2, (by1 + by2) / 2
        candidates = [
            (math.hypot((center_x - x1) / (x2 - x1) - 0.5,
                        (center_y - y1) / (y2 - y1) - 0.5), track_id)
            for (x1, y1, x2, y2), track_id in tracked
            if x1 <= center_x < x2 and y1 <= center_y < y2
        ]
        if candidates:
            _, track_id = min(candidates)
            if track_id not in assigned:
                assigned[track_id] = bounds, score
            used.append(bounds)
    return assigned


class BadgeTrackLabels:
    def __init__(self, hits, window):
        self.hits = hits
        self.window = window
        self.pending = {}
        self.confirmed = set()

    def update(self, frame_index, assignments):
        for track_id, sightings in list(self.pending.items()):
            while sightings and frame_index - sightings[0] >= self.window:
                sightings.popleft()
            if not sightings:
                del self.pending[track_id]
        for track_id in assignments.keys() - self.confirmed:
            sightings = self.pending.setdefault(track_id, deque())
            sightings.append(frame_index)
            if len(sightings) >= self.hits:
                self.confirmed.add(track_id)
                del self.pending[track_id]


def load_detector(weights, size, backend):
    from openvino import Core
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    devices = Core().available_devices
    has_gpu = any(device.startswith("GPU") for device in devices)
    if backend == "gpu" and not has_gpu:
        raise RuntimeError(f"OpenVINO GPU unavailable. Devices: {devices}. Use --detector-device cpu.")
    device = "xpu" if has_gpu and backend != "cpu" else "cpu"
    print(f"YOLO detector: OpenVINO {'Intel GPU' if device == 'xpu' else 'CPU'}")
    if weights.is_dir():
        exported = weights
    else:
        exported = weights.with_name(f"{weights.stem}_openvino_model")
        if not (exported / f"{weights.stem}.xml").is_file() or not (exported / "metadata.yaml").is_file():
            weights.parent.mkdir(parents=True, exist_ok=True)
            model = YOLO(str(weights), task="detect")
            exported = Path(model.export(format="openvino", imgsz=size, dynamic=True, device="cpu"))
    return YOLO(str(exported), task="detect"), device


def bounded_detections(result, width, height):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
    detections = []
    for bounds, confidence, track_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), ids):
        x1 = max(0, math.floor(bounds[0]))
        y1 = max(0, math.floor(bounds[1]))
        x2 = min(width, math.ceil(bounds[2]))
        y2 = min(height, math.ceil(bounds[3]))
        if x2 > x1 and y2 > y1:
            detections.append(((x1, y1, x2, y2), confidence, track_id))
    return detections


def draw_label(frame, bounds, label, color, thickness):
    x1, y1, x2, y2 = bounds
    cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), color, thickness)
    (width, height), baseline = cv2.getTextSize(label, FONT, 0.45, 1)
    left = min(x1, max(0, frame.shape[1] - width - 8))
    top = max(0, y1 - height - baseline - 6)
    cv2.rectangle(frame, (left, top), (left + width + 8, top + height + baseline + 6), color, -1)
    cv2.putText(frame, label, (left + 4, top + height + 2), FONT, 0.45, (15, 15, 15), 1, cv2.LINE_AA)


@contextmanager
def mp4_writer(path, video, width, height, fps):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required to save the annotated H.264 MP4. Add ffmpeg to PATH.")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "pipe:0", "-i", str(video),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(path),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            yield process.stdin
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("FFmpeg could not write the annotated MP4.")


def run(args):
    torch.set_num_threads(2)
    video = args.video.resolve()
    badge_mode = args.badge_detector is not None
    weights = args.badge_detector or args.checkpoint
    suffix = "badged_tracks" if args.label_badged else "badge_detection" if badge_mode else "staff_overlay"
    output = (args.output or ROOT / "output" / f"{video.stem}_{suffix}.mp4").resolve()
    csv_path = args.csv.resolve() if args.csv else None
    is_badge_export = badge_mode and weights.is_dir() and (weights / "metadata.yaml").is_file()
    if not video.is_file() or not (weights.is_file() or is_badge_export):
        raise FileNotFoundError(f"Check input video and checkpoint paths: {video}, {weights}")
    if output.suffix.lower() != ".mp4" or output in {video, weights.resolve()}:
        raise ValueError("Output must be a separate .mp4 file.")
    if csv_path and (csv_path.suffix.lower() != ".csv" or csv_path in {output, video, weights.resolve()}):
        raise ValueError("CSV must be a separate .csv file.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    if badge_mode:
        badge_detector = BadgeDetector(weights, args.badge_imgsz, args.badge_conf)
    else:
        classifier = CropClassifier(weights, args.classifier_device)
    detector, detector_device = load_detector(args.detector.resolve(), args.imgsz, args.detector_device)
    staff_tracks = set(args.staff_track)
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot read {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Video needs at least one readable frame and a valid frame rate.")
        height, width = frame.shape[:2]
        badge_labels = BadgeTrackLabels(args.badge_track_hits, max(args.badge_track_hits, math.ceil(fps)))
        samples = 0
        frame_index = 0
        with ExitStack() as stack:
            temp = Path(stack.enter_context(TemporaryDirectory(dir=output.parent)))
            temp_video = temp / "overlay.mp4"
            csv_file = None
            if csv_path:
                csv_temp = Path(stack.enter_context(TemporaryDirectory(dir=csv_path.parent))) / "detections.csv"
                csv_file = csv_temp.open("w", newline="", encoding="utf-8")
                stack.callback(csv_file.close)
                csv_writer = csv.writer(csv_file)
                if args.label_badged:
                    csv_writer.writerow(["frame", "time_s", "track_id", "x1", "y1", "x2", "y2",
                                         "person_score", "badged", "badge_x1", "badge_y1",
                                         "badge_x2", "badge_y2", "badge_score"])
                elif badge_mode:
                    csv_writer.writerow(["frame", "time_s", "track_id", "badge_x1", "badge_y1",
                                         "badge_x2", "badge_y2", "badge_score"])
                else:
                    csv_writer.writerow(["frame", "time_s", "track_id", "x1", "y1", "x2", "y2",
                                         "person_score", "crop_label", "crop_score", "staff_score", "manual_staff"])
            with mp4_writer(temp_video, video, width, height, fps) as writer:
                while ok and (args.max_frames is None or frame_index < args.max_frames):
                    result = detector.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                                            conf=args.conf, imgsz=args.imgsz, device=detector_device,
                                            verbose=False)[0]
                    detections = bounded_detections(result, width, height)
                    crops = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2), _, _ in detections]
                    if badge_mode:
                        badges = badge_detector.detect(crops)
                    if args.label_badged:
                        assigned = assign_badges(detections, badge_locations(detections, badges))
                        badge_labels.update(frame_index, assigned)
                        for bounds, confidence, track_id in detections:
                            if track_id is None:
                                continue
                            confirmed = track_id in badge_labels.confirmed
                            label = f"ID {track_id} | Badged" if confirmed else f"ID {track_id}"
                            color = (117, 230, 132) if confirmed else (210, 185, 105)
                            draw_label(frame, bounds, label, color, 2 if confirmed else 1)
                            badge = assigned.get(track_id)
                            evidence = [""] * 5
                            if badge is not None:
                                badge_bounds, badge_score = badge
                                bx1, by1, bx2, by2 = badge_bounds
                                cv2.rectangle(frame, (bx1, by1), (bx2 - 1, by2 - 1), (36, 212, 255), 2)
                                evidence = [*badge_bounds, f"{badge_score:.4f}"]
                            if csv_file:
                                csv_writer.writerow([frame_index, f"{frame_index / fps:.3f}", track_id,
                                                     *bounds, f"{confidence:.4f}", int(confirmed), *evidence])
                    elif badge_mode:
                        for badge_bounds, confidence, track_id in badge_locations(detections, badges):
                            label = f"Badge {confidence:.0%}"
                            if track_id is not None:
                                label += f" | ID {track_id}"
                            draw_label(frame, badge_bounds, label, (117, 230, 132), 2)
                            if csv_file:
                                csv_writer.writerow([frame_index, f"{frame_index / fps:.3f}", track_id,
                                                     *badge_bounds, f"{confidence:.4f}"])
                    scores = [] if badge_mode else classifier.classify(crops)
                    for (bounds, confidence, track_id), probabilities in zip(detections, scores):
                        label = max(probabilities, key=probabilities.get)
                        score = probabilities[label]
                        staff_score = (sum(probabilities[name] for name in classifier.staff_classes)
                                       if classifier.can_score_staff else None)
                        confirmed = track_id in staff_tracks
                        text = f"ID {track_id if track_id is not None else '?'} | {label} {score:.0%}"
                        if confirmed:
                            text = "Staff | " + text
                        color = (117, 230, 132) if confirmed else COLORS.get(label, (220, 220, 220))
                        draw_label(frame, bounds, text, color, 3 if confirmed else 2)
                        if csv_file:
                            csv_writer.writerow([frame_index, f"{frame_index / fps:.3f}", track_id,
                                                 *bounds, f"{confidence:.4f}", label, f"{score:.4f}",
                                                 "" if staff_score is None else f"{staff_score:.4f}",
                                                 int(confirmed)])
                    status = f"Frame {frame_index} | {frame_index / fps:.2f}s"
                    if not badge_mode and not classifier.can_score_staff:
                        status += " | A/B scores show appearance"
                    draw_label(frame, (0, 0, 1, 1), status, (235, 235, 235), 1)
                    writer.write(frame.tobytes())
                    samples += len(detections)
                    frame_index += 1
                    if frame_index % 100 == 0:
                        print(f"Processed {frame_index}/{min(count, args.max_frames or count)} frames", flush=True)
                    ok, frame = cap.read()
            if csv_file:
                csv_file.close()
                csv_temp.replace(csv_path)
            temp_video.replace(output)
        print(f"Saved {output} ({frame_index} frames, {samples} person boxes)")
        if args.label_badged:
            print(f"Badged track IDs: {', '.join(map(str, sorted(badge_labels.confirmed))) or 'none'}")
        if csv_path:
            print(f"Saved {csv_path}")
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=ROOT / "sample.mp4")
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--checkpoint", type=Path, default=ROOT / "output" / "staff_crop_classifier.pt")
    checkpoint.add_argument("--badge-detector", type=Path, help="YOLO badge detection checkpoint")
    parser.add_argument("--label-badged", action="store_true",
                        help="Assign badges to the nearest containing track and label confirmed tracks")
    parser.add_argument("--badge-track-hits", type=int, default=3,
                        help="Badge sightings within one second required to mark a track as badged")
    parser.add_argument("--badge-imgsz", type=int, default=320, help="Badge detector input size")
    parser.add_argument("--badge-conf", type=float, default=0.35, help="Minimum badge detection score")
    parser.add_argument("--detector", type=Path, default=ROOT / "models" / "yolo26m.pt",
                        help="YOLO .pt weights or an exported OpenVINO model directory")
    parser.add_argument("--output", type=Path, help="MP4 output path")
    parser.add_argument("--csv", type=Path, help="Optional frame, box, and track CSV")
    parser.add_argument("--detector-device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--classifier-device", choices=["directml", "cpu"], default="cpu",
                        help="CPU is the default for the small crop model. Opt in to DirectML.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO minimum person detection score")
    parser.add_argument("--max-frames", type=int, help="Limit the run to the first N frames")
    parser.add_argument("--staff-track", type=int, action="append", default=[],
                        help="Highlight a manually confirmed track ID. Repeat for fragmented tracks.")
    args = parser.parse_args()
    if args.label_badged and (args.badge_detector is None or args.badge_track_hits <= 0):
        parser.error("--label-badged requires a badge detector and a positive hit count.")
    if args.imgsz <= 0 or not 0 < args.conf <= 1 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error("Use a positive image size and frame limit, and a confidence in (0, 1].")
    if args.badge_imgsz <= 0 or not 0 < args.badge_conf <= 1:
        parser.error("Use a positive badge image size and a badge confidence in (0, 1].")
    run(args)


if __name__ == "__main__":
    main()
