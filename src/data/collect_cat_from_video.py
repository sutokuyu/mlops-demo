import argparse
import os
import re
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config, resolve_config_path

CONFIG = load_config(PROJECT_ROOT / "configs" / "config.yaml")
PICTURES_ROOT = resolve_config_path(CONFIG["paths"]["pictures"])
VIDEO_ROOT = resolve_config_path(CONFIG["paths"]["videos"])

MODEL_PATH = CONFIG["models"]["detection_model"]
CONFIDENCE_THRESHOLD = CONFIG["collect"]["confidence_threshold"]
MIN_BOX_WIDTH = CONFIG["collect"]["min_box_width"]
MIN_BOX_HEIGHT = CONFIG["collect"]["min_box_height"]
SAVE_INTERVAL = CONFIG["collect"]["save_interval_seconds"]
PADDING = CONFIG["collect"]["padding"]

INDEX_RE = re.compile(r"^(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def next_image_index(output_dir: str) -> int:
    existing = [
        int(m.group(1)) for f in os.listdir(output_dir) if (m := INDEX_RE.match(f)) is not None
    ]
    return max(existing, default=0) + 1


def collect_cat_from_video(video_path: str, label: str, interval: float = SAVE_INTERVAL) -> None:
    output_dir = PICTURES_ROOT / label
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Video cat collector")
    print("=" * 60)
    print(f"Video       : {video_path}")
    print(f"Label       : {label}")
    print(f"Output      : {output_dir}")
    print(f"Interval    : {interval}s")
    print(f"Confidence  : {CONFIDENCE_THRESHOLD}")
    print("=" * 60)

    if not os.path.isfile(video_path):
        resolved_video_path = VIDEO_ROOT / video_path
        if resolved_video_path.is_file():
            video_path = str(resolved_video_path)
        else:
            raise FileNotFoundError(f"Video file not found: {video_path}")

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps * interval)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"Video FPS    : {fps:.2f}")
    print(f"Frame step   : {frame_interval}")
    print(f"Total frames : {total_frames}")
    print("Start processing...")
    print()

    saved_count = next_image_index(output_dir) - 1
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval != 0:
            frame_index += 1
            continue

        results = model(frame, verbose=False)
        result = results[0]

        best_cat = None
        best_conf = 0.0

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                if (
                    class_name == "cat"
                    and confidence >= CONFIDENCE_THRESHOLD
                    and confidence > best_conf
                ):
                    best_conf = confidence
                    best_cat = box.xyxy[0].cpu().numpy()

        if best_cat is None:
            frame_index += 1
            continue

        x1, y1, x2, y2 = map(int, best_cat)
        height, width = frame.shape[:2]
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(width, x2 + PADDING)
        y2 = min(height, y2 + PADDING)
        box_width = x2 - x1
        box_height = y2 - y1

        if box_width < MIN_BOX_WIDTH or box_height < MIN_BOX_HEIGHT:
            frame_index += 1
            continue

        cat_image = frame[y1:y2, x1:x2]
        if cat_image.size == 0:
            frame_index += 1
            continue

        saved_count += 1
        filename = output_dir / f"{saved_count:04d}.jpg"
        cv2.imwrite(str(filename), cat_image)

        print(
            f"[{saved_count:4d}] saved {filename} (confidence={best_conf:.2f}, frame={frame_index})"
        )
        frame_index += 1

    cap.release()

    print()
    print("=" * 60)
    print("Video collection completed.")
    print(f"Images saved: {saved_count}")
    print(f"Directory   : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect cat images from a video at a fixed time interval."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--label", required=True, help="Cat label / output subfolder name.")
    parser.add_argument(
        "--interval", type=float, default=SAVE_INTERVAL, help="Seconds between frames to check."
    )
    args = parser.parse_args()

    collect_cat_from_video(
        video_path=args.video,
        label=args.label,
        interval=args.interval,
    )
