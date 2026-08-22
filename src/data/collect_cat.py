import argparse
import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config, resolve_config_path

# ============================================================
# Configuration
# ============================================================

CONFIG = load_config(PROJECT_ROOT / "configs" / "config.yaml")
PICTURES_ROOT = resolve_config_path(CONFIG["paths"]["pictures"])
RTSP_URL = CONFIG["camera"]["rtsp_url"]
MODEL_PATH = CONFIG["models"]["detection_model"]
CONFIDENCE_THRESHOLD = CONFIG["collect"]["confidence_threshold"]
SAVE_INTERVAL = CONFIG["collect"]["save_interval_seconds"]
MIN_BOX_WIDTH = CONFIG["collect"]["min_box_width"]
MIN_BOX_HEIGHT = CONFIG["collect"]["min_box_height"]
PADDING = CONFIG["collect"]["padding"]


# ============================================================
# Main
# ============================================================


def collect_images(label: str, target: int):
    output_dir = PICTURES_ROOT / label
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cat image collector")
    print("=" * 60)
    print(f"Label       : {label}")
    print(f"Target      : {target}")
    print(f"Output      : {output_dir}")
    print(f"Confidence  : {CONFIDENCE_THRESHOLD}")
    print(f"Interval    : {SAVE_INTERVAL}s")
    print("=" * 60)

    print("\nLoading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("Connecting to RTSP camera...")

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError("Unable to open RTSP stream")

    print("Camera connected.")
    print()
    print("Start recording.")
    print("Move the camera slowly around the cat.")
    print("Press Ctrl+C to stop manually.")
    print()

    saved_count = len(
        [f for f in os.listdir(output_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )

    last_save_time = 0

    try:
        while saved_count < target:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame. Retrying...")
                time.sleep(0.2)
                continue

            # Run YOLO
            results = model(frame, verbose=False)

            result = results[0]

            best_cat = None
            best_conf = 0

            # Find the cat with highest confidence
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

            # No cat detected
            if best_cat is None:
                continue

            # Don't save too frequently
            current_time = time.time()

            if current_time - last_save_time < SAVE_INTERVAL:
                continue

            x1, y1, x2, y2 = map(int, best_cat)

            # Make sure coordinates are inside image
            height, width = frame.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width < MIN_BOX_WIDTH or box_height < MIN_BOX_HEIGHT:
                continue

            # Add a little padding around the cat
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(width, x2 + PADDING)
            y2 = min(height, y2 + PADDING)

            # Crop cat
            cat_image = frame[y1:y2, x1:x2]

            if cat_image.size == 0:
                continue

            saved_count += 1

            filename = output_dir / f"{saved_count:04d}.jpg"

            cv2.imwrite(str(filename), cat_image)

            last_save_time = current_time

            print(f"[{saved_count:4d}/{target}] saved {filename} (confidence={best_conf:.2f})")

        print()
        print("=" * 60)
        print("Collection completed!")
        print(f"Images saved: {saved_count}")
        print(f"Directory   : {output_dir}")
        print("=" * 60)

    except KeyboardInterrupt:
        print()
        print("Collection stopped manually.")

    finally:
        cap.release()


# ============================================================
# Command line
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect cat images automatically using YOLO.")

    parser.add_argument("--label", required=True, help="Cat label, e.g. cat_a or cat_b")

    parser.add_argument("--target", type=int, default=300, help="Number of images to collect")

    args = parser.parse_args()

    collect_images(label=args.label, target=args.target)
