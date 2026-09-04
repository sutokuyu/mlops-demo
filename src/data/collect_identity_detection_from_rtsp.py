import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config, resolve_config_path

CONFIG = load_config(PROJECT_ROOT / "configs" / "config.yaml")
COLLECTION_CONFIG = CONFIG["identity_collection"]
IDENTITY_MODEL_PATH = resolve_config_path(CONFIG["models"]["identity_detection_model_path"])
EXPECTED_CLASSES = {0: "bagel", 1: "kurumi"}


@dataclass
class CameraState:
    name: str
    rtsp_url: str
    capture: cv2.VideoCapture | None = None
    last_inference_at: float = 0.0
    last_sample_at: float = 0.0
    last_connect_attempt_at: float = 0.0
    saved_count: int = 0
    sample_count: int = 0
    last_saved_detections: list[tuple[int, tuple[float, float, float, float]]] | None = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect Bagel/Kurumi RTSP detections as images, YOLO labels, and preview images.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="camera_names",
        help="Camera name to collect from. Repeat to select multiple cameras; default: all configured cameras.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=resolve_config_path(COLLECTION_CONFIG["output_dir"]),
    )
    parser.add_argument("--conf", type=float, default=COLLECTION_CONFIG["confidence_threshold"])
    parser.add_argument(
        "--inference-interval",
        dest="inference_interval",
        type=float,
        default=COLLECTION_CONFIG["inference_interval_seconds"],
        help="Seconds between model inferences for each camera.",
    )
    parser.add_argument(
        "--background-sample-interval",
        dest="background_sample_interval",
        type=float,
        default=COLLECTION_CONFIG["background_sample_interval_seconds"],
        help="Seconds between raw-frame samples when no cat identity is detected; set to 0 to disable sampling.",
    )
    parser.add_argument("--imgsz", type=int, default=COLLECTION_CONFIG["imgsz"])
    parser.add_argument("--device", default=CONFIG["training"]["device"])
    parser.add_argument(
        "--max-background-samples-per-camera",
        type=int,
        default=COLLECTION_CONFIG["max_background_samples_per_camera"],
        help="Maximum raw-frame samples to save per camera; 0 means no limit.",
    )
    parser.add_argument(
        "--max-images-per-camera",
        type=int,
        default=0,
        help="Stop after this many saved images per camera; 0 means run until interrupted.",
    )
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=0.95,
        help=(
            "Skip saving a detection when all boxes match the previous saved detection at this "
            "IoU or higher; set to 0 to disable duplicate suppression."
        ),
    )
    return parser.parse_args(argv)


def select_cameras(camera_names: list[str] | None) -> list[CameraState]:
    cameras = []
    for camera_config in COLLECTION_CONFIG["cameras"]:
        name = camera_config["name"]
        if camera_names is not None and name not in camera_names:
            continue
        rtsp_url = camera_config["rtsp_url"]
        if rtsp_url:
            cameras.append(CameraState(name=name, rtsp_url=rtsp_url))

    if not cameras:
        selected = ", ".join(camera_names) if camera_names else "configured cameras"
        raise RuntimeError(f"No RTSP URLs configured for: {selected}")
    return cameras


def connect(camera: CameraState) -> None:
    camera.last_connect_attempt_at = time.monotonic()
    if camera.capture is not None:
        camera.capture.release()
    camera.capture = cv2.VideoCapture(camera.rtsp_url, cv2.CAP_FFMPEG)
    if camera.capture.isOpened():
        print(f"[{camera.name}] connected")
    else:
        camera.capture.release()
        camera.capture = None
        print(f"[{camera.name}] unable to connect; will retry")


def xyxy_to_yolo(box, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = (float(value) for value in box)
    return (
        ((x_min + x_max) / 2) / image_width,
        ((y_min + y_max) / 2) / image_height,
        (x_max - x_min) / image_width,
        (y_max - y_min) / image_height,
    )


def box_iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def is_duplicate_detection(
    detections: list[tuple[int, tuple[float, float, float, float]]],
    previous_detections: list[tuple[int, tuple[float, float, float, float]]] | None,
    iou_threshold: float,
) -> bool:
    if previous_detections is None or len(detections) != len(previous_detections):
        return False

    unmatched_indices = set(range(len(previous_detections)))
    for class_id, box in detections:
        matches = [
            (box_iou(box, previous_box), index)
            for index, (previous_class_id, previous_box) in enumerate(previous_detections)
            if index in unmatched_indices and class_id == previous_class_id
        ]
        if not matches:
            return False
        best_iou, best_index = max(matches)
        if best_iou < iou_threshold:
            return False
        unmatched_indices.remove(best_index)
    return True


def save_detection(
    camera: CameraState,
    frame,
    result,
    output_dir: Path,
    confidence_threshold: float,
    duplicate_iou_threshold: float,
) -> bool:
    image_height, image_width = frame.shape[:2]
    label_rows = []
    box_indices = []
    for box_index, box in enumerate(result.boxes or []):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        if class_id not in EXPECTED_CLASSES or confidence < confidence_threshold:
            continue
        values = xyxy_to_yolo(box.xyxy[0].tolist(), image_width, image_height)
        label_rows.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in values))
        box_indices.append(box_index)

    if not label_rows:
        return False

    detections = [
        (
            int(result.boxes[index].cls[0]),
            tuple(float(value) for value in result.boxes[index].xyxy[0]),
        )
        for index in box_indices
    ]
    if duplicate_iou_threshold and is_duplicate_detection(
        detections, camera.last_saved_detections, duplicate_iou_threshold
    ):
        print(f"[{camera.name}] skipped duplicate detection")
        return True

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    filename = f"{camera.name}_{timestamp}.jpg"
    images_dir = output_dir / "images" / camera.name
    labels_dir = output_dir / "labels" / camera.name
    previews_dir = output_dir / "previews" / camera.name
    for directory in (images_dir, labels_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / filename
    label_path = labels_dir / f"{Path(filename).stem}.txt"
    preview_path = previews_dir / filename
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Could not save image: {image_path}")
    label_path.write_text("\n".join(label_rows) + "\n", encoding="utf-8")
    result.boxes = result.boxes[box_indices]
    if not cv2.imwrite(str(preview_path), result.plot()):
        raise RuntimeError(f"Could not save preview: {preview_path}")

    camera.saved_count += 1
    camera.last_saved_detections = detections
    labels = ", ".join(
        f"{EXPECTED_CLASSES[int(box.cls[0])]}={float(box.conf[0]):.2f}" for box in result.boxes
    )
    print(f"[{camera.name}] saved {image_path.name}: {labels}")
    return True


def save_raw_sample(camera: CameraState, frame, output_dir: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    samples_dir = output_dir / "background_samples" / camera.name
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_path = samples_dir / f"{camera.name}_{timestamp}.jpg"
    if not cv2.imwrite(str(sample_path), frame):
        raise RuntimeError(f"Could not save raw frame sample: {sample_path}")
    camera.sample_count += 1
    print(f"[{camera.name}] sampled {sample_path.name}")


def collect(args) -> None:
    if not IDENTITY_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Identity detection model not found: {IDENTITY_MODEL_PATH}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if args.inference_interval <= 0:
        raise ValueError("--inference-interval must be greater than 0")
    if args.background_sample_interval < 0:
        raise ValueError("--background-sample-interval must be 0 or greater")
    if args.max_background_samples_per_camera < 0:
        raise ValueError("--max-background-samples-per-camera must be 0 or greater")
    if not 0.0 <= args.duplicate_iou_threshold <= 1.0:
        raise ValueError("--duplicate-iou-threshold must be between 0 and 1")

    cameras = select_cameras(args.camera_names)
    model = YOLO(str(IDENTITY_MODEL_PATH))
    if model.names != EXPECTED_CLASSES:
        raise RuntimeError(f"Model classes must be {EXPECTED_CLASSES}, but found {model.names}")

    print("Press Ctrl+C to stop collection.")
    try:
        while True:
            active_cameras = 0
            for camera in cameras:
                current_time = time.monotonic()
                if camera.capture is None:
                    if (
                        current_time - camera.last_connect_attempt_at
                        >= COLLECTION_CONFIG["reconnect_interval_seconds"]
                    ):
                        connect(camera)
                    continue

                ok, frame = camera.capture.read()
                if not ok:
                    print(f"[{camera.name}] lost stream; reconnecting")
                    camera.capture.release()
                    camera.capture = None
                    continue
                active_cameras += 1

                background_sample_due = (
                    args.background_sample_interval
                    and (
                        not args.max_background_samples_per_camera
                        or camera.sample_count < args.max_background_samples_per_camera
                    )
                    and current_time - camera.last_sample_at >= args.background_sample_interval
                )
                inference_due = current_time - camera.last_inference_at >= args.inference_interval
                if not inference_due and not background_sample_due:
                    continue
                camera.last_inference_at = current_time
                result = model.predict(
                    frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False
                )[0]
                detected_identity = save_detection(
                    camera,
                    frame,
                    result,
                    args.output_dir,
                    args.conf,
                    args.duplicate_iou_threshold,
                )
                # A due background-sampling window must be consumed even when an
                # identity is detected.  Otherwise it remains due and bypasses
                # the inference interval on every following frame.
                if background_sample_due:
                    camera.last_sample_at = current_time
                    if not detected_identity:
                        save_raw_sample(camera, frame, args.output_dir)

                if args.max_images_per_camera and all(
                    candidate.saved_count >= args.max_images_per_camera for candidate in cameras
                ):
                    return

            if not active_cameras:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Collection stopped.")
    finally:
        for camera in cameras:
            if camera.capture is not None:
                camera.capture.release()


def main(argv=None):
    collect(parse_args(argv))


if __name__ == "__main__":
    main()
