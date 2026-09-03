import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config, resolve_config_path

CONFIG = load_config(PROJECT_ROOT / "configs" / "config.yaml")
DEFAULT_IMAGES = PROJECT_ROOT / "dataset" / "pictures" / "cats_all"
DEFAULT_LABELS = PROJECT_ROOT / "dataset" / "labels" / "cats_all"
DEFAULT_PREVIEWS = PROJECT_ROOT / "dataset" / "labels" / "cats_all_previews"
DEFAULT_MODEL = Path(CONFIG["models"]["detection_model"])
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate YOLO pre-labels for cat images.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES,
        help=f"Directory containing images (default: {DEFAULT_IMAGES}).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help=f"Directory for YOLO txt labels (default: {DEFAULT_LABELS}).",
    )
    parser.add_argument(
        "--previews",
        type=Path,
        default=DEFAULT_PREVIEWS,
        help=f"Directory for annotated preview images (default: {DEFAULT_PREVIEWS}).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"YOLO detection weights (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Minimum detection confidence (default: 0.20).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 0 or cpu (default: Ultralytics choice).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing label files, including manually edited labels.",
    )
    parser.add_argument(
        "--refresh-previews",
        action="store_true",
        help="Re-run detection and refresh previews without replacing existing labels.",
    )
    return parser.parse_args(argv)


def xyxy_to_yolo(box, image_width: int, image_height: int) -> tuple[float, ...]:
    x_min, y_min, x_max, y_max = (float(value) for value in box)
    center_x = ((x_min + x_max) / 2) / image_width
    center_y = ((y_min + y_max) / 2) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return center_x, center_y, width, height


def resolve_model_reference(model_path: Path) -> str:
    if model_path.is_file():
        return str(model_path)

    project_model_path = resolve_config_path(model_path)
    if project_model_path.is_file():
        return str(project_model_path)

    if model_path.is_absolute() or len(model_path.parts) > 1:
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    return str(model_path)


def generate_labels(args) -> tuple[int, int, int]:
    if not args.images.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.images}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")

    image_paths = sorted(
        path
        for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No supported images found in {args.images}")

    args.labels.mkdir(parents=True, exist_ok=True)
    args.previews.mkdir(parents=True, exist_ok=True)
    model = YOLO(resolve_model_reference(args.model))
    processed = skipped = detected = 0

    for image_path in image_paths:
        label_path = args.labels / f"{image_path.stem}.txt"
        if label_path.exists() and not args.overwrite and not args.refresh_previews:
            skipped += 1
            continue

        predict_args = {
            "source": str(image_path),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "verbose": False,
        }
        if args.device is not None:
            predict_args["device"] = args.device
        result = model.predict(**predict_args)[0]
        image_height, image_width = result.orig_shape
        rows = []
        cat_box_indices = []

        if result.boxes is not None:
            for box_index, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = str(model.names[class_id]).lower()
                if class_name != "cat":
                    continue
                cat_box_indices.append(box_index)
                values = xyxy_to_yolo(box.xyxy[0].tolist(), image_width, image_height)
                rows.append("0 " + " ".join(f"{value:.6f}" for value in values))

        if args.overwrite or not label_path.exists():
            label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        if result.boxes is not None:
            result.boxes = result.boxes[cat_box_indices]
        preview = result.plot()
        preview_path = args.previews / image_path.name
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"Could not save preview image: {preview_path}")
        processed += 1
        if rows:
            detected += 1

    return processed, skipped, detected


def main(argv=None):
    args = parse_args(argv)
    processed, skipped, detected = generate_labels(args)
    print(f"Processed: {processed}")
    print(f"Skipped existing labels: {skipped}")
    print(f"Images with cat detections: {detected}")
    print(f"Labels saved to: {args.labels}")
    print(f"Preview images saved to: {args.previews}")
    print("Review every box and change class 0 to bagel or 1 to kurumi before training.")


if __name__ == "__main__":
    main()
