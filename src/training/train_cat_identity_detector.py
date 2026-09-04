import argparse
import sys
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


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
TRAINING_CONFIG = CONFIG["training"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO detector to identify Bagel and Kurumi in full camera frames.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=resolve_config_path(TRAINING_CONFIG["identity_detection_dataset"]),
        help="YOLO detection dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        default=CONFIG["models"]["detection_model"],
        help="Pretrained YOLO detection weights or model name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_CONFIG["identity_detection_epochs"],
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=TRAINING_CONFIG["identity_detection_imgsz"],
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=TRAINING_CONFIG["identity_detection_batch"],
        help="Batch size; use -1 to let Ultralytics select a GPU-safe batch size.",
    )
    parser.add_argument("--device", default=TRAINING_CONFIG["device"])
    parser.add_argument(
        "--project",
        type=Path,
        default=resolve_config_path(TRAINING_CONFIG["identity_detection_project_dir"]),
    )
    parser.add_argument(
        "--name",
        default=TRAINING_CONFIG["identity_detection_run_name"],
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args(argv)


def validate_data(data_path: Path) -> None:
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    dataset = check_det_dataset(str(data_path))
    expected_names = {0: "bagel", 1: "kurumi"}
    if dataset["names"] != expected_names:
        raise RuntimeError(
            f"Dataset classes must be {expected_names}, but found {dataset['names']}."
        )


def train(args) -> None:
    validate_data(args.data)
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=True,
        resume=args.resume,
        save=True,
    )


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
