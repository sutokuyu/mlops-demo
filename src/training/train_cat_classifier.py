import argparse
import os
import shutil
import sys
from pathlib import Path

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


CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
config = load_config(CONFIG_PATH)

IDENTITY_CLASSES = config["cats"]["identity_classes"]
EMPTY_CLASS = config["cats"]["occupancy_empty_class"]
OCCUPANCY_CLASSES = config["cats"]["occupancy_classes"]

MODEL_WEIGHTS = config["models"]["classification_model"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the identity and occupancy classifiers using YOLO."
    )

    parser.add_argument(
        "--task",
        choices=["identity", "occupancy", "all"],
        default="all",
        help="Which model(s) to train."
    )

    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "dataset" / "processed"),
        help="Dataset root directory containing train/ and val/."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=config["training"]["epochs"],
        help="Number of training epochs."
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=config["training"]["imgsz"],
        help="Input image size."
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=config["training"]["batch"],
        help="Batch size."
    )

    parser.add_argument(
        "--device",
        default=config["training"]["device"],
        help="Device to use for training, e.g. 0 or cpu."
    )

    parser.add_argument(
        "--project",
        default=str(resolve_config_path(config["training"]["project_dir"])),
        help="Output project directory."
    )

    parser.add_argument(
        "--exist_ok",
        action="store_true",
        default=config["training"]["exist_ok"],
        help="Overwrite existing project directory if it exists."
    )

    return parser.parse_args(argv)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def make_symlink_or_copy(src: Path, dst: Path):
    if dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)

    try:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def copy_directory(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def get_split_dirs(dataset_root: Path):
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    if not train_dir.is_dir() or not val_dir.is_dir():
        raise RuntimeError(
            f"Dataset root must contain train/ and val/ directories: {dataset_root}"
        )

    return train_dir, val_dir


def get_class_names(directory: Path):
    return sorted(
        [p.name for p in directory.iterdir() if p.is_dir()]
    )


def validate_dataset(train_dir: Path, val_dir: Path):
    train_classes = set(get_class_names(train_dir))
    val_classes = set(get_class_names(val_dir))

    if EMPTY_CLASS not in train_classes or EMPTY_CLASS not in val_classes:
        raise RuntimeError(
            f"Dataset must contain '{EMPTY_CLASS}' in both train/ and val/."
        )

    non_empty_train = train_classes - {EMPTY_CLASS}
    non_empty_val = val_classes - {EMPTY_CLASS}

    if not non_empty_train or not non_empty_val:
        raise RuntimeError(
            "Dataset must contain at least one non-empty class besides 'empty'."
        )

    missing_identities = set(IDENTITY_CLASSES) - non_empty_train
    if missing_identities:
        raise RuntimeError(
            f"Identity dataset is missing folders: {sorted(missing_identities)} in train/."
        )

    missing_identities = set(IDENTITY_CLASSES) - non_empty_val
    if missing_identities:
        raise RuntimeError(
            f"Identity dataset is missing folders: {sorted(missing_identities)} in val/."
        )

    return train_classes, val_classes


def build_identity_dataset(train_dir: Path, val_dir: Path, tmp_root: Path):
    print(f"Building identity dataset in {tmp_root}")
    identity_root = tmp_root / "identity"
    shutil.rmtree(identity_root, ignore_errors=True)

    for split_dir in [train_dir, val_dir]:
        split_name = split_dir.name
        dest_split = identity_root / split_name
        ensure_dir(dest_split)

        for class_name in IDENTITY_CLASSES:
            src_class_dir = split_dir / class_name
            if not src_class_dir.is_dir():
                raise RuntimeError(
                    f"Missing class folder for identity training: {src_class_dir}"
                )
            make_symlink_or_copy(src_class_dir, dest_split / class_name)

    return identity_root


def build_occupancy_dataset(train_dir: Path, val_dir: Path, tmp_root: Path):
    print(f"Building occupancy dataset in {tmp_root}")
    occupancy_root = tmp_root / "occupancy"
    shutil.rmtree(occupancy_root, ignore_errors=True)

    for split_dir in [train_dir, val_dir]:
        split_name = split_dir.name
        dest_split = occupancy_root / split_name
        ensure_dir(dest_split)

        empty_src = split_dir / EMPTY_CLASS
        if not empty_src.is_dir():
            raise RuntimeError(
                f"Missing empty class folder: {empty_src}"
            )

        copy_directory(empty_src, dest_split / EMPTY_CLASS)

        non_empty_dest = dest_split / "non-empty"
        ensure_dir(non_empty_dest)

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name == EMPTY_CLASS:
                continue
            for image_path in class_dir.iterdir():
                if not image_path.is_file():
                    continue
                target_name = f"{class_dir.name}_{image_path.name}"
                target_path = non_empty_dest / target_name
                if target_path.exists():
                    target_path = non_empty_dest / f"{class_dir.name}_{image_path.stem}_{image_path.suffix}"
                try:
                    target_path.symlink_to(image_path.resolve())
                except OSError:
                    shutil.copy2(image_path, target_path)

    return occupancy_root


def train_model(task: str, data_root: Path, args):
    print(f"Training {task} model using dataset: {data_root}")
    model = YOLO(MODEL_WEIGHTS)
    model.train(
        data=str(data_root),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        degrees=10,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        project=args.project,
        name=task,
        exist_ok=args.exist_ok,
        save=True,
    )
    print(f"Finished training {task}. Best model saved to {args.project}/{task}/weights/best.pt")


def run_training(args):
    dataset_root = Path(args.dataset)

    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root not found: {dataset_root}")

    train_dir, val_dir = get_split_dirs(dataset_root)
    train_classes, val_classes = validate_dataset(train_dir, val_dir)

    tmp_root = PROJECT_ROOT / "tmp_training_data"
    ensure_dir(tmp_root)

    try:
        tasks = [args.task] if args.task != "all" else ["identity", "occupancy"]

        if "identity" in tasks:
            identity_root = build_identity_dataset(train_dir, val_dir, tmp_root)
            train_model("identity", identity_root, args)

        if "occupancy" in tasks:
            occupancy_root = build_occupancy_dataset(train_dir, val_dir, tmp_root)
            train_model("occupancy", occupancy_root, args)

        print("\nAll requested training tasks are complete.")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main(argv=None):
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
