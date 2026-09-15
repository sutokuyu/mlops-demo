import argparse
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Split a YOLO detection dataset into chronological train, validation, and test sets.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/cat_identity_detection"),
        help="Dataset root containing images/ and labels/train/.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the split plan without moving any files.",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Treat images without labels as background and create empty label files.",
    )
    return parser.parse_args(argv)


def _validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Train and validation ratios must be positive and sum to less than 1.")


def _source_group(image_path: Path) -> str:
    return image_path.stem.split("_", maxsplit=1)[0]


def _split_group(images: list[Path], train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    train_end = int(len(images) * train_ratio)
    val_end = train_end + int(len(images) * val_ratio)
    return {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }


def plan_split(
    dataset_root: Path,
    train_ratio: float,
    val_ratio: float,
    allow_missing_labels: bool = False,
):
    images_dir = dataset_root / "images"
    source_images_dir = images_dir / "train" if (images_dir / "train").is_dir() else images_dir
    labels_dir = dataset_root / "labels" / "train"
    if not source_images_dir.is_dir() or not labels_dir.is_dir():
        raise RuntimeError("Dataset must contain images/ and labels/train/ directories.")

    images_by_group = defaultdict(list)
    skipped_images = []
    for image_path in sorted(source_images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if allow_missing_labels or (labels_dir / f"{image_path.stem}.txt").is_file():
            images_by_group[_source_group(image_path)].append(image_path)
        else:
            skipped_images.append(image_path)

    splits = {"train": [], "val": [], "test": []}
    for group_images in images_by_group.values():
        for split_name, images in _split_group(group_images, train_ratio, val_ratio).items():
            splits[split_name].extend(images)
    return splits, skipped_images


def print_summary(splits: dict[str, list[Path]], skipped_images: list[Path]) -> None:
    print("Split summary:")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {len(splits[split_name])} annotated images")
    print(f"  skipped: {len(skipped_images)} images without annotations")


def apply_split(
    dataset_root: Path,
    splits: dict[str, list[Path]],
    skipped_images: list[Path],
    allow_missing_labels: bool = False,
) -> None:
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    source_images_dir = images_dir / "train" if (images_dir / "train").is_dir() else images_dir
    source_labels_dir = labels_dir / "train"
    staging_labels_dir = labels_dir / "all"
    if staging_labels_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing staging directory: {staging_labels_dir}"
        )

    source_labels_dir.rename(staging_labels_dir)
    try:
        for split_name, images in splits.items():
            split_images_dir = images_dir / split_name
            split_labels_dir = labels_dir / split_name
            split_images_dir.mkdir(exist_ok=True)
            split_labels_dir.mkdir()
            for image_path in images:
                label_path = staging_labels_dir / f"{image_path.stem}.txt"
                if split_images_dir != source_images_dir:
                    shutil.move(image_path, split_images_dir / image_path.name)
                if label_path.is_file():
                    shutil.move(label_path, split_labels_dir / label_path.name)
                elif allow_missing_labels:
                    (split_labels_dir / label_path.name).touch()
                else:
                    raise RuntimeError(f"Missing label for image: {image_path}")

        skipped_dir = images_dir / "skipped"
        skipped_dir.mkdir()
        for image_path in skipped_images:
            shutil.move(image_path, skipped_dir / image_path.name)
    finally:
        if staging_labels_dir.exists() and not any(staging_labels_dir.iterdir()):
            staging_labels_dir.rmdir()


def main(argv=None):
    args = parse_args(argv)
    _validate_ratios(args.train_ratio, args.val_ratio)
    dataset_root = args.dataset
    split_directories = (
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
        dataset_root / "labels" / "val",
        dataset_root / "labels" / "test",
    )
    if all(directory.is_dir() for directory in split_directories):
        raise RuntimeError(
            f"Dataset is already split into train/val/test: {dataset_root}. "
            "Do not run the splitter again on this dataset."
        )
    splits, skipped_images = plan_split(
        args.dataset, args.train_ratio, args.val_ratio, args.allow_missing_labels
    )
    print_summary(splits, skipped_images)
    if args.dry_run:
        return
    apply_split(args.dataset, splits, skipped_images, args.allow_missing_labels)


if __name__ == "__main__":
    main()
