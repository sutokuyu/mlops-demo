"""Create an Ultralytics YOLO Detection archive from images and labels."""

import argparse
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create an Ultralytics YOLO Detection annotation ZIP from images and labels."
    )
    parser.add_argument(
        "--images-dir", type=Path, required=True, help="Directory containing images."
    )
    parser.add_argument(
        "--labels-dir", type=Path, required=True, help="Directory containing YOLO .txt labels."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .zip path.")
    parser.add_argument(
        "--class-name",
        action="append",
        dest="class_names",
        required=True,
        help="Class name in YOLO ID order. Repeat once per class, e.g. --class-name bagel.",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Include images without a matching label file as empty annotations.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing an existing output archive."
    )
    return parser.parse_args(argv)


def collect_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    files: dict[str, Path] = {}
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        if path.suffix.lower() not in suffixes:
            continue
        if path.stem in files:
            raise ValueError(
                f"Duplicate basename '{path.stem}' in {directory}: {files[path.stem]} and {path}"
            )
        files[path.stem] = path
    return files


def validate_label(label_path: Path, class_count: int) -> None:
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(
                f"{label_path}:{line_number} must contain class ID and four coordinates"
            )
        try:
            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
        except ValueError as error:
            raise ValueError(
                f"{label_path}:{line_number} contains non-numeric YOLO values"
            ) from error
        if not 0 <= class_id < class_count:
            raise ValueError(
                f"{label_path}:{line_number} has class ID outside 0..{class_count - 1}"
            )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(
                f"{label_path}:{line_number} coordinates must be finite values from 0 to 1"
            )


def create_archive(args) -> None:
    class_names = [name.strip() for name in args.class_names]
    if not all(class_names) or len(set(class_names)) != len(class_names):
        raise ValueError("--class-name values must be non-empty and unique")
    if args.output.suffix.lower() != ".zip":
        raise ValueError("--output must end with .zip")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --overwrite to replace it."
        )

    images = collect_files(args.images_dir, IMAGE_SUFFIXES)
    labels = collect_files(args.labels_dir, {".txt"})
    if not images:
        raise ValueError(f"No supported images found in: {args.images_dir}")

    missing_labels = sorted(set(images) - set(labels))
    orphan_labels = sorted(set(labels) - set(images))
    if orphan_labels:
        raise ValueError(f"Labels without a matching image: {', '.join(orphan_labels)}")
    if missing_labels and not args.allow_missing_labels:
        raise ValueError(
            f"Images without a matching label: {', '.join(missing_labels)}. "
            "Use --allow-missing-labels to include them as empty annotations."
        )

    for label_path in labels.values():
        validate_label(label_path, len(class_names))

    image_names = [images[stem].name for stem in sorted(images)]
    data_yaml = "path: ./\ntrain: train.txt\nnames:\n" + "".join(
        f"  {class_id}: {name}\n" for class_id, name in enumerate(class_names)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("data.yaml", data_yaml)
        archive.writestr(
            "train.txt",
            "\n".join(f"images/train/{image_name}" for image_name in image_names) + "\n",
        )
        for stem in sorted(images):
            label_archive_path = f"labels/train/{stem}.txt"
            label_path = labels.get(stem)
            if label_path is None:
                archive.writestr(label_archive_path, "")
            else:
                archive.write(label_path, label_archive_path)
            archive.write(images[stem], f"images/train/{images[stem].name}")

    print(f"Created {args.output} with {len(images)} images and {len(labels)} label files.")


def main(argv=None):
    create_archive(parse_args(argv))


if __name__ == "__main__":
    main()
