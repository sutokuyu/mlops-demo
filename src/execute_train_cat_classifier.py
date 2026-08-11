import argparse
import subprocess
import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split_dataset import split_processed_to_dataset
from src.training.train_cat_classifier import parse_args as parse_training_args
from src.training.train_cat_classifier import run_training


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split processed data and run cat classifier training"
    )
    parser.add_argument(
        "--task",
        choices=["identity", "occupancy", "all"],
        default="all",
        help="Which model(s) to train."
    )
    parser.add_argument(
        "--dataset",
        default=str(Path.cwd() / "dataset" / "processed"),
        help="Dataset output directory (defaults to ./dataset/processed)."
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio."
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to the training script."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Splitting processed data into dataset/ ...")
    split_processed_to_dataset(val_ratio=args.val_ratio)

    train_argv = ["--task", args.task, "--dataset", args.dataset]
    if args.extra:
        train_argv += args.extra

    training_args = parse_training_args(train_argv)
    print("Running training with imported trainer...")
    run_training(training_args)


if __name__ == "__main__":
    main()
