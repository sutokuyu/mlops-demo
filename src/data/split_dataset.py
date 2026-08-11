from pathlib import Path
import random
import shutil
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

from src.config_loader import load_config, resolve_config_path

# =========================
# Configuration
# =========================

CONFIG = load_config(PROJECT_ROOT / "configs" / "config.yaml")

# Read from collected pictures and write split dataset to project `dataset/processed/`
SOURCE_DIR = resolve_config_path(CONFIG["paths"]["pictures"])
OUTPUT_DIR = resolve_config_path(CONFIG["paths"]["processed"])

VAL_RATIO = CONFIG["split"]["val_ratio"]
RANDOM_SEED = CONFIG["split"]["random_seed"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# =========================
# Split
# =========================

def split_processed_to_dataset(source_dir: Path = SOURCE_DIR,
                               output_dir: Path = OUTPUT_DIR,
                               val_ratio: float = VAL_RATIO,
                               random_seed: int = RANDOM_SEED):
    """Split images under `source_dir/<class>/` into
    `output_dir/train/<class>/` and `output_dir/val/<class>/`.
    """
    random.seed(random_seed)

    if not source_dir.is_dir():
        raise RuntimeError(f"Source directory not found: {source_dir}")

    # clear output dir
    shutil.rmtree(output_dir, ignore_errors=True)

    classes = sorted([p.name for p in source_dir.iterdir() if p.is_dir()])

    for class_name in classes:
        source_class_dir = source_dir / class_name

        images = [
            p for p in source_class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        random.shuffle(images)

        val_count = int(len(images) * val_ratio)
        val_images = images[:val_count]
        train_images = images[val_count:]

        train_dir = output_dir / "train" / class_name
        val_dir = output_dir / "val" / class_name

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nClass: {class_name}")
        print(f"  Total : {len(images)}")
        print(f"  Train : {len(train_images)}")
        print(f"  Val   : {len(val_images)}")

        for image in train_images:
            shutil.copy2(image, train_dir / image.name)

        for image in val_images:
            shutil.copy2(image, val_dir / image.name)

    print(f"\nDataset split completed! Output: {output_dir}")


if __name__ == "__main__":
    split_processed_to_dataset()