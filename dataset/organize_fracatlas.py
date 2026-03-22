from pathlib import Path
import csv
import random
import shutil
import sys


PROJECT_DATASET_DIR = Path(__file__).resolve().parent
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def ensure_output_dirs():
    for split in ("train", "val", "test"):
        for label in ("fractured", "non_fractured"):
            (PROJECT_DATASET_DIR / split / label).mkdir(parents=True, exist_ok=True)


def clear_existing_images():
    for split in ("train", "val", "test"):
        for label in ("fractured", "non_fractured"):
            target_dir = PROJECT_DATASET_DIR / split / label
            for item in target_dir.iterdir():
                if item.name == ".gitkeep":
                    continue
                if item.is_file():
                    item.unlink()


def find_images_dir(fracatlas_root):
    candidates = [
        fracatlas_root / "images",
        fracatlas_root / "Images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find FracAtlas images folder.")


def load_binary_records(fracatlas_root):
    metadata_path = fracatlas_root / "dataset.csv"
    if not metadata_path.exists():
        raise FileNotFoundError("Could not find FracAtlas dataset.csv metadata file.")

    records = {"fractured": [], "non_fractured": []}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            image_id = row.get("image_id", "").strip()
            fractured_flag = row.get("fractured", "").strip()
            if not image_id or fractured_flag not in {"0", "1"}:
                continue

            label = "fractured" if fractured_flag == "1" else "non_fractured"
            records[label].append(image_id)

    return records


def split_records(file_names):
    shuffled = list(file_names)
    random.Random(RANDOM_SEED).shuffle(shuffled)

    train_end = int(len(shuffled) * TRAIN_RATIO)
    val_end = train_end + int(len(shuffled) * VAL_RATIO)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def copy_records(images_dir, split_name, label, file_names):
    source_dir_name = "Fractured" if label == "fractured" else "Non_fractured"
    source_dir = images_dir / source_dir_name
    destination_dir = PROJECT_DATASET_DIR / split_name / label

    copied = 0
    for file_name in file_names:
        source_path = source_dir / file_name
        if not source_path.exists():
            continue
        shutil.copy2(source_path, destination_dir / file_name)
        copied += 1
    return copied


def organize_fracatlas(fracatlas_root):
    fracatlas_root = Path(fracatlas_root)
    ensure_output_dirs()
    clear_existing_images()

    images_dir = find_images_dir(fracatlas_root)
    records = load_binary_records(fracatlas_root)

    summary = {}
    for label, file_names in records.items():
        split_sets = split_records(file_names)
        for split_name, split_files in split_sets.items():
            summary.setdefault(split_name, {"fractured": 0, "non_fractured": 0})
            copied = copy_records(images_dir, split_name, label, split_files)
            summary[split_name][label] = copied

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python organize_fracatlas.py <path_to_extracted_fracatlas>")
        sys.exit(1)

    source_root = Path(sys.argv[1])
    results = organize_fracatlas(source_root)

    print("FracAtlas dataset organized successfully.")
    for split_name, counts in results.items():
        fractured_count = counts["fractured"]
        normal_count = counts["non_fractured"]
        total_count = fractured_count + normal_count
        print(
            f"{split_name}: total={total_count}, "
            f"fractured={fractured_count}, non_fractured={normal_count}"
        )
