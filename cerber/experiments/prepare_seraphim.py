from __future__ import annotations

import argparse
import random
import shutil
import sys
import zipfile
from pathlib import Path

from cerber.config import project_root

REPO_ID = "lgrzybowski/seraphim-drone-detection-dataset"


def _extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)


def _pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image in sorted(images_dir.rglob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            nested = next(labels_dir.rglob(f"{image.stem}.txt"), None)
            if nested is None:
                continue
            label = nested
        pairs.append((image, label))
    return pairs


def _copy_split(pairs: list[tuple[Path, Path]], images_out: Path, labels_out: Path) -> None:
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for image, label in pairs:
        shutil.copy2(image, images_out / image.name)
        shutil.copy2(label, labels_out / label.name)


def split_train_val(
    pairs: list[tuple[Path, Path]],
    val_fraction: float,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if not pairs:
        raise RuntimeError("Seraphim subset is empty after download/extract")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if len(pairs) < 2:
        raise ValueError("need at least 2 images to split train/val without leakage")
    val_count = int(len(pairs) * val_fraction)
    val_count = min(max(val_count, 1), len(pairs) - 1)
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]
    if not train_pairs or not val_pairs:
        raise RuntimeError("train/val split produced an empty split")
    overlap = {path for path, _ in train_pairs} & {path for path, _ in val_pairs}
    if overlap:
        raise RuntimeError("train/val split leaked images")
    return train_pairs, val_pairs


def write_data_yaml(root: Path) -> Path:
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: drone",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def prepare(
    batches: int,
    val_fraction: float,
    max_images: int | None,
    seed: int,
    output_dir: Path,
) -> Path:
    from huggingface_hub import hf_hub_download

    raw_root = output_dir / "raw"
    subset_root = output_dir / "subset"
    images_raw = raw_root / "train" / "images"
    labels_raw = raw_root / "train" / "labels"
    images_raw.mkdir(parents=True, exist_ok=True)
    labels_raw.mkdir(parents=True, exist_ok=True)

    for index in range(1, batches + 1):
        batch = f"{index:03d}"
        image_zip = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=f"train/images/batch_{batch}.zip",
            )
        )
        label_zip = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=f"train/labels/batch_{batch}.zip",
            )
        )
        _extract_zip(image_zip, images_raw)
        _extract_zip(label_zip, labels_raw)

    pairs = _pairs(images_raw, labels_raw)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_images is not None:
        pairs = pairs[: max(0, max_images)]
    train_pairs, val_pairs = split_train_val(pairs, val_fraction)
    _copy_split(train_pairs, subset_root / "images" / "train", subset_root / "labels" / "train")
    _copy_split(val_pairs, subset_root / "images" / "val", subset_root / "labels" / "val")
    yaml_path = write_data_yaml(subset_root)
    print(f"train={len(train_pairs)} val={len(val_pairs)} yaml={yaml_path}")
    print("official Seraphim test split was not downloaded")
    return yaml_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a Seraphim train subset and split val from train")
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-images", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/seraphim")
    args = parser.parse_args(argv)
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_root() / output_dir
    prepare(
        batches=args.batches,
        val_fraction=args.val_fraction,
        max_images=args.max_images,
        seed=args.seed,
        output_dir=output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
