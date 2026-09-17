from pathlib import Path

import pytest

from cerber.experiments.prepare_seraphim import split_train_val, write_data_yaml


def _pairs(n: int, tmp_path: Path) -> list[tuple[Path, Path]]:
    items = []
    for index in range(n):
        image = tmp_path / f"{index}.jpg"
        label = tmp_path / f"{index}.txt"
        items.append((image, label))
    return items


def test_write_seraphim_data_yaml(tmp_path: Path) -> None:
    yaml_path = write_data_yaml(tmp_path)
    text = yaml_path.read_text(encoding="utf-8")
    assert "0: drone" in text
    assert "train: images/train" in text
    assert "val: images/val" in text


def test_split_rejects_full_val_fraction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        split_train_val(_pairs(10, tmp_path), 1.0)


def test_split_does_not_leak_when_fraction_would_empty_train(tmp_path: Path) -> None:
    pairs = _pairs(10, tmp_path)
    train_pairs, val_pairs = split_train_val(pairs, 0.99)
    train_ids = {path.name for path, _ in train_pairs}
    val_ids = {path.name for path, _ in val_pairs}
    assert train_pairs
    assert val_pairs
    assert train_ids.isdisjoint(val_ids)
    assert len(train_pairs) + len(val_pairs) == 10


def test_split_default_fraction(tmp_path: Path) -> None:
    train_pairs, val_pairs = split_train_val(_pairs(10, tmp_path), 0.1)
    assert len(val_pairs) == 1
    assert len(train_pairs) == 9
