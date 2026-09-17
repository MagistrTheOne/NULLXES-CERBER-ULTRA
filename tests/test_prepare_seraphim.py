from pathlib import Path

from cerber.experiments.prepare_seraphim import write_data_yaml


def test_write_seraphim_data_yaml(tmp_path: Path) -> None:
    yaml_path = write_data_yaml(tmp_path)
    text = yaml_path.read_text(encoding="utf-8")
    assert "0: drone" in text
    assert "train: images/train" in text
    assert "val: images/val" in text
