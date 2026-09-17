from pathlib import Path

import pytest

from cerber.config import load_runtime_config, project_root
from cerber.experiments.config import load_experiment_config


def test_project_root_contains_cerber() -> None:
    assert (project_root() / "cerber" / "config.py").is_file()


def test_load_runtime_config(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        "\n".join(
            [
                "weights: yolo26n-seg.pt",
                "source: 0",
                "task: segment",
                "device: cpu",
                "tracker: bytetrack.yaml",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config = load_runtime_config(path, source="1", show=True)
    assert config.weights == "yolo26n-seg.pt"
    assert config.source == 1
    assert config.device == "cpu"
    assert config.task == "segment"
    assert config.show is True
    assert config.persist is True


def test_runtime_yaml_files_exist() -> None:
    root = project_root()
    for name in ("runtime.yaml", "runtime-visdrone.yaml", "runtime-drone.yaml"):
        assert (root / "configs" / name).is_file()


def test_load_experiment_config() -> None:
    config = load_experiment_config("configs/experiments/visdrone-n.yaml")
    assert config.model == "yolo26n.pt"
    assert config.data == "VisDrone.yaml"
    assert config.name == "visdrone-n"
    kwargs = config.train_kwargs()
    assert kwargs["imgsz"] == 640
    assert kwargs["optimizer"] == "MuSGD"


def test_rejects_bad_task(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("weights: x.pt\ntask: pose\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported task"):
        load_runtime_config(path)
