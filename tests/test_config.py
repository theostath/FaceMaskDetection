"""Tests for the typed configuration objects."""

from __future__ import annotations

from pathlib import Path

import pytest

from facemask.config import CLASS_NAMES, IMAGE_SIZE, DetectConfig, TrainConfig


def test_class_names_order_matches_label_encoding() -> None:
    """with_mask must occupy column 0, matching what training produced."""
    assert CLASS_NAMES == ("with_mask", "without_mask")


def test_image_size_matches_mobilenet_input() -> None:
    """MobileNetV2 is configured for 224x224 inputs."""
    assert IMAGE_SIZE == 224


def test_paths_are_coerced_from_strings() -> None:
    """String paths are accepted and normalised to Path."""
    config = TrainConfig(name="x", dataroot="some/dir", checkpoints_dir="ckpt")
    assert isinstance(config.dataroot, Path)
    assert isinstance(config.checkpoints_dir, Path)


def test_model_filename_matches_the_legacy_scheme() -> None:
    """Checkpoints keep the names the previous implementation wrote."""
    config = TrainConfig(name="MaskDetect", n_epochs=20, checkpoints_dir="checkpoints")
    assert config.experiment_dir == Path("checkpoints/MaskDetect")
    assert config.model_path.name == "model-20_MaskDetect.h5"
    assert config.figure_path.name == "accuracy_figure_MaskDetect.png"
    assert config.train_log_path.name == "train_logs_20_MaskDetect.txt"
    assert config.summary_path.name == "model_summary.txt"


def test_settings_filename_reflects_the_phase() -> None:
    """Training and detection write their settings to different files."""
    assert TrainConfig(name="x").settings_path.name == "train_opt.txt"
    assert DetectConfig(name="x").settings_path.name == "test_opt.txt"


def test_suffix_is_applied_exactly_once() -> None:
    """The suffix must not be re-applied, which previously doubled the name."""
    config = TrainConfig(name="MaskDetect", suffix="{n_epochs}ep", n_epochs=20)
    assert config.name == "MaskDetect_20ep"


def test_train_and_detect_agree_on_the_model_path_with_a_suffix() -> None:
    """The save and load sides must resolve to the same checkpoint.

    Previously the suffix was applied during option parsing and again when
    loading, so detection looked for a file training never wrote.
    """
    shared = {"name": "MaskDetect", "suffix": "{n_epochs}ep", "n_epochs": 20}
    assert TrainConfig(**shared).model_path == DetectConfig(**shared).model_path
    assert TrainConfig(**shared).model_path.name == "model-20_MaskDetect_20ep.h5"


def test_suffix_referencing_an_unknown_field_is_rejected() -> None:
    """A bad template names the offending field and the available ones."""
    with pytest.raises(ValueError, match="unknown field"):
        TrainConfig(name="x", suffix="{nope}")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_epochs": 0}, "n_epochs must be at least 1"),
        ({"batch_size": 0}, "batch_size must be at least 1"),
        ({"name": ""}, "name must not be empty"),
        ({"lr": 0}, "lr must be positive"),
        ({"beta1": 1.0}, r"beta1 must be in \[0, 1\)"),
        ({"beta2": -0.1}, r"beta2 must be in \[0, 1\)"),
        ({"val_size": 0}, r"val_size must be in \(0, 1\)"),
        ({"val_size": 1}, r"val_size must be in \(0, 1\)"),
    ],
)
def test_invalid_training_settings_are_rejected(kwargs: dict, message: str) -> None:
    """Out-of-range settings raise with an actionable message."""
    base = {"name": "x"}
    with pytest.raises(ValueError, match=message):
        TrainConfig(**{**base, **kwargs})


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_invalid_confidence_is_rejected(confidence: float) -> None:
    """Confidence outside [0, 1] raises."""
    with pytest.raises(ValueError, match=r"confidence must be in \[0, 1\]"):
        DetectConfig(name="x", confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_confidence_bounds_are_inclusive(confidence: float) -> None:
    """Both endpoints of the confidence range are accepted."""
    assert DetectConfig(name="x", confidence=confidence).confidence == confidence


def test_describe_marks_values_that_differ_from_the_default() -> None:
    """The settings table annotates overridden values."""
    described = TrainConfig(name="x", batch_size=8).describe()
    assert "[default: 32]" in described
    assert "batch_size" in described
    # A value left at its default carries no annotation.
    beta_line = next(line for line in described.splitlines() if "beta1" in line)
    assert "[default:" not in beta_line


def test_save_writes_the_settings_and_creates_the_directory(tmp_path: Path) -> None:
    """Saving settings creates the experiment directory if absent."""
    config = TrainConfig(name="Run", checkpoints_dir=tmp_path / "ckpt")
    assert not config.experiment_dir.exists()

    config.save()

    assert config.settings_path.is_file()
    assert "Run" in config.settings_path.read_text(encoding="utf-8")


def test_split_directories_are_derived_from_dataroot() -> None:
    """Both splits hang off dataroot."""
    config = TrainConfig(name="x", dataroot="data")
    assert config.train_dir == Path("data/train")
    assert config.test_dir == Path("data/test")


def test_has_test_split_detects_a_held_out_set(tmp_path: Path) -> None:
    """A held-out test set is recognised only when the directory exists."""
    config = TrainConfig(name="x", dataroot=tmp_path)
    assert config.has_test_split is False

    config.test_dir.mkdir()
    assert config.has_test_split is True


def test_evaluation_report_path() -> None:
    """The classification reports are saved beside the other artifacts."""
    config = TrainConfig(name="MaskDetect")
    assert config.evaluation_path.name == "evaluation_MaskDetect.txt"
    assert config.evaluation_path.parent == config.experiment_dir
