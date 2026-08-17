"""Tests for dataset discovery, label encoding, and splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facemask.config import CLASS_NAMES, TrainConfig
from facemask.data import encode_labels, find_images, load_dataset, prepare_training_data


def test_find_images_is_recursive_and_sorted(dataset: Path) -> None:
    """Images are found beneath nested directories in a deterministic order."""
    found = find_images(dataset / "train")
    assert len(found) == 16
    assert found == sorted(found)


def test_find_images_ignores_non_image_files(dataset: Path) -> None:
    """Files that are not images are skipped."""
    (dataset / "train" / "notes.txt").write_text("ignore me", encoding="utf-8")
    (dataset / "train" / "data.csv").write_text("a,b", encoding="utf-8")
    assert len(find_images(dataset / "train")) == 16


def test_find_images_matches_extensions_case_insensitively(tmp_path: Path) -> None:
    """An uppercase extension is still recognised as an image."""
    (tmp_path / "A.PNG").write_bytes(b"")
    (tmp_path / "b.JPEG").write_bytes(b"")
    assert len(find_images(tmp_path)) == 2


def test_find_images_on_an_empty_directory(tmp_path: Path) -> None:
    """An empty directory yields no images rather than raising."""
    assert find_images(tmp_path) == []


def test_encode_labels_puts_with_mask_in_column_zero() -> None:
    """Encoding order must match what the trained models expect."""
    encoded = encode_labels(["with_mask", "without_mask"])
    assert encoded.shape == (2, 2)
    np.testing.assert_array_equal(encoded[0], [1.0, 0.0])
    np.testing.assert_array_equal(encoded[1], [0.0, 1.0])


def test_encode_labels_is_one_hot() -> None:
    """Every row has exactly one active class."""
    encoded = encode_labels(list(CLASS_NAMES) * 5)
    np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(10))


def test_encode_labels_rejects_an_unknown_class() -> None:
    """A third class directory fails loudly instead of silently mis-encoding."""
    with pytest.raises(ValueError, match="unexpected class directories"):
        encode_labels(["with_mask", "surgical_mask"])


def test_load_dataset_preprocesses_for_mobilenet(dataset: Path) -> None:
    """Images are resized to 224x224 and scaled into MobileNetV2's range."""
    images, labels = load_dataset(TrainConfig(name="x", dataroot=dataset))

    assert images.shape == (16, 224, 224, 3)
    assert images.dtype == np.float32
    assert images.min() >= -1.0 and images.max() <= 1.0
    assert labels.shape == (16, 2)
    np.testing.assert_array_equal(labels.sum(axis=0), [8.0, 8.0])


def test_load_dataset_reports_a_missing_dataroot(tmp_path: Path) -> None:
    """A missing dataset directory names the path and how to fix it."""
    config = TrainConfig(name="x", dataroot=tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="training directory not found"):
        load_dataset(config)


def test_load_dataset_reports_an_empty_dataroot(tmp_path: Path) -> None:
    """A train directory with no images raises rather than returning nothing."""
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="no images found"):
        load_dataset(TrainConfig(name="x", dataroot=tmp_path))


def test_prepare_training_data_splits_and_batches(dataset: Path) -> None:
    """The split honours test_size and yields correctly shaped batches."""
    config = TrainConfig(name="x", dataroot=dataset, batch_size=2, test_size=0.25)
    images, labels = load_dataset(config)

    data = prepare_training_data(config, images, labels)

    assert len(data.train_images) == 12
    assert len(data.test_images) == 4
    assert data.steps_per_epoch == 6
    assert data.validation_steps == 2

    batch_images, batch_labels = next(iter(data.generator))
    assert batch_images.shape == (2, 224, 224, 3)
    assert batch_labels.shape == (2, 2)


def test_prepare_training_data_rejects_an_oversized_batch(dataset: Path) -> None:
    """A batch larger than the dataset explains the problem."""
    config = TrainConfig(name="x", dataroot=dataset, batch_size=500)
    images, labels = load_dataset(config)

    with pytest.raises(ValueError, match="too large for this dataset"):
        prepare_training_data(config, images, labels)


def test_split_is_reproducible(dataset: Path) -> None:
    """The same settings produce the same split, via the fixed random state."""
    config = TrainConfig(name="x", dataroot=dataset, batch_size=2)
    images, labels = load_dataset(config)

    first = prepare_training_data(config, images, labels)
    second = prepare_training_data(config, images, labels)

    np.testing.assert_array_equal(first.test_images, second.test_images)
