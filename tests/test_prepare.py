"""Tests for splitting a flat two-class directory into train and test sets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facemask.config import CLASS_NAMES
from facemask.prepare import prepare_dataset, split_class_files


@pytest.fixture
def flat_dataset(tmp_path: Path) -> Path:
    """A flat directory holding one folder per class, as the sources ship."""
    from PIL import Image

    rng = np.random.default_rng(0)
    root = tmp_path / "flat"
    for label, count in zip(CLASS_NAMES, (20, 10), strict=True):
        directory = root / label
        directory.mkdir(parents=True)
        for index in range(count):
            pixels = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(directory / f"{index:03d}.png")
    return root


def test_split_class_files_honours_the_fraction() -> None:
    """The test portion is the requested share of the input."""
    paths = [Path(f"{index}.png") for index in range(100)]
    train, test = split_class_files(paths, test_fraction=0.2)
    assert len(test) == 20
    assert len(train) == 80


def test_split_class_files_is_a_partition() -> None:
    """Every file lands in exactly one side."""
    paths = [Path(f"{index}.png") for index in range(37)]
    train, test = split_class_files(paths, test_fraction=0.3)
    assert set(train) | set(test) == set(paths)
    assert not set(train) & set(test)


def test_split_class_files_is_deterministic() -> None:
    """The same seed produces the same split."""
    paths = [Path(f"{index}.png") for index in range(50)]
    assert split_class_files(paths, 0.2, seed=1) == split_class_files(paths, 0.2, seed=1)


def test_split_class_files_varies_with_the_seed() -> None:
    """A different seed shuffles differently."""
    paths = [Path(f"{index}.png") for index in range(50)]
    assert split_class_files(paths, 0.2, seed=1) != split_class_files(paths, 0.2, seed=2)


def test_split_class_files_never_empties_either_side() -> None:
    """A tiny class still yields at least one file on each side."""
    train, test = split_class_files([Path("a.png"), Path("b.png")], test_fraction=0.01)
    assert len(train) == 1
    assert len(test) == 1


def test_prepare_dataset_creates_both_splits(flat_dataset: Path, tmp_path: Path) -> None:
    """train/ and test/ are created with one folder per class."""
    destination = tmp_path / "arranged"

    counts = prepare_dataset(flat_dataset, destination, test_fraction=0.2)

    for split in ("train", "test"):
        for label in CLASS_NAMES:
            assert (destination / split / label).is_dir()

    assert counts["train"]["with_mask"] == 16
    assert counts["test"]["with_mask"] == 4
    assert counts["train"]["without_mask"] == 8
    assert counts["test"]["without_mask"] == 2


def test_prepare_dataset_is_stratified(flat_dataset: Path, tmp_path: Path) -> None:
    """Each class is split by the same fraction, preserving its proportion."""
    destination = tmp_path / "arranged"
    counts = prepare_dataset(flat_dataset, destination, test_fraction=0.2)

    for label in CLASS_NAMES:
        total = counts["train"][label] + counts["test"][label]
        assert counts["test"][label] == pytest.approx(total * 0.2, abs=1)


def test_prepare_dataset_copies_by_default(flat_dataset: Path, tmp_path: Path) -> None:
    """The source directory is left intact unless moving is requested."""
    before = sorted(p.name for p in (flat_dataset / "with_mask").iterdir())

    prepare_dataset(flat_dataset, tmp_path / "arranged", test_fraction=0.2)

    after = sorted(p.name for p in (flat_dataset / "with_mask").iterdir())
    assert before == after


def test_prepare_dataset_can_move(flat_dataset: Path, tmp_path: Path) -> None:
    """Moving empties the source directories."""
    prepare_dataset(flat_dataset, tmp_path / "arranged", test_fraction=0.2, move=True)
    assert list((flat_dataset / "with_mask").iterdir()) == []


def test_prepare_dataset_produces_no_overlap(flat_dataset: Path, tmp_path: Path) -> None:
    """No image appears in both splits, which would leak test data into training."""
    destination = tmp_path / "arranged"
    prepare_dataset(flat_dataset, destination, test_fraction=0.2)

    for label in CLASS_NAMES:
        train = {p.name for p in (destination / "train" / label).iterdir()}
        test = {p.name for p in (destination / "test" / label).iterdir()}
        assert not train & test


def test_prepare_dataset_rejects_a_missing_source(tmp_path: Path) -> None:
    """An absent source directory is reported by path."""
    with pytest.raises(FileNotFoundError, match="source directory not found"):
        prepare_dataset(tmp_path / "absent", tmp_path / "out")


def test_prepare_dataset_rejects_missing_class_directories(tmp_path: Path) -> None:
    """A source lacking the expected class folders names what it wanted."""
    source = tmp_path / "flat"
    (source / "with_mask").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="missing class directories"):
        prepare_dataset(source, tmp_path / "out")


@pytest.mark.parametrize("fraction", [0, 1, -0.5, 1.5])
def test_prepare_dataset_rejects_an_invalid_fraction(
    flat_dataset: Path, tmp_path: Path, fraction: float
) -> None:
    """The test fraction must be strictly between 0 and 1."""
    with pytest.raises(ValueError, match="test_fraction must be in"):
        prepare_dataset(flat_dataset, tmp_path / "out", test_fraction=fraction)


def test_prepare_dataset_rejects_an_empty_class(tmp_path: Path) -> None:
    """A class folder with no images raises rather than producing an empty split."""
    source = tmp_path / "flat"
    for label in CLASS_NAMES:
        (source / label).mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no images found"):
        prepare_dataset(source, tmp_path / "out")
