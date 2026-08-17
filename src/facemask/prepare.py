"""Split a flat two-class image directory into train and test sets.

The datasets linked from the README ship as a single directory per class::

    dataset/
    ├── with_mask/
    └── without_mask/

Training expects a held-out test set alongside the training images::

    face-mask-dataset/
    ├── train/
    │   ├── with_mask/
    │   └── without_mask/
    └── test/
        ├── with_mask/
        └── without_mask/

This module performs that rearrangement. The split is stratified, so both
classes keep their proportions, and seeded, so the same source directory always
produces the same split.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from facemask.config import CLASS_NAMES
from facemask.images import find_images

#: Seed for the shuffle, so a given source directory always splits identically.
SPLIT_SEED = 42


def split_class_files(
    paths: list[Path], test_fraction: float, seed: int = SPLIT_SEED
) -> tuple[list[Path], list[Path]]:
    """Divide one class's files into training and test portions.

    Args:
        paths: Files belonging to a single class.
        test_fraction: Share of the files to place in the test set.
        seed: Shuffle seed.

    Returns:
        A pair of ``(train_paths, test_paths)``.
    """
    import random

    shuffled = sorted(paths)
    random.Random(seed).shuffle(shuffled)

    test_count = round(len(shuffled) * test_fraction)
    # Never let either side end up empty when there is data to go around.
    test_count = max(1, min(test_count, len(shuffled) - 1)) if len(shuffled) > 1 else 0
    return shuffled[test_count:], shuffled[:test_count]


def prepare_dataset(
    source: Path,
    destination: Path,
    test_fraction: float = 0.2,
    seed: int = SPLIT_SEED,
    move: bool = False,
) -> dict[str, dict[str, int]]:
    """Arrange a flat two-class directory into train and test splits.

    Args:
        source: Directory containing one folder per class.
        destination: Directory to create ``train/`` and ``test/`` beneath.
        test_fraction: Share of each class held out for testing.
        seed: Shuffle seed.
        move: Move files instead of copying them.

    Returns:
        Counts per split and class, e.g. ``{"train": {"with_mask": 1732}}``.

    Raises:
        FileNotFoundError: If the source or an expected class folder is absent.
        ValueError: If ``test_fraction`` is not strictly between 0 and 1.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    source = Path(source)
    destination = Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(f"source directory not found: {source}")

    missing = [name for name in CLASS_NAMES if not (source / name).is_dir()]
    if missing:
        expected = ", ".join(CLASS_NAMES)
        raise FileNotFoundError(
            f"{source} is missing class directories {missing}; expected: {expected}"
        )

    transfer = shutil.move if move else shutil.copy2
    counts: dict[str, dict[str, int]] = {"train": {}, "test": {}}

    for class_name in CLASS_NAMES:
        paths = find_images(source / class_name)
        if not paths:
            raise FileNotFoundError(f"no images found in {source / class_name}")

        train_paths, test_paths = split_class_files(paths, test_fraction, seed)

        for split, split_paths in (("train", train_paths), ("test", test_paths)):
            target = destination / split / class_name
            target.mkdir(parents=True, exist_ok=True)
            for path in split_paths:
                transfer(str(path), str(target / path.name))
            counts[split][class_name] = len(split_paths)

        print(
            f"(Info) {class_name}: {len(train_paths)} train, {len(test_paths)} test"
        )

    return counts
