"""Shared fixtures.

Nothing here reaches the network, a GPU, a camera, or the real dataset. Where a
model is needed, a stub standing in for its ``predict`` method is used instead
of building MobileNetV2, which would download pretrained weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facemask.config import CLASS_NAMES


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Create a tiny two-class image dataset and return its root.

    Args:
        tmp_path: Pytest's per-test temporary directory.

    Returns:
        A directory containing ``train/with_mask`` and ``train/without_mask``.
    """
    from PIL import Image

    rng = np.random.default_rng(0)
    root = tmp_path / "data"
    for label in CLASS_NAMES:
        directory = root / "train" / label
        directory.mkdir(parents=True)
        for index in range(8):
            pixels = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(directory / f"{index:02d}.png")
    return root


class StubModel:
    """Stands in for a trained network, returning a fixed prediction."""

    def __init__(self, probabilities: tuple[float, float] = (0.9, 0.1)) -> None:
        self.probabilities = probabilities
        self.calls: list[tuple[int, ...]] = []

    def predict(self, batch: np.ndarray, **_: object) -> np.ndarray:
        """Record the batch shape and return the configured probabilities."""
        self.calls.append(tuple(batch.shape))
        return np.array([self.probabilities], dtype="float32")


@pytest.fixture
def stub_model() -> StubModel:
    """A model stub that confidently predicts the first class."""
    return StubModel()
