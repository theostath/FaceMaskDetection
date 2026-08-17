"""Tests for face detection and frame annotation.

The camera loop itself is not covered; these exercise the pieces around it
using stub models and stub cascades so no hardware is required.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from facemask.config import CLASS_NAMES, IMAGE_SIZE, DetectConfig
from facemask.detect import (
    DETECTION_SCALE,
    annotate_frame,
    classify_face,
    resolve_cascade,
    run_detection,
)


class StubCascade:
    """Returns a fixed set of boxes, in downscaled coordinates."""

    def __init__(self, boxes: list[tuple[int, int, int, int]]) -> None:
        self.boxes = boxes

    def detectMultiScale(self, image: np.ndarray) -> list:
        """Return the configured boxes regardless of the image."""
        return self.boxes


def test_resolve_cascade_finds_the_bundled_file() -> None:
    """The cascade is located inside the installed opencv-python package."""
    classifier = resolve_cascade()
    assert not classifier.empty()


def test_resolve_cascade_does_not_use_a_hardcoded_path() -> None:
    """The cascade resolves relative to the installed OpenCV, wherever that is."""
    bundled = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    assert bundled.is_file()


def test_resolve_cascade_reports_a_missing_file(tmp_path: Path) -> None:
    """A missing cascade names the path it looked for."""
    with pytest.raises(FileNotFoundError, match="Haar cascade not found"):
        resolve_cascade(tmp_path / "absent.xml")


def test_resolve_cascade_reports_an_unparseable_file(tmp_path: Path) -> None:
    """A file that is not a cascade raises rather than returning empty."""
    junk = tmp_path / "junk.xml"
    junk.write_text("<not-a-cascade/>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="could not parse"):
        resolve_cascade(junk)


def test_classify_face_returns_a_known_class(stub_model) -> None:
    """A face crop is classified into one of the known classes."""
    face = np.zeros((50, 40, 3), dtype=np.uint8)
    name, probability = classify_face(stub_model, face)

    assert name in CLASS_NAMES
    assert 0.0 <= probability <= 1.0


def test_classify_face_feeds_the_network_the_expected_shape(stub_model) -> None:
    """Any crop size is resized to the network's input shape."""
    classify_face(stub_model, np.zeros((37, 91, 3), dtype=np.uint8))
    assert stub_model.calls == [(1, IMAGE_SIZE, IMAGE_SIZE, 3)]


def test_classify_face_preprocesses_into_the_training_range(stub_model) -> None:
    """Inputs must land in [-1, 1], matching preprocess_input used in training.

    The previous implementation divided by 255, giving [0, 1] with the red and
    blue channels transposed relative to training.
    """
    captured = {}

    class Capturing:
        def predict(self, batch, **_):
            captured["batch"] = batch
            return np.array([[0.9, 0.1]], dtype="float32")

    # A black crop is the discriminating case: preprocess_input maps 0 to -1,
    # whereas dividing by 255 maps it to 0. A white crop would pass under both.
    classify_face(Capturing(), np.zeros((60, 60, 3), dtype=np.uint8))
    assert np.isclose(captured["batch"].min(), -1.0)

    # And a white crop still reaches the top of the range.
    classify_face(Capturing(), np.full((60, 60, 3), 255, dtype=np.uint8))
    assert np.isclose(captured["batch"].max(), 1.0)

    # Mid grey lands near zero rather than near 0.5.
    classify_face(Capturing(), np.full((60, 60, 3), 128, dtype=np.uint8))
    assert abs(float(captured["batch"].mean())) < 0.02


def test_classify_face_converts_bgr_to_rgb() -> None:
    """OpenCV supplies BGR; the network was trained on RGB."""
    captured = {}

    class Capturing:
        def predict(self, batch, **_):
            captured["batch"] = batch
            return np.array([[0.9, 0.1]], dtype="float32")

    # Pure blue in BGR is (255, 0, 0); as RGB it must become (0, 0, 255).
    face = np.zeros((40, 40, 3), dtype=np.uint8)
    face[:, :, 0] = 255
    classify_face(Capturing(), face)

    channel_means = captured["batch"][0].reshape(-1, 3).mean(axis=0)
    assert channel_means[2] > channel_means[0]


def test_annotate_frame_draws_when_confident(stub_model) -> None:
    """A confident prediction draws a box onto the frame."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cascade = StubCascade([(10, 10, 20, 20)])

    annotate_frame(frame, stub_model, cascade, confidence=0.5)

    assert frame.any()


def test_annotate_frame_skips_low_confidence_predictions(stub_model) -> None:
    """Below the threshold nothing is drawn."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cascade = StubCascade([(10, 10, 20, 20)])

    annotate_frame(frame, stub_model, cascade, confidence=0.99)

    assert not frame.any()


def test_annotate_frame_clamps_boxes_that_exceed_the_frame(stub_model) -> None:
    """Boxes scaled past the frame edge are clamped, not sliced into nothing."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # In downscaled coordinates; multiplied by DETECTION_SCALE internally.
    beyond_right = (320 // DETECTION_SCALE - 5, 10, 40, 40)
    beyond_bottom = (10, 240 // DETECTION_SCALE - 5, 40, 40)
    cascade = StubCascade([beyond_right, beyond_bottom])

    result = annotate_frame(frame, stub_model, cascade, confidence=0.5)

    assert result.shape == (240, 320, 3)


def test_annotate_frame_skips_empty_crops(stub_model) -> None:
    """A zero-area box is skipped rather than passed to cv2.resize."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cascade = StubCascade([(80, 60, 0, 0)])

    annotate_frame(frame, stub_model, cascade, confidence=0.5)

    assert stub_model.calls == []


def test_annotate_frame_with_no_faces_leaves_the_frame_untouched(stub_model) -> None:
    """No detections means no drawing and no predictions."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    annotate_frame(frame, stub_model, StubCascade([]), confidence=0.5)

    assert not frame.any()
    assert stub_model.calls == []


def test_run_detection_reports_a_missing_model(tmp_path: Path) -> None:
    """Detection checks for the model before touching the camera."""
    config = DetectConfig(name="absent", checkpoints_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="no trained model at"):
        run_detection(config, camera_index=0)
