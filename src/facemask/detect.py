"""Live face mask detection from a camera feed.

Faces are located with OpenCV's frontal-face Haar cascade, then each face crop
is classified by the trained network. Detection runs on a downscaled copy of
the frame -- the cascade is the slow part, and it does not need full
resolution -- while the boxes are drawn on the full-size frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facemask.config import CLASS_NAMES, IMAGE_SIZE, DetectConfig
from facemask.model import load_trained_model

#: Factor by which frames are downscaled before face detection.
DETECTION_SCALE = 4

#: Name of the cascade shipped with the opencv-python wheel.
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

#: Key that closes the preview window.
ESCAPE_KEY = 27

MASK_COLOUR = (0, 255, 0)
NO_MASK_COLOUR = (0, 0, 255)


def resolve_cascade(path: Path | None = None) -> cv2.CascadeClassifier:
    """Load the frontal-face Haar cascade.

    The cascade ships inside the opencv-python wheel, so its location is
    derived from the installed package rather than hardcoded. The previous
    implementation embedded an absolute path into one developer's conda
    environment and instructed users to edit the source before running.

    Args:
        path: Explicit cascade file to use instead of the bundled one.

    Returns:
        A loaded cascade classifier.

    Raises:
        FileNotFoundError: If the cascade file does not exist.
        RuntimeError: If the file exists but OpenCV cannot parse it.
    """
    if path is not None:
        cascade_path = Path(path)
    else:
        cascade_path = Path(cv2.data.haarcascades) / CASCADE_FILENAME

    if not cascade_path.is_file():
        raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")

    # A malformed file makes the constructor raise rather than return an empty
    # classifier, so both outcomes have to be handled. The binding surfaces the
    # underlying cv2.error as a SystemError ("returned a result with an
    # exception set"), so both types have to be caught.
    try:
        classifier = cv2.CascadeClassifier(str(cascade_path))
    except (cv2.error, SystemError) as exc:
        raise RuntimeError(f"OpenCV could not parse the Haar cascade at {cascade_path}") from exc

    if classifier.empty():
        raise RuntimeError(f"OpenCV could not load the Haar cascade at {cascade_path}")

    return classifier


def classify_face(model: object, face: np.ndarray) -> tuple[str, float]:
    """Classify a single face crop.

    Args:
        model: The trained network.
        face: Face crop in BGR, of any size.

    Returns:
        The predicted class name and its probability.
    """
    resized = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
    normalized = resized / 255.0
    batch = np.reshape(normalized, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return CLASS_NAMES[index], float(probabilities[index])


def annotate_frame(
    frame: np.ndarray,
    model: object,
    classifier: cv2.CascadeClassifier,
    confidence: float,
) -> np.ndarray:
    """Draw a labelled box around every confidently classified face.

    Args:
        frame: Full-size BGR frame; modified in place and returned.
        model: The trained network.
        classifier: Face detector.
        confidence: Minimum probability before a box is drawn.

    Returns:
        The annotated frame.
    """
    small = cv2.resize(
        frame, (frame.shape[1] // DETECTION_SCALE, frame.shape[0] // DETECTION_SCALE)
    )

    for face_box in classifier.detectMultiScale(small):
        x, y, width, height = (int(value) * DETECTION_SCALE for value in face_box)

        # Clamp to the frame; the cascade can return boxes that extend past
        # the edge once scaled back up, and slicing past it yields an empty
        # crop that cv2.resize rejects.
        x, y = max(x, 0), max(y, 0)
        right, bottom = min(x + width, frame.shape[1]), min(y + height, frame.shape[0])
        face = frame[y:bottom, x:right]
        if face.size == 0:
            continue

        name, probability = classify_face(model, face)
        if probability <= confidence:
            continue

        label = "Mask" if name == CLASS_NAMES[0] else "No Mask"
        colour = MASK_COLOUR if name == CLASS_NAMES[0] else NO_MASK_COLOUR
        cv2.putText(
            frame,
            f"{label}: {probability * 100:.2f}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            2,
        )
        cv2.rectangle(frame, (x, y), (right, bottom), colour, 2)

    return frame


def run_detection(
    config: DetectConfig,
    camera_index: int = 0,
    cascade_path: Path | None = None,
) -> None:
    """Open the camera and classify faces until Escape is pressed.

    Args:
        config: Settings supplying the model path and confidence threshold.
        camera_index: Index of the camera to open.
        cascade_path: Explicit cascade file, if not the bundled one.

    Raises:
        FileNotFoundError: If the model or the cascade is missing.
        RuntimeError: If the camera cannot be opened.
    """
    model = load_trained_model(config)
    classifier = resolve_cascade(cascade_path)

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"could not open camera {camera_index}. Check that a camera is connected "
            "and not already in use by another application."
        )

    print("(Info) detecting; press Escape to quit")
    try:
        while True:
            read_ok, frame = capture.read()
            if not read_ok:
                print("(Warning) dropped frame from camera")
                continue

            frame = cv2.flip(frame, 1)
            annotate_frame(frame, model, classifier, config.confidence)

            cv2.imshow("LIVE", frame)
            if cv2.waitKey(20) == ESCAPE_KEY:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
