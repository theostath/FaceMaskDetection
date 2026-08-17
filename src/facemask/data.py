"""Dataset discovery, loading, and augmentation.

Images are expected in one directory per class beneath ``<dataroot>/train``::

    face-mask-dataset/
    └── train/
        ├── with_mask/
        └── without_mask/

The directory name is the label. Class order follows :data:`CLASS_NAMES`, which
is alphabetical and therefore matches what ``LabelBinarizer`` produced in the
previous implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from facemask.config import CLASS_NAMES, IMAGE_SIZE, TrainConfig

#: File extensions treated as images, matching what imutils.paths.list_images
#: accepted in the previous implementation.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


@dataclass(frozen=True)
class TrainingData:
    """A dataset split into training and validation halves.

    Attributes:
        generator: Yields augmented batches drawn from the training split.
        train_images: Training images, shape ``(n, 224, 224, 3)``.
        test_images: Validation images, shape ``(m, 224, 224, 3)``.
        test_labels: One-hot validation labels, shape ``(m, 2)``.
        steps_per_epoch: Batches to draw per epoch from the generator.
        validation_steps: Batches to use per validation pass.
    """

    generator: object
    train_images: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    steps_per_epoch: int
    validation_steps: int


def find_images(directory: Path) -> list[Path]:
    """Recursively collect image files beneath a directory, in a stable order.

    Replaces ``imutils.paths.list_images``. Sorting makes dataset order
    reproducible across platforms, which the previous implementation did not
    guarantee because it depended on filesystem walk order.

    Args:
        directory: Directory to search.

    Returns:
        Sorted paths of every image file found beneath ``directory``.
    """
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def encode_labels(names: list[str]) -> np.ndarray:
    """One-hot encode class names according to :data:`CLASS_NAMES`.

    Args:
        names: Per-image class names, each matching a directory name.

    Returns:
        One-hot array of shape ``(len(names), 2)``, where column 0 is
        ``with_mask`` and column 1 is ``without_mask``.

    Raises:
        ValueError: If a name is not one of the known classes.
    """
    index = {name: position for position, name in enumerate(CLASS_NAMES)}
    unknown = sorted(set(names) - index.keys())
    if unknown:
        expected = ", ".join(CLASS_NAMES)
        raise ValueError(
            f"unexpected class directories {unknown}; expected exactly: {expected}"
        )

    encoded = np.zeros((len(names), len(CLASS_NAMES)), dtype="float32")
    encoded[np.arange(len(names)), [index[name] for name in names]] = 1.0
    return encoded


def load_dataset(config: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess every training image.

    Each image is resized to 224x224 and scaled to the range [-1, 1] expected
    by MobileNetV2.

    Args:
        config: Settings supplying ``dataroot``.

    Returns:
        A pair of ``(images, labels)``; images have shape ``(n, 224, 224, 3)``
        and labels are one-hot with shape ``(n, 2)``.

    Raises:
        FileNotFoundError: If the train directory is absent or holds no images.
    """
    train_dir = config.dataroot / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"training directory not found: {train_dir}\n"
            "Download the dataset and unpack it so that "
            f"{train_dir / 'with_mask'} and {train_dir / 'without_mask'} exist."
        )

    print("(Info) loading images...")
    paths = find_images(train_dir)
    if not paths:
        raise FileNotFoundError(f"no images found beneath {train_dir}")

    images = []
    names = []
    for path in paths:
        image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        images.append(preprocess_input(img_to_array(image)))
        names.append(path.parent.name)

    print(f"(Info) loaded {len(images)} images across {len(set(names))} classes")
    return np.array(images, dtype="float32"), encode_labels(names)


def prepare_training_data(
    config: TrainConfig, images: np.ndarray, labels: np.ndarray
) -> TrainingData:
    """Split the dataset and build an augmenting generator for the training half.

    Augmentation applies random rotation, shifts, shear, zoom, and optionally a
    horizontal flip, which helps the network generalise beyond the training set.

    Args:
        config: Settings supplying ``test_size``, ``batch_size``, and ``flip``.
        images: Preprocessed images from :func:`load_dataset`.
        labels: Matching one-hot labels.

    Returns:
        The split data together with its generator and per-epoch step counts.

    Raises:
        ValueError: If the split leaves too few images to form a single batch.
    """
    train_images, test_images, train_labels, test_labels = train_test_split(
        images,
        labels,
        test_size=config.test_size,
        stratify=labels,
        random_state=42,
    )

    steps_per_epoch = len(train_images) // config.batch_size
    validation_steps = len(test_images) // config.batch_size
    if steps_per_epoch < 1 or validation_steps < 1:
        raise ValueError(
            f"batch_size {config.batch_size} is too large for this dataset: "
            f"{len(train_images)} training and {len(test_images)} validation images "
            "yield fewer than one batch. Use a smaller --batch-size."
        )

    augmenter = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=config.flip,
        fill_mode="nearest",
    )

    return TrainingData(
        generator=augmenter.flow(train_images, train_labels, batch_size=config.batch_size),
        train_images=train_images,
        test_images=test_images,
        test_labels=test_labels,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
