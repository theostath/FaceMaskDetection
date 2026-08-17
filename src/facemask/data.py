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

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from facemask.config import CLASS_NAMES, IMAGE_SIZE, TrainConfig
from facemask.images import find_images


@dataclass(frozen=True)
class TrainingData:
    """Training images split into a fitting half and a validation half.

    This validation split is carved out of ``dataroot/train`` and is used to
    monitor training. It is not the held-out test set, which lives under
    ``dataroot/test`` and is only touched once training has finished.

    Attributes:
        generator: Yields augmented batches drawn from the fitting split.
        train_images: Images used for fitting, shape ``(n, 224, 224, 3)``.
        val_images: Validation images, shape ``(m, 224, 224, 3)``.
        val_labels: One-hot validation labels, shape ``(m, 2)``.
        steps_per_epoch: Batches to draw per epoch from the generator.
        validation_steps: Batches to use per validation pass.
    """

    generator: object
    train_images: np.ndarray
    val_images: np.ndarray
    val_labels: np.ndarray
    steps_per_epoch: int
    validation_steps: int


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


def load_dataset(config: TrainConfig, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess every image in one split of the dataset.

    Each image is resized to 224x224 and scaled to the range [-1, 1] expected
    by MobileNetV2.

    Args:
        config: Settings supplying ``dataroot``.
        split: Subdirectory to read, ``"train"`` or ``"test"``.

    Returns:
        A pair of ``(images, labels)``; images have shape ``(n, 224, 224, 3)``
        and labels are one-hot with shape ``(n, 2)``.

    Raises:
        FileNotFoundError: If the split directory is absent or holds no images.
    """
    directory = config.dataroot / split
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{split} directory not found: {directory}\n"
            "Download a dataset and arrange it so that "
            f"{directory / 'with_mask'} and {directory / 'without_mask'} exist. "
            "See the README for dataset sources."
        )

    print(f"(Info) loading {split} images...")
    paths = find_images(directory)
    if not paths:
        raise FileNotFoundError(f"no images found beneath {directory}")

    images = []
    names = []
    for path in paths:
        image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        images.append(preprocess_input(img_to_array(image)))
        names.append(path.parent.name)

    print(f"(Info) loaded {len(images)} {split} images across {len(set(names))} classes")
    return np.array(images, dtype="float32"), encode_labels(names)


def prepare_training_data(
    config: TrainConfig, images: np.ndarray, labels: np.ndarray
) -> TrainingData:
    """Split the training data and build an augmenting generator for the fitting half.

    Augmentation applies random rotation, shifts, shear, zoom, and optionally a
    horizontal flip, which helps the network generalise beyond the training set.

    Args:
        config: Settings supplying ``val_size``, ``batch_size``, and ``flip``.
        images: Preprocessed images from :func:`load_dataset`.
        labels: Matching one-hot labels.

    Returns:
        The split data together with its generator and per-epoch step counts.

    Raises:
        ValueError: If the split leaves too few images to form a single batch.
    """
    train_images, val_images, train_labels, val_labels = train_test_split(
        images,
        labels,
        test_size=config.val_size,
        stratify=labels,
        random_state=42,
    )

    steps_per_epoch = len(train_images) // config.batch_size
    validation_steps = len(val_images) // config.batch_size
    if steps_per_epoch < 1 or validation_steps < 1:
        raise ValueError(
            f"batch_size {config.batch_size} is too large for this dataset: "
            f"{len(train_images)} fitting and {len(val_images)} validation images "
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
        val_images=val_images,
        val_labels=val_labels,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
