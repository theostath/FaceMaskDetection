"""Typed configuration objects for the training and detection entry points.

These dataclasses replace the previous ``BaseOptions``/``TrainOptions``/
``TestOptions`` argparse-class hierarchy. Configuration here is pure data with
no argparse dependency, so it can be constructed and tested directly; the
command line interface in :mod:`facemask.cli` is responsible for turning
arguments into one of these objects.

Every path derived from a run's settings -- the saved model, the training log,
the accuracy figure -- is computed once here and read everywhere else. Keeping
a single definition of the model filename is what prevents the save and load
sides from disagreeing about where a checkpoint lives.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

DEFAULT_DATAROOT = Path("face-mask-dataset")
DEFAULT_CHECKPOINTS_DIR = Path("checkpoints")

#: Class labels, ordered to match the network's output units. The order is
#: fixed by ``LabelBinarizer``, which sorts class names alphabetically.
CLASS_NAMES: tuple[str, str] = ("with_mask", "without_mask")

#: Side length in pixels of the square input the network expects.
IMAGE_SIZE = 224


@dataclass
class BaseConfig:
    """Settings shared by the training and detection entry points.

    Attributes:
        name: Experiment name; determines the checkpoint subdirectory.
        dataroot: Directory holding the ``train`` image folders.
        checkpoints_dir: Directory under which experiment output is written.
        n_epochs: Number of training epochs; also part of the model filename.
        batch_size: Images per batch.
        flip: Whether to flip images horizontally during augmentation.
        suffix: Optional template appended to ``name``, formatted against the
            other settings, e.g. ``"{n_epochs}ep"``.
        verbose: Print the network architecture and save it alongside the run.
    """

    name: str = "experiment_name"
    dataroot: Path = DEFAULT_DATAROOT
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR
    n_epochs: int = 20
    batch_size: int = 32
    flip: bool = True
    suffix: str = ""
    verbose: bool = False

    #: Names the settings file written into the experiment directory.
    phase: ClassVar[str] = "base"

    def __post_init__(self) -> None:
        """Normalise paths, apply the name suffix, and validate the settings.

        The suffix is applied here, which runs exactly once per instance. The
        previous implementation applied it during option parsing *and* again
        when loading a network, producing a doubled name and a checkpoint path
        that pointed at a file which was never written.

        Raises:
            ValueError: If a numeric setting is outside its valid range, or the
                suffix template references an unknown field.
        """
        self.dataroot = Path(self.dataroot)
        self.checkpoints_dir = Path(self.checkpoints_dir)

        if self.suffix:
            values = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
            try:
                self.name = f"{self.name}_{self.suffix.format(**values)}"
            except KeyError as exc:
                known = ", ".join(sorted(values))
                raise ValueError(
                    f"suffix template references unknown field {exc}; available fields: {known}"
                ) from exc

        if not self.name:
            raise ValueError("name must not be empty")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be at least 1, got {self.n_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")

    @property
    def train_dir(self) -> Path:
        """Directory holding the training images, one folder per class."""
        return self.dataroot / "train"

    @property
    def test_dir(self) -> Path:
        """Directory holding the held-out test images, one folder per class."""
        return self.dataroot / "test"

    @property
    def has_test_split(self) -> bool:
        """Whether a held-out test set is available alongside the training data."""
        return self.test_dir.is_dir()

    @property
    def experiment_dir(self) -> Path:
        """Directory holding every artifact produced by this run."""
        return self.checkpoints_dir / self.name

    @property
    def model_path(self) -> Path:
        """Path of the saved model, used by both the training and detection sides."""
        return self.experiment_dir / f"model-{self.n_epochs}_{self.name}.h5"

    @property
    def summary_path(self) -> Path:
        """Path of the saved network architecture summary."""
        return self.experiment_dir / "model_summary.txt"

    @property
    def figure_path(self) -> Path:
        """Path of the saved accuracy and loss figure."""
        return self.experiment_dir / f"accuracy_figure_{self.name}.png"

    @property
    def train_log_path(self) -> Path:
        """Path of the appended per-epoch training log."""
        return self.experiment_dir / f"train_logs_{self.n_epochs}_{self.name}.txt"

    @property
    def evaluation_path(self) -> Path:
        """Path of the saved classification reports."""
        return self.experiment_dir / f"evaluation_{self.name}.txt"

    @property
    def settings_path(self) -> Path:
        """Path of the saved copy of these settings."""
        return self.experiment_dir / f"{self.phase}_opt.txt"

    def describe(self) -> str:
        """Render the settings as a table, marking any that differ from default.

        Returns:
            A multi-line string suitable for printing or writing to a file.
        """
        lines = ["----------------- Options ---------------"]
        for field in sorted(dataclasses.fields(self), key=lambda f: f.name):
            value = getattr(self, field.name)
            comment = "" if value == field.default else f"\t[default: {field.default}]"
            lines.append(f"{field.name:>25}: {value!s:<30}{comment}")
        lines.append("----------------- End -------------------")
        return "\n".join(lines)

    def save(self) -> None:
        """Write the settings into the experiment directory, creating it if needed.

        A permission error is reported but not raised: failing to record the
        settings should not abort a training run that is otherwise fine.
        """
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.settings_path.write_text(f"{self.describe()}\n", encoding="utf-8")
        except PermissionError as error:
            print(f"(Warning) could not save settings: {error}")


@dataclass
class TrainConfig(BaseConfig):
    """Settings for a training run.

    Attributes:
        lr: Initial learning rate for Adam.
        beta1: Exponential decay rate for Adam's first moment estimates.
        beta2: Exponential decay rate for Adam's second moment estimates.
        val_size: Fraction of the training data held back for validation. This
            is distinct from the held-out test set under ``dataroot/test``,
            which the model never sees during training.
    """

    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    val_size: float = 0.20

    phase: ClassVar[str] = "train"

    def __post_init__(self) -> None:
        """Validate the training hyperparameters after the shared checks.

        Raises:
            ValueError: If a hyperparameter is outside its valid range.
        """
        super().__post_init__()
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not 0 <= self.beta1 < 1:
            raise ValueError(f"beta1 must be in [0, 1), got {self.beta1}")
        if not 0 <= self.beta2 < 1:
            raise ValueError(f"beta2 must be in [0, 1), got {self.beta2}")
        if not 0 < self.val_size < 1:
            raise ValueError(f"val_size must be in (0, 1), got {self.val_size}")


@dataclass
class DetectConfig(BaseConfig):
    """Settings for live detection from a camera.

    Attributes:
        confidence: Minimum softmax probability before a box is drawn.
    """

    confidence: float = 0.85

    phase: ClassVar[str] = "test"

    def __post_init__(self) -> None:
        """Validate the detection settings after the shared checks.

        Raises:
            ValueError: If the confidence threshold is outside [0, 1].
        """
        super().__post_init__()
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
