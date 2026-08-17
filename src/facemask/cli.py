"""Command line entry points for dataset preparation, training, and detection.

Three commands are installed with the package:

``facemask-prepare``
    Split a flat two-class image directory into train and test sets.

``facemask-train``
    Train the classifier on a downloaded dataset.

``facemask-detect``
    Run live detection from a camera using a trained model.

Multi-word options accept both hyphenated and underscored spellings, so the
commands documented for the previous script layout (``--n_epochs``,
``--batch_size``) keep working alongside the conventional ``--n-epochs``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from facemask import __version__
from facemask.config import (
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_DATAROOT,
    DetectConfig,
    TrainConfig,
)


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the options common to both commands.

    Args:
        parser: Parser to extend.
    """
    parser.add_argument(
        "--name",
        default="experiment_name",
        help="name of the experiment; decides where models are stored",
    )
    parser.add_argument(
        "--checkpoints-dir",
        "--checkpoints_dir",
        dest="checkpoints_dir",
        type=Path,
        default=DEFAULT_CHECKPOINTS_DIR,
        help="directory holding experiment output",
    )
    parser.add_argument(
        "--n-epochs",
        "--n_epochs",
        dest="n_epochs",
        type=int,
        default=20,
        help="number of training epochs; also part of the model filename",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=32,
        help="images per batch",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="template appended to the experiment name, e.g. '{n_epochs}ep'",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print the network architecture and save it alongside the run",
    )
    parser.add_argument("--version", action="version", version=f"facemask {__version__}")


def build_train_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training command.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="facemask-train",
        description="Train the face mask classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_arguments(parser)
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=DEFAULT_DATAROOT,
        help="directory holding the train/with_mask and train/without_mask folders",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate for Adam")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta_1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta_2")
    parser.add_argument(
        "--val-size",
        "--test-size",
        "--test_size",
        dest="val_size",
        type=float,
        default=0.20,
        help="fraction of the training data held back for validation",
    )
    parser.add_argument(
        "--no-flip",
        "--no_flip",
        dest="flip",
        action="store_false",
        help="do not flip images horizontally during augmentation",
    )
    parser.add_argument(
        "--no-show-figure",
        dest="show_figure",
        action="store_false",
        help="save the accuracy figure without opening a window",
    )
    return parser


def build_detect_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the detection command.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="facemask-detect",
        description="Detect face masks live from a camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_arguments(parser)
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.85,
        help="minimum probability before a box is drawn",
    )
    parser.add_argument("--camera", type=int, default=0, help="index of the camera to open")
    parser.add_argument(
        "--cascade",
        type=Path,
        default=None,
        help="explicit Haar cascade file; defaults to the one bundled with OpenCV",
    )
    return parser


def build_prepare_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the dataset preparation command.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="facemask-prepare",
        description=(
            "Split a flat two-class image directory into train and test sets. "
            "The split is stratified and seeded, so it is reproducible."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="directory containing with_mask/ and without_mask/ folders",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DATAROOT,
        help="directory to create train/ and test/ beneath",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="share of each class held out for testing",
    )
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed")
    parser.add_argument(
        "--move",
        action="store_true",
        help="move files instead of copying them",
    )
    parser.add_argument("--version", action="version", version=f"facemask {__version__}")
    return parser


def prepare_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``facemask-prepare``.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 if the dataset could not be prepared.
    """
    args = build_prepare_parser().parse_args(argv)

    from facemask.prepare import prepare_dataset

    try:
        counts = prepare_dataset(
            source=args.source,
            destination=args.dest,
            test_fraction=args.test_fraction,
            seed=args.seed,
            move=args.move,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        return _report(error)

    total = sum(sum(classes.values()) for classes in counts.values())
    print(f"(Info) prepared {total} images under {args.dest}")
    return 0


def _report(error: Exception) -> int:
    """Print an error without a traceback and return a failing exit code.

    Args:
        error: The exception to report.

    Returns:
        The exit code 1.
    """
    print(f"Error: {error}")
    return 1


def train_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``facemask-train``.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 if the run could not be completed.
    """
    args = build_train_parser().parse_args(argv)

    # Imported here so that --help and argument errors do not pay the cost of
    # importing TensorFlow, which takes several seconds.
    from facemask.train import train

    try:
        config = TrainConfig(
            name=args.name,
            dataroot=args.dataroot,
            checkpoints_dir=args.checkpoints_dir,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            flip=args.flip,
            suffix=args.suffix,
            verbose=args.verbose,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            val_size=args.val_size,
        )
        train(config, show_figure=args.show_figure)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        return _report(error)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    return 0


def detect_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``facemask-detect``.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 if detection could not start.
    """
    args = build_detect_parser().parse_args(argv)

    from facemask.detect import run_detection

    try:
        config = DetectConfig(
            name=args.name,
            checkpoints_dir=args.checkpoints_dir,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            suffix=args.suffix,
            verbose=args.verbose,
            confidence=args.confidence,
        )
        run_detection(config, camera_index=args.camera, cascade_path=args.cascade)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        return _report(error)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    return 0


__all__ = [
    "build_detect_parser",
    "build_prepare_parser",
    "build_train_parser",
    "detect_main",
    "prepare_main",
    "train_main",
]
