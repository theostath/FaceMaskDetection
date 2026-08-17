"""Tests for the command line interface.

Argument parsing is exercised directly so these run without importing
TensorFlow, which the entry points defer until after parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from facemask.cli import build_detect_parser, build_train_parser, detect_main, train_main


def test_train_defaults_match_the_config_defaults() -> None:
    """Parser defaults agree with the dataclass they populate."""
    args = build_train_parser().parse_args([])
    assert args.name == "experiment_name"
    assert args.n_epochs == 20
    assert args.batch_size == 32
    assert args.lr == pytest.approx(1e-4)
    assert args.test_size == pytest.approx(0.20)
    assert args.flip is True
    assert args.show_figure is True


def test_detect_defaults() -> None:
    """Detection defaults match the previous test_options values."""
    args = build_detect_parser().parse_args([])
    assert args.confidence == pytest.approx(0.85)
    assert args.camera == 0
    assert args.cascade is None


@pytest.mark.parametrize(
    ("flag", "attribute", "value"),
    [
        ("--n_epochs", "n_epochs", 5),
        ("--n-epochs", "n_epochs", 5),
        ("--batch_size", "batch_size", 8),
        ("--batch-size", "batch_size", 8),
    ],
)
def test_both_flag_spellings_are_accepted(flag: str, attribute: str, value: int) -> None:
    """The previously documented underscore spellings keep working."""
    args = build_train_parser().parse_args([flag, str(value)])
    assert getattr(args, attribute) == value


@pytest.mark.parametrize("flag", ["--checkpoints_dir", "--checkpoints-dir"])
def test_both_checkpoints_dir_spellings_are_accepted(flag: str) -> None:
    """Paths are parsed into Path objects under either spelling."""
    args = build_train_parser().parse_args([flag, "somewhere"])
    assert args.checkpoints_dir == Path("somewhere")


@pytest.mark.parametrize("flag", ["--no_flip", "--no-flip"])
def test_no_flip_disables_augmentation_flipping(flag: str) -> None:
    """Either spelling of the negative flag clears the flip setting."""
    assert build_train_parser().parse_args([flag]).flip is False


def test_no_show_figure_disables_the_window() -> None:
    """Training can run unattended without opening a figure."""
    assert build_train_parser().parse_args(["--no-show-figure"]).show_figure is False


def test_help_exits_zero() -> None:
    """--help succeeds for both commands."""
    for parser in (build_train_parser(), build_detect_parser()):
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--help"])
        assert excinfo.value.code == 0


def test_version_reports_the_package_version() -> None:
    """--version prints the installed version and exits successfully."""
    from facemask import __version__

    with pytest.raises(SystemExit) as excinfo:
        build_train_parser().parse_args(["--version"])
    assert excinfo.value.code == 0
    assert __version__


def test_unknown_argument_is_rejected() -> None:
    """An unrecognised option fails rather than being silently ignored."""
    with pytest.raises(SystemExit) as excinfo:
        build_train_parser().parse_args(["--not-an-option"])
    assert excinfo.value.code == 2


def test_train_main_reports_a_missing_dataset(tmp_path: Path, capsys) -> None:
    """A missing dataset exits 1 with a message rather than a traceback."""
    code = train_main(
        [
            "--name",
            "x",
            "--dataroot",
            str(tmp_path / "absent"),
            "--checkpoints-dir",
            str(tmp_path / "ckpt"),
        ]
    )
    assert code == 1
    assert "Error:" in capsys.readouterr().out


def test_detect_main_reports_a_missing_model(tmp_path: Path, capsys) -> None:
    """Detection without a trained model exits 1 and suggests training one."""
    code = detect_main(["--name", "absent", "--checkpoints-dir", str(tmp_path)])
    assert code == 1
    output = capsys.readouterr().out
    assert "no trained model" in output
    assert "facemask-train" in output


def test_detect_main_rejects_an_out_of_range_confidence(tmp_path: Path, capsys) -> None:
    """Configuration validation surfaces as a clean CLI error."""
    code = detect_main(
        ["--name", "x", "--checkpoints-dir", str(tmp_path), "--confidence", "5"]
    )
    assert code == 1
    assert "confidence must be in [0, 1]" in capsys.readouterr().out
