# FaceMaskDetection

Real-time face mask detection from a webcam, using MobileNetV2 transfer learning.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15-orange)](https://www.tensorflow.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)](#platform-support)
[![CI](https://github.com/theostath/FaceMaskDetection/actions/workflows/ci.yml/badge.svg)](https://github.com/theostath/FaceMaskDetection/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A frozen MobileNetV2 base pretrained on ImageNet, with a small classification
head trained to tell masked faces from unmasked ones. Faces are located in each
camera frame with OpenCV's frontal-face Haar cascade, and every face crop is
classified and labelled in place.

- [Install](#install)
- [Dataset](#dataset)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Platform support](#platform-support)
- [License](#license)

## Install

Requires Python 3.10 or 3.11. TensorFlow 2.15 publishes wheels for CPython 3.9,
3.10 and 3.11 only, which is what sets the upper bound.

```bash
git clone https://github.com/theostath/FaceMaskDetection
cd FaceMaskDetection
pip install -e .
```

That installs two commands, `facemask-train` and `facemask-detect`.

For development, install the test and lint extras as well:

```bash
pip install -e ".[dev]"
```

## Dataset

Download the dataset from this link:
<https://data-flair.training/blogs/download-face-mask-data/>

Unzip it so that the class directories sit beneath `train`:

```text
FaceMaskDetection
└── face-mask-dataset
    └── train
        ├── with_mask
        └── without_mask
```

Directory names are the class labels and must match exactly; any other name is
rejected rather than silently mis-encoded. Point `--dataroot` elsewhere if you
keep the data outside the repository.

## Usage

### Train

```bash
facemask-train --name MaskDetect --verbose
```

Every artifact is written to `checkpoints/<name>/`:

| File | Contents |
| --- | --- |
| `model-<n_epochs>_<name>.h5` | The trained network |
| `accuracy_figure_<name>.png` | Accuracy and loss per epoch |
| `train_logs_<n_epochs>_<name>.txt` | Per-epoch metrics, appended across runs |
| `train_opt.txt` | The settings the run used |
| `model_summary.txt` | Network architecture, with `--verbose` |

| Option | Default | Meaning |
| --- | --- | --- |
| `--name` | `experiment_name` | Experiment name; sets the output directory |
| `--dataroot` | `face-mask-dataset` | Directory holding `train/` |
| `--checkpoints-dir` | `checkpoints` | Where experiment output is written |
| `--n-epochs` | `20` | Training epochs; also part of the model filename |
| `--batch-size` | `32` | Images per batch |
| `--lr` | `0.0001` | Initial Adam learning rate |
| `--beta1`, `--beta2` | `0.9`, `0.999` | Adam decay rates |
| `--test-size` | `0.2` | Fraction held out for validation |
| `--no-flip` | off | Disable horizontal flips during augmentation |
| `--no-show-figure` | off | Save the figure without opening a window |
| `--suffix` | empty | Template appended to the name, e.g. `{n_epochs}ep` |
| `--verbose` | off | Print and save the network architecture |

Training reports a classification report over the validation split when it
finishes. Use `--no-show-figure` to run unattended.

### Detect

```bash
facemask-detect --name MaskDetect
```

A window opens showing the camera feed with a labelled box around each detected
face. Press `Escape` to quit.

| Option | Default | Meaning |
| --- | --- | --- |
| `--name` | `experiment_name` | Experiment to load the model from |
| `--checkpoints-dir` | `checkpoints` | Where to look for the model |
| `--n-epochs` | `20` | Together with `--name`, identifies the model file |
| `--confidence` | `0.85` | Minimum probability before a box is drawn |
| `--camera` | `0` | Camera index |
| `--cascade` | bundled | Explicit Haar cascade file |

`--name` and `--n-epochs` must match the training run, since together they name
the checkpoint. Both commands accept underscored spellings too, so `--n_epochs`
and `--batch_size` work alongside `--n-epochs` and `--batch-size`.

## How it works

MobileNetV2 pretrained on ImageNet provides the convolutional base, with its
classifier head removed and every layer frozen. A new head is trained on top:

```text
MobileNetV2 (frozen)  →  AveragePooling2D (7×7)  →  Flatten
                      →  Dense(128, relu)  →  Dropout(0.5)
                      →  Dense(2, softmax)
```

That leaves 164,226 trainable parameters out of 2,422,210 — which is what makes
the model trainable on a CPU in reasonable time.

Images are resized to 224×224 and scaled to the range [−1, 1] that MobileNetV2
expects. Training augments each batch with random rotation, shifts, shear, zoom
and an optional horizontal flip. Detection applies the identical preprocessing,
including the BGR to RGB conversion that OpenCV frames require, so the network
sees inputs of the same form it was trained on.

## Tests

```bash
pytest
```

The suite runs without a dataset, a camera, a GPU, or network access. Linting:

```bash
ruff check .
```

Both run in CI on Python 3.10 and 3.11 on Linux, and on Python 3.11 on Windows.

## Project layout

```text
FaceMaskDetection
├── pyproject.toml            packaging, dependencies, ruff and pytest config
├── src/facemask
│   ├── config.py             typed settings and every derived output path
│   ├── data.py               dataset discovery, loading, augmentation
│   ├── model.py              network construction, saving, loading
│   ├── train.py              training, logging, evaluation
│   ├── detect.py             camera loop and frame annotation
│   └── cli.py                facemask-train and facemask-detect
├── tests                     pytest suite
├── docs/adr                  architecture decision records
└── .github/workflows/ci.yml  lint and test on push and pull request
```

## Platform support

Linux and Windows, on Python 3.10 and 3.11. CI covers Ubuntu on both versions
and Windows on 3.11. macOS is untested; `tensorflow` rather than
`tensorflow-macos` is declared, so Apple Silicon may need adjusting.

Detection needs a camera OpenCV can open and a display for the preview window.
The Haar cascade is loaded from the installed `opencv-python` package, so no
path needs editing.

## References

- Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*,
  CVPR 2018. <https://arxiv.org/abs/1801.04381>
- Viola and Jones, *Rapid Object Detection using a Boosted Cascade of Simple
  Features*, CVPR 2001.

## License

MIT — see [LICENSE](LICENSE).
