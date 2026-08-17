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
- [Results](#results)
- [Limitations](#limitations)
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

That installs three commands: `facemask-prepare`, `facemask-train` and
`facemask-detect`.

For development, install the test and lint extras as well:

```bash
pip install -e ".[dev]"
```

## Dataset

Neither dataset below needs a Kaggle account. Both ship as one folder per
class, which `facemask-prepare` splits into training and test sets.

| Source | Images | License |
| --- | --- | --- |
| [chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection) | 4,095 | MIT |
| [prajnasb/observations](https://github.com/prajnasb/observations) | 1,376 | none stated |

The second is the dataset this project originally used. It was previously
linked through DataFlair, whose download now returns HTTP 403; `prajnasb` is the
upstream source and is still available.

Fetch just the images rather than the whole repository:

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/chandrikadeb7/Face-Mask-Detection.git /tmp/fmd
git -C /tmp/fmd sparse-checkout set dataset
```

Then split it into train and test sets:

```bash
facemask-prepare --source /tmp/fmd/dataset --dest face-mask-dataset
```

which produces:

```text
FaceMaskDetection
└── face-mask-dataset
    ├── train
    │   ├── with_mask
    │   └── without_mask
    └── test
        ├── with_mask
        └── without_mask
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--source` | required | Directory holding `with_mask/` and `without_mask/` |
| `--dest` | `face-mask-dataset` | Where to create `train/` and `test/` |
| `--test-fraction` | `0.2` | Share of each class held out for testing |
| `--seed` | `42` | Shuffle seed |
| `--move` | off | Move files instead of copying them |

The split is stratified, so both classes keep their proportions, and seeded, so
the same source always produces the same split. Class directory names must be
exactly `with_mask` and `without_mask`; anything else is rejected rather than
silently mis-encoded.

Arranging the directories by hand works just as well. Only `train/` is required
— if `test/` is absent, training reports validation scores alone and says so.
Point `--dataroot` elsewhere to keep the data outside the repository.

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
| `evaluation_<name>.txt` | Classification reports, appended across runs |
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
| `--val-size` | `0.2` | Fraction of `train/` held back for validation |
| `--no-flip` | off | Disable horizontal flips during augmentation |
| `--no-show-figure` | off | Save the figure without opening a window |
| `--suffix` | empty | Template appended to the name, e.g. `{n_epochs}ep` |
| `--verbose` | off | Print and save the network architecture |

Two sets are scored when training finishes. The **validation** split is carved
out of `train/` by `--val-size` and guides training, so its score is optimistic.
The **held-out test** set under `test/` is never seen during training, so it is
the honest number. Both are printed and appended to `evaluation_<name>.txt`.

Use `--no-show-figure` to run unattended.

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

## Results

Training the defaults on the chandrikadeb7 dataset, split with
`facemask-prepare --test-fraction 0.2`:

```bash
facemask-prepare --source /tmp/fmd/dataset --dest face-mask-dataset
facemask-train --name MaskDetect --no-show-figure
```

| Set | Images | Accuracy |
| --- | --- | --- |
| Validation, split from `train/` | 655 | 0.99 |
| Held-out test, from `test/` | 818 | **0.98** |

```text
held-out test   precision    recall  f1-score   support

   with_mask         0.98      0.99      0.98       432
without_mask         0.99      0.97      0.98       386

    accuracy                             0.98       818
```

The held-out figure is the one to quote: those 818 images played no part in
fitting or in any decision about the run. The one-point gap against validation
is the optimism described in [ADR 0004](docs/adr/0004-hold-out-a-test-set.md).

Numbers will shift with a different dataset, split fraction, or seed.

## Limitations

The 0.98 above is measured on images drawn from the same distribution as the
training set. Live camera input is not that distribution, and the gap shows.

**Any lower-face occlusion reads as a mask.** Covering your mouth with a hand
is classified `Mask` at over 99% confidence. The training set contains exactly
two kinds of image: faces wearing masks, and clear unobstructed faces. It
contains no face occluded by something that is not a mask, so the network has
no basis for separating "mask" from "obscured", and has effectively learned the
latter. A scarf, a raised collar, or a hand all trigger it.

**Confidence is not reliability.** That misclassification carries a higher
score than most correct ones. Softmax outputs are only meaningful over the
classes the network was trained to distinguish; on an input unlike anything it
has seen, a high number means the input landed deep inside a region of feature
space, not that the answer is right. Raising `--confidence` does not filter
these out.

**Frontal faces only.** Detection uses a Haar cascade, which wants a
reasonably well-lit, forward-facing face. Profiles, steep angles, and low light
mean no box at all -- the classifier is never consulted.

**Two classes only.** There is no `mask_worn_incorrectly`. A mask below the
nose is simply `Mask`.

**Dataset bias carries through.** The lighting, demographics, and mask types in
the training images bound where the model works. Neither dataset linked above
documents its composition.

Closing the occlusion gap needs training data that represents it -- hand-over-
face and scarf images as `without_mask`, or a third `occluded` class. That is a
data problem, not a code one, and no threshold tuning substitutes for it.

Treat this as a demonstration of transfer learning. It is not suitable for
access control, compliance monitoring, or any decision affecting a person.

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
│   ├── images.py             image file discovery, free of TensorFlow
│   ├── data.py               dataset loading and augmentation
│   ├── prepare.py            splitting a flat dataset into train and test
│   ├── model.py              network construction, saving, loading
│   ├── train.py              training, logging, evaluation
│   ├── detect.py             camera loop and frame annotation
│   └── cli.py                the three console commands
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
