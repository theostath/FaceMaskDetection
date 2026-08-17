"""Training, logging, and evaluation of the mask detection network."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

from facemask.config import CLASS_NAMES, TrainConfig
from facemask.data import TrainingData, load_dataset, prepare_training_data
from facemask.model import build_model, print_summary, save_summary

if TYPE_CHECKING:
    from tensorflow.keras.callbacks import History
    from tensorflow.keras.models import Model


def plot_history(history: History, config: TrainConfig, show: bool = True) -> None:
    """Plot accuracy and loss per epoch and save the figure.

    Args:
        history: The record returned by ``model.fit``.
        config: Settings supplying the figure path.
        show: Whether to open the figure in a window. On a machine with no
            display, matplotlib's non-interactive backend makes this a no-op.
    """
    figure = plt.figure()
    for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
        plt.plot(history.history[key], label=key)
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy/Loss")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")

    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(config.figure_path)
    print(f"(Info) figure saved to {config.figure_path}")

    if show:
        plt.show()
    plt.close(figure)


def write_training_log(history: History, config: TrainConfig, steps_per_epoch: int) -> None:
    """Append a per-epoch record of losses and accuracies to the training log.

    Args:
        history: The record returned by ``model.fit``.
        config: Settings supplying the log path.
        steps_per_epoch: Batches drawn per epoch, recorded alongside each row.
    """
    losses = history.history["loss"]
    accuracy = history.history["accuracy"]
    val_loss = history.history["val_loss"]
    val_accuracy = history.history["val_accuracy"]

    # Derived from the history rather than config.n_epochs so that an
    # interrupted run logs the epochs it actually completed instead of
    # raising IndexError part way through.
    lines = [
        f"epoch: {epoch + 1} - iters: {steps_per_epoch} - loss: {losses[epoch]:.4f} - "
        f"accuracy: {accuracy[epoch]:.4f} - val_loss: {val_loss[epoch]:.4f} - "
        f"val_accuracy: {val_accuracy[epoch]:.4f}"
        for epoch in range(len(losses))
    ]

    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    with config.train_log_path.open("a", encoding="utf-8") as log_file:
        now = time.strftime("%c")
        log_file.write(f"================ Training Loss ({now}) ================\n")
        log_file.write("\n" + "\n".join(lines) + "\n\n")


def evaluate(model: Model, data: TrainingData, config: TrainConfig) -> str:
    """Score the trained network on the validation split.

    Args:
        model: The trained network.
        data: The split holding the validation images and labels.
        config: Settings supplying the batch size.

    Returns:
        The scikit-learn classification report as a string.
    """
    print("(Info) evaluating network...")
    probabilities = model.predict(data.test_images, batch_size=config.batch_size)
    predictions = np.argmax(probabilities, axis=1)

    report = classification_report(
        data.test_labels.argmax(axis=1),
        predictions,
        target_names=list(CLASS_NAMES),
    )
    print(report)
    return report


def train(config: TrainConfig, show_figure: bool = True) -> Model:
    """Run a full training pass and write every artifact for the experiment.

    The trained model is saved immediately after fitting, before the figure is
    drawn, so that a long training run is not lost if the figure window is
    closed or interrupted.

    Args:
        config: Settings for the run.
        show_figure: Whether to display the accuracy figure when training ends.

    Returns:
        The trained model.
    """
    config.save()
    print(config.describe())

    images, labels = load_dataset(config)
    data = prepare_training_data(config, images, labels)

    model = build_model(config)
    if config.verbose:
        print_summary(model)
        save_summary(model, config.summary_path)

    print("(Info) training head of network...")
    history = model.fit(
        data.generator,
        steps_per_epoch=data.steps_per_epoch,
        epochs=config.n_epochs,
        validation_data=(data.test_images, data.test_labels),
        validation_steps=data.validation_steps,
    )

    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    model.save(config.model_path)
    print(f"(Info) model saved to {config.model_path}")

    write_training_log(history, config, data.steps_per_epoch)
    plot_history(history, config, show=show_figure)
    evaluate(model, data, config)

    return model
