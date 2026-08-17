"""Construction, persistence, and loading of the mask detection network.

The network is MobileNetV2 pretrained on ImageNet with its classifier head
removed, followed by a small fully connected head trained from scratch. The
convolutional base stays frozen, so training only fits the head -- a few
hundred thousand parameters instead of several million, which is what makes
the model trainable on a CPU in a reasonable time.
"""

from __future__ import annotations

from pathlib import Path

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import InverseTimeDecay

from facemask.config import CLASS_NAMES, IMAGE_SIZE, BaseConfig, TrainConfig

#: Neurons in the hidden layer of the classification head.
HEAD_UNITS = 128

#: Fraction of head activations dropped during training.
DROPOUT_RATE = 0.5

#: Pool window applied to the 7x7 feature map MobileNetV2 emits for a 224x224 input.
POOL_SIZE = (7, 7)


def build_model(config: TrainConfig) -> Model:
    """Build and compile the mask detection network.

    The MobileNetV2 base is frozen, so only the classification head is fitted.

    Args:
        config: Settings supplying the Adam hyperparameters and epoch count,
            the latter setting the optimizer's decay schedule.

    Returns:
        A compiled model accepting ``(None, 224, 224, 3)`` inputs preprocessed
        to the range [-1, 1], and emitting a softmax over
        :data:`~facemask.config.CLASS_NAMES`.
    """
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    )

    # Freeze the pretrained base; only the head below is trained.
    for layer in base.layers:
        layer.trainable = False

    head = base.output
    head = AveragePooling2D(pool_size=POOL_SIZE)(head)
    head = Flatten(name="flatten")(head)
    head = Dense(HEAD_UNITS, activation="relu")(head)
    head = Dropout(DROPOUT_RATE)(head)
    head = Dense(len(CLASS_NAMES), activation="softmax")(head)

    model = Model(inputs=base.input, outputs=head)

    print("(Info) compiling model...")

    # The original passed decay=lr/n_epochs to Adam. Keras removed that
    # argument in 2.11, so the call raises on any modern TensorFlow.
    # InverseTimeDecay with decay_steps=1 is its exact replacement: legacy
    # decay computed lr / (1 + decay * step), which is what this schedule
    # evaluates to. Note the step is an optimizer iteration, not an epoch.
    schedule = InverseTimeDecay(
        initial_learning_rate=config.lr,
        decay_steps=1,
        decay_rate=config.lr / config.n_epochs,
    )
    optimizer = Adam(learning_rate=schedule, beta_1=config.beta1, beta_2=config.beta2)

    # categorical_crossentropy is the loss that matches a softmax head with
    # one-hot labels. At two units it happens to be algebraically identical to
    # the binary_crossentropy used previously -- both reduce to -log(p) for the
    # correct class -- so this changes nothing today. It stops being identical
    # the moment a third class is added, where binary_crossentropy would keep
    # training without complaint on a quietly wrong objective.
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def save_summary(model: Model, path: Path) -> None:
    """Write the network architecture to a text file, creating its directory.

    Args:
        model: The model to describe.
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(model: Model) -> None:
    """Print the network architecture between two rules.

    Args:
        model: The model to describe.
    """
    print("---------- Network initialized -------------")
    model.summary()
    print("-----------------------------------------------")


def load_trained_model(config: BaseConfig) -> Model:
    """Load a previously trained model from the path implied by the settings.

    Args:
        config: Settings whose ``model_path`` locates the checkpoint.

    Returns:
        The loaded model, ready for inference.

    Raises:
        FileNotFoundError: If no checkpoint exists at that path, with the
            training command that would produce it.
    """
    if not config.model_path.is_file():
        raise FileNotFoundError(
            f"no trained model at {config.model_path}\n"
            f"Train one first: facemask-train --name {config.name} "
            f"--n-epochs {config.n_epochs}"
        )
    return load_model(config.model_path)
