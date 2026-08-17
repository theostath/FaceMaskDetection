# ADR 0002: Pin TensorFlow to 2.15 and cap Python below 3.12

## Status

Accepted

## Context

The project was written against `keras==2.9.0` in 2022 and depends on two APIs
that have since been removed or deprecated:

- `ImageDataGenerator`, used for training augmentation, was removed in Keras 3
  (TensorFlow 2.16 and later).
- `Adam(decay=...)` was removed earlier still, in the Keras 2.11 optimizer
  rewrite. On any TensorFlow newer than 2.10 the call raises:

  ```text
  ValueError: decay is deprecated in the new Keras optimizer, please check
  the docstring for valid arguments, or use the legacy optimizer
  ```

The code therefore ran on no currently installable TensorFlow. A version had to
be chosen before anything else could be verified.

## Decision

Pin `tensorflow>=2.15,<2.16`, declare `requires-python = ">=3.10,<3.12"`, and
replace `Adam(decay=...)` with an equivalent `InverseTimeDecay` schedule.
Porting to Keras 3 is deliberately deferred.

## Rationale

TensorFlow 2.15 is the last release before Keras 3 became the default, so
`ImageDataGenerator` still exists and the augmentation pipeline is unchanged.
This keeps the modernization behaviour-preserving: differences observed after
the port are attributable to the restructure, not to a framework migration
happening at the same time.

The Python upper bound tracks wheel availability rather than a language
preference. TensorFlow 2.15 publishes wheels for `cp39`, `cp310` and `cp311`
only; resolution against 3.12 fails with no matching wheel. Python 3.9 is
excluded because it reached end of life in October 2025.

`Adam(decay=...)` could have been kept via `tf.keras.optimizers.legacy.Adam`,
but that path is itself removed in Keras 3. `InverseTimeDecay` with
`decay_steps=1` computes `lr / (1 + decay * step)`, which is exactly what the
legacy `decay` argument did, and it survives the eventual port. Equivalence was
confirmed against `legacy.Adam` at steps 0, 1, 10, 100, 1000 and 5000, agreeing
to within 1e-12 relative tolerance.

## Consequences

- The project runs again on a currently installable TensorFlow.
- Python 3.12 and later are unavailable until the Keras 3 port happens.
- Models are saved as `.h5`, which Keras now warns is legacy in favour of
  `.keras`. The format is kept so existing checkpoints still load.
- A Keras 3 port remains outstanding. It touches augmentation
  (`ImageDataGenerator` to `tf.data` or preprocessing layers), the optimizer
  schedule, and the serialization format, and it is what lifts the Python cap.
- `ImageDataGenerator` emits deprecation warnings from SciPy and TensorFlow
  during training. They originate inside Keras, not this project's code.
