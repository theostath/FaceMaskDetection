# ADR 0003: Replace the option-class hierarchy with typed dataclasses

## Status

Accepted

## Context

Configuration lived in a three-class hierarchy, `BaseOptions` with `TrainOptions`
and `TestOptions` subclasses, each contributing `argparse` arguments and
returning a `Namespace`. Settings could not be constructed without going through
a command line, which made them untestable in isolation.

Output paths were derived ad hoc at each use site. The model filename in
particular had two independent definitions, one in `train.py` for saving and one
in `BaseModel.load_network` for loading, and they disagreed whenever `--suffix`
was set: option parsing applied the suffix, then `load_network` applied it a
second time.

```text
after parse()            MaskDetect_20ep
after load_network()     MaskDetect_20ep_20ep
train.py saved to        model-20_MaskDetect_20ep.h5
load_network looked for  model-20_MaskDetect_20ep_20ep.h5
```

Detection could therefore never find a model trained with a suffix. Two parsed
options, `--nf` and `--no_dropout`, were never read by any code.

## Decision

Replace the hierarchy with `BaseConfig` and its `TrainConfig` and `DetectConfig`
subclasses: plain dataclasses with type hints, validation in `__post_init__`,
and a property for every derived path. Argument parsing moves to `cli.py`, which
constructs one of these objects. Drop the unused options.

## Rationale

Configuration as data can be constructed directly in a test, which is what makes
the path and validation behaviour testable without invoking a CLI.

Deriving every path from one place removes the class of bug above by
construction. `model_path` has a single definition that both the training and
detection sides read, so they cannot disagree. Applying the suffix in
`__post_init__` runs exactly once per instance: there is no second step that
could apply it again, so the doubling is not merely fixed but unrepresentable.

Validating on construction means an out-of-range setting is reported before a
long training run starts, naming the field and the value, rather than surfacing
as an obscure failure inside Keras much later.

Artifact filenames are deliberately unchanged, so checkpoints written by the
previous implementation still load.

## Consequences

- Settings are validated early, with actionable messages.
- The save and load sides cannot disagree about a checkpoint path.
- `--nf` and `--no_dropout` no longer exist; passing them is now an error
  rather than silently ignored. Neither was ever read.
- `no_flip` is modelled as `flip: bool = True`. The CLI still accepts
  `--no-flip`, but the saved settings file records `flip` instead of
  `no_flip`.
- Adding a setting means touching both the dataclass and the parser, a small
  cost for keeping configuration independent of `argparse`.
