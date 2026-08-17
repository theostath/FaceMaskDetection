# ADR 0004: Hold out a test set separate from the validation split

## Status

Accepted

## Context

Training loaded `dataroot/train` and split it 80/20 with `train_test_split`. The
20% was passed to `model.fit` as `validation_data` and then scored with
`classification_report` at the end of the run. That single number was the only
reported measure of quality.

It is not an unbiased one. The split is evaluated after every epoch, and its
loss curve is what a practitioner reads when deciding how many epochs to run,
what learning rate to use, or whether to keep a change. Those decisions leak
information from the split into the model, so its score drifts optimistic.

The original README documented a dataset layout containing both `train/` and
`test/` directories, but no code ever read `test/`. The intent was recorded and
never implemented.

## Decision

Read an optional `dataroot/test` as a held-out set, scored once after training
alongside the validation split. Report both, labelled. Rename the in-training
split to "validation" throughout: `TrainingData.val_images`, `val_labels`, and
`TrainConfig.val_size`.

Add `facemask-prepare` to produce the two directories from the flat, unsplit
layout the available datasets ship in.

## Rationale

Two numbers with different meanings should not share a name. Retaining
`test_size` for the fraction fed to `validation_data` while `test/` meant the
held-out directory would have been a durable source of confusion.

The held-out set is optional rather than required. A dataset with no `test/`
still trains, and training states plainly that the scores come from validation
alone -- silence there would let an optimistic number pass as a final one.

Splitting is a separate command rather than something training does implicitly.
A split performed on every run would either be non-deterministic, or duplicate
the seeding already in `train_test_split`, and either way it would hide from the
user which images the model is never allowed to see. Doing it once, visibly, on
disk makes the boundary inspectable.

The split is stratified and seeded so that a given source directory always
produces the same division, and copies rather than moves by default so a failed
run cannot destroy the download.

## Consequences

- Reported accuracy is measured on data the model has never seen. On the
  current dataset the validation split scores 99% and the held-out set 98%;
  the gap is the optimism this ADR exists to expose.
- Both reports are appended to `evaluation_<name>.txt`, so runs can be
  compared after the fact.
- `--test-size` and `--test_size` remain accepted as spellings of
  `--val-size`, so existing invocations keep working.
- Training on a dataset without `test/` behaves as before, minus the
  implication that its score is a final one.
- Roughly 20% of the data no longer contributes to fitting. For a frozen-base
  transfer learning task at this size that cost is small, and an honest
  estimate is worth more than the marginal images.
- See [ADR 0003](0003-typed-dataclass-configuration.md) for the configuration
  object these settings live on.
