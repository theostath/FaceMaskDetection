# ADR 0001: Package the project as an installable distribution with a src layout

## Status

Accepted

## Context

The project shipped as loose top-level scripts (`train.py`, `test.py`) importing
sibling directories (`options/`, `models/`, `data/`). That arrangement only
works when the interpreter's working directory is the repository root, so the
code could be run but not installed, imported from elsewhere, or tested against
what a user would actually get.

Dependencies were declared twice, in `requirements.txt` and `environment.yml`,
and the two had drifted apart: `requirements.txt` omitted `tensorflow` entirely
despite every module importing it, and pinned `argparse==1.4.0`, a PyPI backport
that shadows the standard library module. `environment.yml` pinned Windows-only
conda builds while the README advertised Linux support.

There was no packaging metadata, no entry points, and no way to state which
Python versions were supported.

## Decision

Package the project as a distribution named `facemask`, declared entirely in
`pyproject.toml` with a hatchling backend, and move the code under
`src/facemask/`. Expose two console entry points, `facemask-train` and
`facemask-detect`. Delete `requirements.txt` and `environment.yml`.

## Rationale

A src layout cannot be imported by accident from the repository root, so tests
and CI exercise the installed package rather than the working copy. This is what
catches a missing module or a broken entry point before a user does.

`pyproject.toml` gives one place for dependencies, supported Python versions,
entry points, and the ruff and pytest configuration, removing the drift that let
the two previous manifests disagree.

A committed lockfile was considered and rejected: resolution on Windows pulls
`tensorflow-intel`, which does not exist on Linux, so a lockfile generated on
one platform would be wrong on the other. Reproducible pinning would need one
lockfile per platform, generated in CI, which is more machinery than this
project warrants.

Console entry points replace `python train.py`, which depended on the working
directory. Both accept underscored option spellings alongside hyphenated ones,
so previously documented invocations keep working.

## Consequences

- `pip install -e .` produces a working environment; previously it could not.
- Both commands run from any directory.
- Tests import the package the way users do.
- Contributors must install the package before running anything; running a
  script from the repository root no longer works.
- Exact dependency versions are not pinned, so a future release of a
  transitive dependency could break a fresh install. CI running on every push
  is what surfaces that.
- See [ADR 0002](0002-pin-tensorflow-2-15.md) for the version constraints this
  packaging declares.
