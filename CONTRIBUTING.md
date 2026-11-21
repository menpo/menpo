# Contributing to Menpo

Thank you for helping. Menpo is BSD-3 licensed and contributions are accepted under the same licence
(see `LICENSE.txt`).

## Issues

Please say which versions you use
(`python -c "import menpo, numpy, sys; print(menpo.__version__, numpy.__version__, sys.version)"`), what you
did, what you expected and what happened, with the smallest piece of code that shows it. A question about
how to do something is fine as an issue too.

## Getting set up

Menpo needs Python 3.10 or newer. With [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/menpo/menpo && cd menpo
uv sync --group dev          # menpo plus the test, lint and type-check tools
```

Some features need an optional dependency, which are extras: `--extra imageio` (video and GIFs; it also needs
`ffmpeg` on your `PATH`), `--extra opencv` and `--extra cyvlfeat`. The 3D viewing methods need `menpo3d`.
Tests that need one of these are skipped without it, and CI runs them once with them installed.

## Before you open a pull request

CI runs these on Linux, macOS and Windows with Python 3.10 and 3.14, and all of them must pass:

```
uv run pytest                  # the tests live next to the code, in menpo/**/test/
uv run ruff check .            # add --fix to apply the fixes it can
uv run ruff format --check .   # `uv run ruff format .` to fix
uv run mypy menpo
```

The documentation is built in CI too (an autodoc import that fails fails the build). To build it yourself,
install `docs/requirements.txt` and menpo, then run `sphinx-build -b html docs/source docs/_build`.

## What a good pull request looks like

- **One topic per pull request**, small enough to review, with a description that says what changed and
  why.
- **A test that fails without your change** and passes with it, next to the code in the relevant `test/`
  folder. A bug fix needs a test that reproduces the bug.
- **Docstrings** in numpydoc style for anything public, with cross-references written as `:map:`Name``
  (any class in `menpo/shape` shows the pattern).
- **A linear history.** We merge by rebase, so rebase your branch on `master` instead of merging `master`
  into it, and fold commits that only fix up earlier ones.
- **Mechanical changes** (formatting, renames) **go in their own pull request or commit**, not mixed with a
  real change.

## How the code is laid out

Each subpackage (`shape`, `image`, `transform`, `model`, `io`, ...) keeps its main classes in `base.py` (or
in modules named for what they hold), and its `__init__.py` re-exports the public names. The order of the
imports in an `__init__.py` can matter, because modules import each other: if `import menpo` starts failing
with a circular import after you reordered something, put the order back.

New file formats go in `menpo/io/input` (or `output`) and are registered by extension in that folder's
`extensions.py`.

## Changelog and releases

User-visible changes are listed per release in `docs/source/changelog.rst`, with their pull request numbers.
A release needs an entry there: **the release workflow refuses a final `vX.Y.Z` tag that has no
`X.Y.Z (YYYY/MM/DD)` heading in it.** Maintainers write the entry when they release; if you know what should
be said about your change (especially a breaking one, and how to migrate), put it in the pull request
description.

Releases are made by pushing a signed `vX.Y.Z` tag, as described under "Releasing" in the
[README](README.md).
