---
status: accepted
updated: 2026-09-07
---

# 0001 — Python, PyTorch, and uv

## Context

The project needs a training stack before the data pipeline is written, since
the shard format is chosen to suit whatever reads it.

## Options

- **Python with PyTorch.** Largest ecosystem for this kind of work,
  `python-chess` already in use by the analysis scripts, straightforward ONNX
  export.
- **Python with JAX.** Faster on accelerators and cleaner functionally, but a
  thinner chess ecosystem and more friction reaching a deployment target.
- **Training in Python, engine in a compiled language.** More work, and the
  stronger portfolio piece, but premature before there is anything to serve.

For dependencies: uv, pip with venv, or conda.

## Decision

Python with PyTorch, dependencies managed by uv, formatting and linting by
ruff.

The compiled-engine option is not rejected, only deferred: see
[0003-alpha-beta-search.md](0003-alpha-beta-search.md), which brings part of it
back for the search layer.

## Consequences

- `uv.lock` is committed, so a reader can reproduce the environment exactly.
  This matters for a repository meant to be read by other people.
- Anyone cloning installs uv first. Acceptable — it is a single binary.
- PyTorch CUDA builds need an explicit index configured in `pyproject.toml`
  rather than a plain `pip install torch`.
- `uv run` is the only supported way to execute project code. No manual venv
  activation, no bare `python`.
