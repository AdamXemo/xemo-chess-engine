# chess_ai

A chess engine built from neural networks trained on human games. Personal
portfolio project, not a product: correctness and legibility matter more than
speed of delivery, and the repository is meant to be read by other people.

The end goal is a model strong enough to be worth playing against, served
through a web app where inference runs in the browser.

## Current state

Research phase. The two source datasets have been profiled; **no engine code
exists yet**. The next task is the data pipeline (see Next task below).

Start with [research/README.md](research/README.md). It carries the prioritised
backlog and the open questions. Do not restate its contents here or in code
comments; link to it.

## What is being built

Two ways to play, sharing one data pipeline and one training codebase.

1. **Rating-conditioned policy network** — position and target rating to move,
   trained by behaviour cloning on 5M human games. One forward pass, no search.
   This is the human-like opponent: it should play like a 1900, mistakes
   included.
2. **Evaluation network with alpha-beta search** — a small, fast evaluation
   driving an optimised alpha-beta search. This is where playing strength comes
   from, and it is in scope from the start.
3. **Web front end** — board, move input, both modes, inference in the browser.

These need **different networks**, which is the main architectural consequence
to keep in mind. Alpha-beta is only strong when it can search a great many
positions per second, so its evaluation has to be small and cheap — an
NNUE-style net, not a residual tower. The policy network is a separate,
larger model that runs once per move. See
[research/decisions/0003-alpha-beta-search.md](research/decisions/0003-alpha-beta-search.md).

## Stack

- **Python + PyTorch.** `python-chess` for board logic and PGN parsing.
- **uv** for environments and dependencies. Never call `pip` or activate a venv
  by hand.
- **ruff** for formatting and linting.
- Export path to the browser is ONNX for the policy network. The search engine
  and its evaluation are expected to compile to WebAssembly.
- Everything ships to the client, so model size is a deployment constraint
  first and an accuracy lever second.

```sh
uv sync                          # reproduce the environment
uv add torch                     # add a dependency, updates uv.lock
uv run scripts/train.py          # run inside the project environment
uv run ruff format . && uv run ruff check .
```

## Layout

```
data/raw/     source datasets, never tracked in git
research/     analysis, design notes, decisions, experiment log
```

Application code does not exist yet. When it does, it goes in a top-level
package, with entry points under `scripts/`.

## Data

The two source files live in `data/raw/` and total 14 GB. They are ignored by
git and must never be committed, in whole or in part. The same applies to
derived shards, model weights, and checkpoints.

Scripts locate the data through `$CHESS_DATA`, a first positional argument, or
`<repo>/data/raw` — see `research/data/_scripts/_common.py`. Follow that
pattern in new scripts rather than hardcoding paths.

Read [research/data/](research/data/) before touching either file. Both have
parsing traps that are cheap to hit and expensive to notice:

- PGN movetext is hard-wrapped mid-token, so a line can begin with `[%clk`. A
  `startswith("[")` header test silently eats moves.
- Evaluation-database FENs have four fields, not six.
- Centipawn scores are side-to-move-relative in the eval database and
  White-relative in the PGN.
- `cp` and `mate` are mutually exclusive in the eval database.

## Next task

Build the conversion pipeline described in
[research/data/preprocessing.md](research/data/preprocessing.md): 12 GB of PGN
into training shards. Conversion is a single expensive pass, so the encoding
and the fields carried per position must be settled before writing it — record
those as decisions, not as comments in the converter.

Two open questions in the backlog change the pipeline's output and should be
answered first: whether the `%eval` subset is representative, and whether
bullet games belong in policy training.

## Conventions

### Numbers

Every quoted measurement must be reproducible by a committed script and
labelled with its provenance: `(full pass)` or `(sampled, n=X)`. An unlabelled
number is a bug. If a figure cannot be reproduced, remove it rather than
leaving it in prose.

### Code

Write it the way a careful colleague would, and match the surrounding file.

- No decorative comment banners, no `# ---- section ----` separators, no ASCII
  art, no emoji.
- Comments explain *why*, and only where the reason is not evident. Do not
  narrate what the next line does.
- No "Note:", "Important:", or summary comments restating the code.
- Type hints on function signatures. Docstrings on modules and non-obvious
  functions, one or two lines, not a template.
- Prefer a plain function to a class, and standard library to a dependency.
- No defensive `try`/`except` around things that should not fail. Let a bug
  crash where it happens.

### Documents

`research/` is prose, not slide bullets. Same rules: no banners, no emoji, no
filler. State what was measured and what it means. Mark open questions as open
rather than guessing at an answer.

Do not add README files, changelogs, or summary documents that were not asked
for.

### Commits

```
type: brief lowercase subject

Body only when the reason is not obvious from the diff. Wrap at 79.
```

- Types: `add`, `fix`, `chore`, `docs`, `refactor`, `test`, `perf`.
- Subject is lowercase and imperative, under about 70 characters.
- One logical change per commit.
- No attribution or co-author trailers.
- Never commit data files, weights, or `__pycache__`.

### Decisions and experiments

Choices that are expensive to reverse — data format, move encoding,
architecture family — get a numbered record in
[research/decisions/](research/decisions/) before the code that depends on
them. Training runs get a numbered record in
[research/experiments/](research/experiments/), written when the run starts and
kept even when it fails.

No claim about playing strength without a match count and error bars. Move
match accuracy against held-out human games is a useful metric but is not
playing strength; do not report it as such.
