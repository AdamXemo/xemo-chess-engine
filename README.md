# chess_ai

Building a chess engine from neural networks, starting from five million human
games and five million engine-evaluated positions. A hobby project; the intended
end products are a playable model and a front end for it.

## Status

Research phase. The datasets have been profiled; no model exists yet.

## Layout

```
data/raw/     source datasets, not tracked in git
research/     analysis, design notes, experiment log
```

Start at [research/README.md](research/README.md) for what the data contains and
what is planned. [research/data/](research/data/) documents both datasets in
detail, including the parsing traps in each.

## Data

The two source files are not in this repository. See
[data/raw/README.md](data/raw/README.md) for what belongs there and where it
comes from.

The scripts under `research/data/_scripts/` reproduce every number quoted in the
research notes. They need Python 3 and, for two of them, `python-chess`.

```sh
python3 research/data/_scripts/evals_profile.py
```
