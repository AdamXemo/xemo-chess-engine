---
status: current
updated: 2026-09-07
source: lichess_evals_5m.jsonl (2,070,948,574 bytes)
---

# Lichess evaluation database

## Summary

Five million chess positions with Stockfish evaluations, one JSON object per
line. Each position carries one or more independent engine analyses, each with
its own depth, node count, and set of principal variations. Median depth is 25;
half the positions have an analysis at depth 30 or deeper. Every position is
distinct.

This is a **position database, not a game database**. Positions are
endgame-skewed relative to real play and carry no game context: no players, no
result, no move that was actually made. It is a value-target source and a
puzzle-mining source, not a policy-cloning source.

## Provenance

The Lichess community evaluation database export. Entries are contributed by
users running analysis, which is why depth and hardware vary so widely between
records and why the position distribution does not match played games.

## File facts

| Property | Value |
|---|---|
| Size | 2,070,948,574 bytes |
| Lines | 5,000,000 (full pass) |
| Distinct FENs | 5,000,000 — no duplicates (full pass) |
| Mean bytes per line | ~414 |

## Schema

Exactly two top-level keys, `fen` and `evals`, present on 100% of sampled lines.

```json
{"fen":"7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -",
 "evals":[
   {"depth":46,
    "knodes":4189972,
    "pvs":[{"cp":69,"line":"f7g7 e6e2 h8d8 e2d2 b7b5 c4b3 g7f6 d1e1 a6a5 a2a3"},
           {"cp":163,"line":"h8d8 d1e1 a6a5 a2a3 c6d7 e6e7 f7f6 e1f2 b7b5 c4b3"}]}]}
```

| Field | Type | Notes |
|---|---|---|
| `fen` | string | **Four fields only**: placement, side to move, castling, en passant |
| `evals[]` | array | Independent analyses of the same position, 1 to 8, mean 1.5 |
| `evals[].depth` | int | Search depth reached |
| `evals[].knodes` | int | Thousands of nodes searched |
| `evals[].pvs[]` | array | MultiPV lines, 1 to 49, mean 2.9 |
| `pvs[].cp` | int | Centipawns, **from the side to move**, clipped to ±20000 |
| `pvs[].mate` | int | Signed mate distance. Present instead of `cp`, never alongside |
| `pvs[].line` | string | **UCI** moves, space separated, at most 10 plies |

### Parsing gotchas

- **The FEN has four fields, not six.** Append `" 0 1"` before handing it to a
  board library or the parse fails.
- **`cp` is relative to the side to move**, so it must be negated for a
  White-relative target. The game PGN's `[%eval]` uses the opposite convention.
- **`line` is UCI, not SAN**, and does not match the notation used in the PGN.
- **`cp` and `mate` are mutually exclusive.** Code that reads `pv["cp"]`
  unconditionally will crash on 11.5% of entries.
- **`evals` is not sorted.** The deepest analysis is not necessarily first.

## Distributions

Sampled at ten evenly spaced file offsets, n=200,000 positions, unless noted.

### Position character

| Metric | min | p25 | median | p75 | p95 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pieces on board | 3 | 11 | 21 | 28 | 32 | 51 | 19.5 |
| Analyses per position | 1 | 1 | 1 | 2 | 3 | 8 | 1.5 |
| PVs per analysis | 1 | 1 | 3 | 5 | 5 | 49 | 2.9 |
| PV length (plies) | 1 | 10 | 10 | 10 | 10 | 10 | 9.8 |

Side to move is White on 51.4% of positions. Castling rights are fully gone on
73.4% and are the full `KQkq` on only 14.1%. A median of 21 pieces against 32 at
game start means this file is not a sample of positions that arise in play; it
is weighted toward endgames and toward positions someone thought worth
analysing.

### Search depth and effort

| Metric | min | p25 | median | p75 | p95 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Depth | 1 | 21 | 25 | 32 | 95 | 245 | 35.4 |
| knodes | 0 | 5,417 | 16,745 | 53,378 | 278,136 | 593,446,314 | 217,056 |

Best available depth per position, sampled n=500,000:

| Depth band | Positions | Share |
|---|---:|---:|
| under 10 | 3,385 | 0.68% |
| 10-19 | 45,150 | 9.03% |
| 20-29 | 201,941 | 40.39% |
| 30-39 | 114,068 | 22.81% |
| 40-49 | 40,362 | 8.07% |
| 50-59 | 20,558 | 4.11% |
| 60 and above | 74,536 | 14.91% |

90.3% of positions reach depth 20 or better and 49.9% reach depth 30 or better.
Filtering at depth 30 leaves roughly 2.5M positions. Node counts span five orders
of magnitude, so depth alone is not a reliable quality proxy across records
produced on different hardware.

### Scores

882,236 principal variations across the sample.

| | |
|---|---|
| Entries with `cp` | 780,503 (88.5%) |
| Entries with `mate` | 101,733 (11.5%) |
| Positions containing any mate score | 20.6% (sampled n=500,000) |

| Metric | min | p25 | median | p75 | p95 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Centipawns | −20,000 | −18 | 17 | 87 | 563 | 20,000 | 100.1 |
| Mate distance | −68 | 1 | 5 | 12 | 24 | 67 | 5.7 |

Mate distance concentrates hard at the short end — 1 through 8 accounts for most
of it — so mates here are mostly tactical shots rather than deep conversions.

## Quality and caveats

- **Endgame skew.** Median 21 pieces. Training a value head on this file alone
  produces a model calibrated on positions that rarely occur in the middlegame.
- **Heterogeneous quality.** Depth 1 and depth 245 records sit side by side.
  Filter before use.
- **Illegal positions.** 4 in 200,000 (0.002%) are not reachable in a legal game:
  three knights of one colour, or in one case 51 pieces on the board. They come
  from users analysing hand-set positions.
- **A FEN parser will not catch them.** `python-chess` accepted every sampled
  FEN, including the 51-piece one, and every position has exactly two kings.
  Legality gating needs an explicit piece census, not a successful parse.
- **No game context.** No result, no rating, no move actually played.
- **Score sign convention differs from the PGN**, which is an easy source of a
  silent sign error across the two datasets.

## Implications for use

Good for:

- Value-head training and distillation, after a depth filter.
- Soft policy targets from MultiPV ordering.
- Puzzle mining: a mate line, or a large centipawn gap between the first and
  second principal variation, marks a position with one clearly best move.
- Endgame-specific training, where the skew is an advantage rather than a
  problem.

Not usable for:

- Behaviour cloning. There is no human move here.
- Labelling positions from the game archive. See
  [cross-dataset-alignment.md](cross-dataset-alignment.md).

## Open questions

- How much does a value head trained here disagree with one trained on the PGN's
  inline evaluations, and where in the game does the disagreement concentrate?
- Is `knodes` a better quality filter than `depth` given the hardware variance?
- What fraction of the 5M positions survive a joint depth-30 and legality filter?

## Reproducing these numbers

```sh
python3 research/data/_scripts/evals_profile.py         # schema, scores, PV shape
python3 research/data/_scripts/evals_depth_profile.py   # best depth per position
python3 research/data/_scripts/evals_legality_check.py  # impossible positions

# Distinct FENs, full pass, about a minute.
LC_ALL=C sed 's/^{"fen":"//; s/",.*$//' data/raw/lichess_evals_5m.jsonl \
  | LC_ALL=C sort -u | wc -l
```
