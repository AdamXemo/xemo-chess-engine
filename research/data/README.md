---
status: current
updated: 2026-09-07
---

# Data inventory

Raw files live in [`data/raw/`](../../data/raw/) and are not tracked in version
control.

| File | Size | Records | Document |
|---|---:|---:|---|
| `lichess_1800plus_5m.pgn` | 12,606,157,374 B (11.7 GiB) | 5,000,000 games | [lichess-pgn-1800plus-5m.md](lichess-pgn-1800plus-5m.md) |
| `lichess_evals_5m.jsonl` | 2,070,948,574 B (1.93 GiB) | 5,000,000 positions | [lichess-evals-5m.md](lichess-evals-5m.md) |

Both are derived from Lichess open-data exports covering June 2026. They are two
independent sources, not a paired dataset: see
[cross-dataset-alignment.md](cross-dataset-alignment.md).

## Combined scale

- ~363M played moves in the PGN, from ~368M game positions.
- ~35M positions carrying an inline engine evaluation inside the PGN.
- 5M positions with standalone deep evaluations in the JSONL.

## Scripts

`_scripts/` holds the exact programs that produced every number in these
documents. Each takes the data directory as `$CHESS_DATA` or as its first
argument, defaulting to `data/raw/`.

| Script | Produces |
|---|---|
| `pgn_headers.awk` | Full-pass PGN header distributions |
| `pgn_movetext_sample.py` | Ply counts, tag presence, annotation coverage |
| `evals_profile.py` | Eval schema, score and PV distributions |
| `evals_legality_check.py` | Impossible positions in the eval database |
| `evals_depth_profile.py` | Best-available depth per position |
| `fen_overlap.py` | PGN/eval FEN intersection by ply index |

The full PGN pass takes roughly seven minutes. Prefilter header lines with
`grep '^\[[A-Za-z]'` before the `awk` stage; feeding the movetext through `awk` as well
takes about half an hour for the same result.
