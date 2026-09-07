# Raw data

Immutable source files. Nothing in this directory is tracked in git or modified
in place; derived data is written elsewhere.

| File | Size | Records |
|---|---:|---:|
| `lichess_1800plus_5m.pgn` | 12,606,157,374 B | 5,000,000 games |
| `lichess_evals_5m.jsonl` | 2,070,948,574 B | 5,000,000 positions |

Both are derived from the [Lichess open database](https://database.lichess.org/)
for June 2026: the game archive filtered to games where both players are rated
1800 or above and truncated at five million games, and the community evaluation
database truncated at five million positions.

For what is actually in them, see
[research/data/](../../research/data/).

## Expected layout

```
data/
  raw/        source files, read-only
  interim/    intermediate conversion output
  processed/  training shards
```

Scripts locate this directory through `$CHESS_DATA`, a first positional
argument, or by walking up from their own location. Set `CHESS_DATA` if the
files live elsewhere:

```sh
export CHESS_DATA=/path/to/data/raw
```
