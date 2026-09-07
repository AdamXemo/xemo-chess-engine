---
status: draft
updated: 2026-09-07
---

# Preprocessing plan

Not yet implemented. This file records intent; measured facts belong in the
dataset documents.

## Target format

Open. Candidates: packed bitboard records in a flat binary file, or Parquet
shards. Constraints that any choice has to meet:

- ~363M training positions. A single file is not workable; write shards.
- Random access for shuffling without loading a shard into memory.
- Conversion is a one-off cost of a full 12 GB PGN pass, so everything needed
  downstream must be extracted in that single pass.

Per-position fields to carry: position encoding, played move, side to move,
ply index, game id, both ratings, time control class, clock remaining, game
result, termination, and inline evaluation where present.

## Splits

Split **by game and by date**, never by position. Positions from one game are
near-duplicates of each other, and a position-level split leaks the test set into
training.

- Test: 2026-06-06, 106,143 games. Already a natural truncation boundary.
- Validation: a game-level sample from 2026-06-05.
- Train: everything before.

## Filters

Expose as flags rather than baking in, so ablations are cheap:

| Filter | Default | Reason |
|---|---|---|
| Drop `BOT` players | on | 39,888 engine player slots |
| Drop zero-ply games | on | ~0.21% of games |
| Drop `*` results | on | 764 games, no outcome |
| Minimum rating | none | The file is already 1800+ |
| Time-control class | open question | Bullet is 46.5%, quality is unclear |
| Termination | keep, expose as feature | 36% end on time |
| Eval depth (eval DB) | 30 | Leaves ~2.5M of 5M positions |
| Legality check | on | 0.002% of eval FENs are illegal |

## Label construction

- Policy: the played move, as an index into a fixed move vocabulary.
- Value from result: win/draw/loss, with class weighting for the 4.6% draw rate.
- Value from evaluation: centipawns through a logistic scaling to a win
  probability. Mate scores need an explicit mapping, not a large centipawn value.
- Watch the sign conventions: the PGN's `[%eval]` is White-relative, the eval
  database's `cp` is side-to-move-relative.

## Validation gates

Run once at conversion, not per epoch: FEN legality, move legality against the
reconstructed board, monotonic ply indices, and a checksum of game count per
shard.
