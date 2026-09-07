---
status: current
updated: 2026-09-07
---

# Cross-dataset alignment

## Finding

**The game archive and the evaluation database do not join.** Their overlap is an
opening-book artifact: positions reached in the first few moves of almost every
game, and nothing else. Any plan that assumes the eval database can label
positions from the game archive is unworkable.

## Method

Every position from the first 2,000 games of the PGN was normalised to a
four-field FEN, matching the eval database's format, giving 135,678 distinct
positions. All 5,000,000 eval records were then streamed and tested for
membership. First occurrence ply index was recorded for each position.

## Result

10,326 of 135,678 positions matched, **7.61% overall**. Broken down by the ply at
which the position first appeared:

| Ply | Positions | Matched | Match rate |
|---|---:|---:|---:|
| 0-4 | 1,047 | 968 | 92.45% |
| 5-9 | 6,748 | 4,756 | 70.48% |
| 10-14 | 9,474 | 3,459 | 36.51% |
| 15-19 | 9,824 | 815 | 8.30% |
| 20-24 | 9,774 | 153 | 1.57% |
| 25-29 | 9,662 | 33 | 0.34% |
| 30-34 | 9,405 | 7 | 0.07% |
| 35-39 | 9,107 | 1 | 0.01% |
| 40+ | 70,637 | 134 | 0.19% |

Coverage collapses from 92% to under 2% within twenty plies. Past move 15 the
eval database has effectively nothing to say about positions that occur in real
games.

The small revival in the 40+ bucket is consistent with common simplified
endgames being independently analysed, and is worth confirming if endgame value
training becomes a priority.

## Why

The two files are sampled from different processes. The game archive is a census
of what was played. The eval database is a record of what users chose to analyse,
which is dominated by isolated positions of interest, heavily weighted toward
endgames — its median position has 21 pieces against 32 at game start. The only
place the two distributions coincide is the opening, where every game passes
through the same small set of positions.

## Consequence

Three separate signals, to be used separately rather than merged:

1. **Policy**, from the game archive: ~363M position-to-move pairs with rating,
   clock, and result context.
2. **Aligned value**, from the game archive's inline `[%eval]` annotations:
   ~35M labelled positions on the same distribution as the policy data. This,
   not the eval database, is the bridge between position and evaluation.
3. **Deep value**, from the eval database: 5M positions with far more search
   behind them, on a different and endgame-heavy distribution.

If a deeper value signal on game-distribution positions is wanted, the options
are to analyse sampled positions locally with an engine, or to distil a model
trained on signal 3 into signal 2's distribution. Joining the files is not one of
the options.

## Reproducing

```sh
python3 research/data/_scripts/fen_overlap.py
```

Runs in about a minute: one PGN prefix read plus one full pass over the eval
file.
