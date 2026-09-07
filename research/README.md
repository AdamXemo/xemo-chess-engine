---
status: current
updated: 2026-09-07
---

# Research

Working notes for a from-scratch chess engine project: dataset analysis, model
architecture research, engine prior art, and the experiment log. The intended
end products are a playable neural engine and a front end for it (web app or
Telegram bot).

This folder is text and small plots only. No raw data, no generated shards, no
notebook output.

## Reading order

| Document | Status | What it covers |
|---|---|---|
| [data/README.md](data/README.md) | current | Raw file inventory |
| [data/lichess-pgn-1800plus-5m.md](data/lichess-pgn-1800plus-5m.md) | current | 5M game PGN: schema, distributions, caveats |
| [data/lichess-evals-5m.md](data/lichess-evals-5m.md) | current | 5M position eval database: schema, depth quality |
| [data/cross-dataset-alignment.md](data/cross-dataset-alignment.md) | current | Why the two files do not join |
| [data/preprocessing.md](data/preprocessing.md) | draft | Conversion, splits, filters |
| [engines/](engines/) | draft | Prior art, search, strength measurement |
| [models/](models/) | draft | Encoding, architectures, training plan |
| [product/targets.md](product/targets.md) | draft | Deployment constraints |
| [experiments/README.md](experiments/README.md) | current | Run log |
| [decisions/README.md](decisions/README.md) | current | Decision records |

## Conventions

- Filenames are kebab-case. Experiments and decisions are zero-padded and
  numbered; numbers are never reused.
- Every document opens with a metadata block giving `status`, `updated`, and
  where relevant `source`.
- **Every number states its provenance.** `(full pass)` means the whole file was
  read. `(sampled, n=X)` means it was not. An unlabelled number is a bug.
- Every non-trivial number has a reproducing command, either inline or as a
  script in `data/_scripts/`.
- Findings and plans live in separate files. A dataset document describes what
  is in a file; `preprocessing.md` describes what to do about it.

## Backlog

Prioritised. Items marked **open question** should be resolved before committing
to a training pipeline, because the answer changes what the pipeline should be.

### Training targets the data supports directly

1. **Elo-conditioned policy network.** Position plus target rating to move,
   trained by behaviour cloning on the PGN. This is the most product-shaped use
   of the data: it gives an opponent that plays like a human of a chosen
   strength, which a strong engine at reduced depth does not. Rating labels and
   a wide rating spread are already present.
2. **Value head from the inline `%eval` subset.** Roughly 485k games carry
   server-side Stockfish evaluations on essentially every move, about 35M
   labelled positions, already aligned to real game positions. Convert centipawns to a
   win probability through a logistic scaling rather than regressing raw
   centipawns.
3. **Value head from the eval database.** Higher engine quality but a different
   position distribution, skewed toward endgames. Complementary to item 2, not a
   substitute. Filtering to depth 30 or more leaves roughly 2.5M positions.
4. **Soft policy targets from MultiPV.** The eval database averages 2.9 principal
   variations per analysis, giving a top-k move distribution instead of a
   one-hot label.
5. **Game-result value target.** The result tag as a win/draw/loss label across
   all ~363M positions: noisy, free, unlimited. Draws are only 4.6% of games, so
   a WDL head is badly imbalanced by default.

### Derived datasets and product features

6. **Puzzle mining.** Positions where the best line is mate, or where the
   centipawn gap between the first and second principal variation is large,
   yield a tactics set. Applied to the inline-eval games it yields puzzles from
   real human games together with the blunder that created them.
7. **Blunder detection and coaching.** Per-move centipawn loss from the inline
   evals trains a "how bad was this move" model, the backbone of any post-game
   analysis feature.
8. **Clock modelling.** Move clocks are present on 99.8% of games. Two payoffs: a
   bot that paces itself like a human instead of replying instantly, and an
   analysis of accuracy against remaining time.
9. **Opening explorer.** 498 ECO codes and 2,948 named openings crossed with
   result and rating band. Cheap, immediately useful in a front end, and needs
   no neural network.
10. **Rating estimation.** Predict player rating from a move sequence.

### Analyses to run before training anything

11. **Selection bias in the `%eval` subset** — *open question*. Lichess computes
    those evaluations when a user requests analysis, which plausibly biases the
    subset toward longer, closer, or more contentious games. Compare the 9.7%
    subset against the remainder on length, rating, result, and termination. If
    it is skewed, the best available value-training set is skewed with it.
12. **Centipawn loss against rating, and against remaining clock.** Validates the
    pipeline end to end and produces the most interesting plots in the project.
13. **Coverage by time control** — *open question*. Bullet is 46.5% of the file
    and time-scramble moves are noisy policy targets. Decide whether to
    down-weight or drop bullet before building shards, not after.
14. **Position duplication and transposition rate** across ~363M positions.
    Determines how much dedup is needed and how badly a position-level split
    would leak.

### Data engineering

15. Target format: packed bitboards or Parquet shards, written as a stream. A
    single file is not workable at this position count.
16. Split by game and by date, holding out the last day (2026-06-06, 106k games)
    rather than sampling positions at random.
17. Expose filter axes as flags: minimum rating, time-control class, termination,
    ply range, evaluation depth.
18. Sanity gates at conversion time, not per epoch: drop zero-ply games,
    unfinished results, and positions failing an explicit legality check.
