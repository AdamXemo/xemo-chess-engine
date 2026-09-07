---
status: draft
updated: 2026-09-07
---

# Training plan

Empty. Intended contents: targets, losses, and schedule.

## Targets available

From [data/](../data/):

| Signal | Source | Volume |
|---|---|---|
| Played move | Game archive | ~363M |
| Game result | Game archive | ~363M, 4.6% draws |
| Engine evaluation, aligned | Game archive `[%eval]` | ~35M |
| Engine evaluation, deep | Eval database | 5M, ~2.5M at depth 30+ |
| MultiPV move ordering | Eval database | 5M positions, mean 2.9 lines |

## To decide

- Loss weighting between policy and value.
- Whether the value head trains on game results, on engine evaluations, or on
  both with separate heads.
- Centipawn-to-win-probability mapping, including how mate scores enter it.
- Draw class imbalance: weighting, resampling, or a two-head split.
- Rating conditioning: an input feature, or separate models per band.
- Curriculum: whether to pretrain on the full corpus and fine-tune on a filtered
  high-rating subset.

## Order of work

1. Overfit a tiny model on a few thousand positions. If that does not work,
   nothing downstream will.
2. Policy only, no value head, no conditioning. Establish move-match accuracy.
3. Add the value head on the aligned inline evaluations.
4. Add rating conditioning and measure whether play at a requested rating
   actually tracks the request.
5. Only then consider search, self-play, or a larger model.
