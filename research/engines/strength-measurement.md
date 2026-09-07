---
status: draft
updated: 2026-09-07
---

# Measuring strength

Empty. Intended contents: how any claim about the engine's strength gets made.

## To cover

- **Move-match accuracy** against held-out human games, split by rating band.
  Cheap, runs every epoch, and is the right metric for a human-imitation model —
  but it is not playing strength.
- **Head-to-head matches** via `cutechess-cli` against fixed opponents, with
  ratings computed by an established tool rather than by hand.
- **SPRT** for deciding whether a change is an improvement, so that A-versus-B
  claims are not made on fifty games.
- **Puzzle accuracy** on the Lichess puzzle set, as a tactics-specific check
  that catches value-head regressions.
- **Calibration** of the value head: predicted win probability against observed
  outcome, bucketed. A miscalibrated value head is a specific, fixable failure.

## Rule

No strength claim in `experiments/` without the match count and the error bars.
