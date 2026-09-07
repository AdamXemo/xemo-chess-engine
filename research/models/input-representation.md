---
status: draft
updated: 2026-09-07
---

# Input representation

Empty. Intended contents: how a position becomes a tensor, and how a move
becomes a label.

## To cover

### Position encoding

- Plane stack, in the AlphaZero style: one binary plane per piece type per
  colour, plus planes for side to move, castling rights, en passant, and
  repetition. Simple, well understood, wasteful.
- Compact encodings for a small network, where the plane stack is the dominant
  cost.
- Whether to orient the board from the side to move, which halves what the
  network has to learn about colour symmetry.
- History planes: whether the previous positions are worth their cost. The data
  supports them; the target inference budget may not.

### Move encoding

- A fixed move vocabulary (from-square, to-square, promotion) sized so that every
  legal move maps to exactly one index, with illegal moves masked at inference.
- The eval database's principal variations are in **UCI**, the game archive's
  moves are in **SAN**. Both must reach the same vocabulary.

### Conditioning inputs

- Target rating, for the rating-conditioned policy. Bucketed or continuous.
- Time control class and remaining clock, if human-like pacing is a goal.

## Constraint

Whatever is chosen here is baked into the shards at conversion time, and
reconverting is a full 12 GB pass. Decide before writing the pipeline.
