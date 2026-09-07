---
status: draft
updated: 2026-09-07
---

# Engine prior art

Empty. Intended contents: what each reference engine actually does, and which
ideas are worth borrowing at hobby scale.

## To cover

- **Stockfish / NNUE** — hand-written alpha-beta search with a small
  incrementally-updated network as the evaluation. The closest prior art to the
  search side of this project; read how the evaluation is kept cheap enough to
  call millions of times.
- **Leela Chess Zero** — policy and value network driving MCTS, trained by
  self-play. The relevant question is what the network looks like without the
  self-play budget behind it.
- **Maia** — behaviour cloning on human games, rating-conditioned. The closest
  prior art to the policy network; read what it gets right and where it fails.
- **Small engines** (Sunfish and similar) — useful as a correctness baseline and
  as a sparring partner during development.

## Questions

- Where is the strength/compute frontier for a network small enough to run in a
  browser?
- How much strength does the search contribute over the same evaluation with no
  search, measured rather than assumed?
