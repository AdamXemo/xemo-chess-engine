---
status: draft
updated: 2026-09-07
---

# Engine prior art

Empty. Intended contents: what each reference engine actually does, and which
ideas are worth borrowing at hobby scale.

## To cover

- **Stockfish / NNUE** — hand-written alpha-beta search with a small
  incrementally-updated network as the evaluation. The relevant question is how
  much of its strength comes from search rather than evaluation.
- **Leela Chess Zero** — policy and value network driving MCTS, trained by
  self-play. The relevant question is what the network looks like without the
  self-play budget behind it.
- **Maia** — behaviour cloning on human games, rating-conditioned. The closest
  prior art to the primary plan in this project; read what it gets right and
  where it fails.
- **Small engines** (Sunfish and similar) — useful as a correctness baseline and
  as a sparring partner during development.

## Questions

- Where is the strength/compute frontier for a network small enough to run in a
  browser or a bot backend?
- Is a search layer needed at all for the target use, or does a strong policy
  network alone play acceptably?
