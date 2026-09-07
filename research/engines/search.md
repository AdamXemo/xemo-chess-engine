---
status: draft
updated: 2026-09-07
---

# Search

Empty. Intended contents: what search layer, if any, sits on top of the network.

## To cover

- Alpha-beta with a learned evaluation against MCTS with a policy prior:
  implementation cost, node throughput, and how each behaves with a weak
  evaluation.
- Policy-only play with no search, which is the cheapest option and the one that
  most resembles human play at a given rating.
- Move generation: use a library or write one. Writing one is a large share of
  the total project effort and is not on the critical path to a first model.
- Time management, which matters for a real opponent and is directly informed by
  the clock data in the game archive.
