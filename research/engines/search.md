---
status: draft
updated: 2026-09-07
---

# Search

An optimised alpha-beta search is in scope from the start. MCTS is ruled out on
the browser-inference constraint. See
[../decisions/0003-alpha-beta-search.md](../decisions/0003-alpha-beta-search.md)
for the reasoning and the consequences.

This document is otherwise empty. Intended contents: how the search is actually
built.

## To cover

- Minimum viable version: negamax with alpha-beta pruning and a fixed depth.
  Everything below is an optimisation on top of a search that already works.
- **Move ordering**, which matters more than anything else here. Alpha-beta's
  pruning depends on trying good moves first, and the policy network is an
  obvious source of an ordering prior.
- Transposition table, iterative deepening, quiescence search, null-move
  pruning, late move reductions. Add them one at a time, each measured.
- Node throughput: where the time actually goes, and whether the library move
  generator becomes the bottleneck.
- Time management, informed by the clock data in the game archive.

## Constraint

The evaluation has to be cheap enough to call millions of times. That is what
makes this an NNUE-style network rather than the policy network, and it is the
constraint every other choice here answers to.

## Rule

Every optimisation is a separate experiment with a measured result. "Should be
faster" is not a result. See
[strength-measurement.md](strength-measurement.md).
