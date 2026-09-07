---
status: accepted
updated: 2026-09-07
---

# 0003 — Alpha-beta search, in scope from the start

## Context

A network on its own plays on pattern recognition: one forward pass, no
verification. Its characteristic failure is playing a natural-looking move that
loses material to a short tactic it never checked. Nearly all of a modern
engine's strength lives in the search rather than the evaluation, so the
question is not whether search helps but whether this project builds one.

## Options

- **No search.** A policy network alone. Cheapest, and the most human-like
  opponent, since its mistakes stay human mistakes.
- **Alpha-beta.** Depth-first with pruning, needing a very fast evaluation.
  Well understood, and a tractable amount of code for a large strength gain.
- **MCTS.** Selective lookahead guided by a policy prior and a value estimate,
  in the AlphaZero style. Pairs naturally with a large policy and value
  network, but costs one network forward pass per node, which is the worst
  possible shape for browser inference.

## Decision

An optimised alpha-beta search, built alongside the networks rather than after
them.

## Consequences

- **Two networks, not one.** Alpha-beta is only strong when it searches a great
  many positions per second, so its evaluation must be small and cheap: an
  NNUE-style net, incrementally updatable, not a residual tower. The
  rating-conditioned policy network is a separate and larger model that runs
  once per move.
- The project therefore has two play modes: a human-like opponent with no
  search, and a strength-oriented engine with search. They share the data
  pipeline and the training codebase but not the network.
- MCTS is ruled out for now on the browser-inference constraint from
  [0002-browser-first-deployment.md](0002-browser-first-deployment.md).
- Search work brings the usual list with it — move ordering, transposition
  table, iterative deepening, quiescence — none of which is needed to get a
  first working version.
- Move generation performance now matters. A library is fine to start; if
  node throughput becomes the limit, that is the first thing to replace.
