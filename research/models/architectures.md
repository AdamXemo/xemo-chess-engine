---
status: draft
updated: 2026-09-07
---

# Architectures

Two networks, because the two play modes need different things from them. See
[../decisions/0003-alpha-beta-search.md](../decisions/0003-alpha-beta-search.md).

| | Policy network | Evaluation network |
|---|---|---|
| Runs | once per move | millions of times per move |
| Feeds | the human-like opponent | the alpha-beta search |
| Shape | residual tower or transformer | NNUE-style, incrementally updatable |
| Trained on | played moves, rating-conditioned | engine evaluations |

This document is otherwise empty. Intended contents: the candidates for each,
and the budget they have to fit.

## To cover

### Policy network

- Residual convolutional tower with a policy head, the default for board games
  and the sane starting point.
- Transformer over squares as tokens, worth one honest comparison rather than
  an assumption.
- Whether a value head is worth carrying on the same trunk, given that the
  search has its own evaluation.

### Evaluation network

- NNUE: what "incrementally updatable" actually requires of the input encoding,
  and why that constrains the feature set.
- How small it has to be to sustain a useful node rate in WebAssembly.
- Quantisation, which is standard practice here rather than an optimisation.

## Budget

Set by [../product/targets.md](../product/targets.md): everything is
downloaded by the visitor. Neither network's parameter count can be chosen
until the size ceiling is.

## Baselines to beat

- Random legal move.
- Material-count evaluation with the same alpha-beta search. This is the honest
  baseline for the evaluation network — it isolates what the network adds from
  what the search adds.
- Move-match accuracy of "play the most common move in this position", for the
  policy network.
