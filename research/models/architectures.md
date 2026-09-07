---
status: draft
updated: 2026-09-07
---

# Architectures

Empty. Intended contents: candidate networks and the budget each has to fit.

## To cover

- Residual convolutional tower with policy and value heads, the default for
  board games and the sane starting point.
- Transformer over squares as tokens, which is the more current approach and
  worth one honest comparison rather than an assumption.
- NNUE-style small evaluation network, if a classical search is used instead of
  a policy net.
- Multi-head design: policy, value, and possibly an auxiliary head predicting
  the game result separately from the engine evaluation, since the data provides
  both labels and they disagree in informative ways.

## Budget

Fill in once [product/targets.md](../product/targets.md) is decided. The
deployment target — browser inference, a bot backend, or a server with a GPU —
sets the parameter count, and the parameter count determines almost everything
else.

## Baselines to beat

- Random legal move.
- Material-count evaluation with shallow alpha-beta search.
- Move-match accuracy of "play the most common move in this position".
