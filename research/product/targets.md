---
status: draft
updated: 2026-09-07
---

# Deployment target

A web app with inference in the browser. See
[../decisions/0002-browser-first-deployment.md](../decisions/0002-browser-first-deployment.md)
for the reasoning.

Everything ships to the client, so size is a deployment constraint before it is
an accuracy lever. This document is otherwise empty.

## To cover

- **The size ceiling**, which is the number the whole project hangs on: how
  large a download a visitor will actually wait for, split between the policy
  network and the evaluation network.
- Runtime: ONNX for the policy network, WebAssembly for the search and its
  evaluation. What each costs in bundle size and in throughput.
- Target move latency for each mode. The no-search opponent is one forward
  pass; the search engine is a time budget, and how many nodes fit in it is the
  question.
- Threading and whether the search can use more than one core in a browser.
- The front end itself: board, move input, mode selection, and how the two play
  modes are presented to someone who does not know what either is.

## Later

A Telegram bot against the same models. It needs a server-side runtime that
does not exist yet, and is not on the path to a first playable version.
