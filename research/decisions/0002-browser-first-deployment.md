---
status: accepted
updated: 2026-09-07
---

# 0002 — Browser-first deployment

## Context

The deployment target sets the parameter budget, and the parameter budget
constrains every architecture choice after it. It could not stay open.

The candidates were a web app and a Telegram bot.

## Options

- **Web app, inference in the browser.** No per-game server cost, no cold
  start, works offline, and the better portfolio surface. Imposes a hard
  ceiling on model size, because the model is downloaded by every visitor.
- **Web app, inference on a server.** No size ceiling, but a per-request cost
  and cold starts on hosting that scales to zero.
- **Telegram bot.** Turn-based and latency-tolerant, so the cheapest to serve
  and the fastest route to a working opponent. Weaker as a portfolio artifact.

## Decision

Web app with inference in the browser.

## Consequences

- Model size is a deployment constraint before it is an accuracy lever. Both
  networks have to be small enough that a visitor will wait for the download.
- The policy network exports to ONNX; the search engine and its evaluation are
  expected to compile to WebAssembly.
- No inference server, no per-game cost, and the whole thing can be served as
  static files.
- A Telegram bot remains possible later against the same models, and would then
  need a server-side runtime that does not exist yet.
- Revisit if the size ceiling turns out to cost more strength than the hosting
  simplicity is worth.
