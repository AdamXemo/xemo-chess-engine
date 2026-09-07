---
status: draft
updated: 2026-09-07
---

# Deployment targets

Empty. Intended contents: where the model runs, and what that costs it.

The candidates are a web front end and a Telegram bot. They impose very
different constraints and the choice cannot be deferred past architecture
selection, since it sets the parameter budget.

## To cover

### Browser inference

- Model ships to the client; no per-game server cost, no cold start, works
  offline.
- Hard size ceiling: a model users will actually wait to download.
- Runtime options and what each costs in size and latency.

### Server inference

- No size ceiling, one implementation, immediate updates.
- Per-request cost and cold starts on hosting that scales to zero.
- Concurrency: how many simultaneous games one instance can serve.

### Telegram bot specifically

- Turn-based and latency-tolerant, which makes it the cheaper of the two to
  serve and the better first target.
- Board rendering: image per move, or inline keyboard.
- State per game and where it lives.

## Decide first

Target move latency and acceptable model size. Everything in
[models/architectures.md](../models/architectures.md) follows from those two
numbers.
