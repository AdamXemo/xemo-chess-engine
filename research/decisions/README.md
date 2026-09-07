---
status: current
updated: 2026-09-07
---

# Decisions

Short records of choices that are expensive to reverse or easy to forget the
reason for: data format, move encoding, architecture family, deployment target.

One file per decision, `NNNN-short-slug.md`, numbered in order. A decision is
never edited after it is accepted; it is superseded by a later one that links
back to it.

## Log

| ID | Date | Decision | Status |
|---|---|---|---|
| — | — | — | none yet |

## Template

```markdown
---
status: accepted | superseded by NNNN
updated: YYYY-MM-DD
---

# NNNN — <decision>

## Context
What forced a choice. Link the measurements that constrain it.

## Options
What was considered, and the cost of each.

## Decision
What was chosen.

## Consequences
What this makes easy, what it makes hard, and what would have to be true to
revisit it.
```

## Candidates

- Training shard format and what fields it carries.
- Move vocabulary and position encoding.
- Whether bullet games are included in policy training.
- Deployment target, which sets the model size budget.
