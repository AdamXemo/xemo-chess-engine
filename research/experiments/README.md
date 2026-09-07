---
status: current
updated: 2026-09-07
---

# Experiments

One file per run, `NNNN-short-slug.md`, numbered in order and never renumbered.
A run gets a file when it starts, not when it succeeds. Failed runs stay, with
the reason.

## Results

| ID | Date | What changed | Result | Status |
|---|---|---|---|---|
| — | — | — | — | none yet |

## Template

```markdown
---
status: running | done | abandoned
updated: YYYY-MM-DD
---

# NNNN — <short title>

## Question
One sentence. What would a negative result rule out?

## Setup
Data (which shards, which filters), model, hyperparameters, hardware, commit.

## Result
The metric, with the comparison it is against. Plots if they help.

## Reading
What this actually shows, and what it does not.

## Next
The one experiment this suggests.
```

## Rules

- Change one thing per run, or accept that the result attributes to nothing.
- Record the commit hash. A result that cannot be traced to code is an anecdote.
- No strength claim without a match count and error bars. See
  [../engines/strength-measurement.md](../engines/strength-measurement.md).
- Negative results get written up. They are most of the information.
