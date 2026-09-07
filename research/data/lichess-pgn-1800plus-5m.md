---
status: current
updated: 2026-09-07
source: lichess_1800plus_5m.pgn (12,606,157,374 bytes)
---

# Lichess 1800+ game archive

## Summary

Five million rated Lichess games from the first six days of June 2026, filtered
so that **both players are rated 1800 or above**. Standard PGN with full Lichess
tag set, move clocks on nearly every move, and server-side engine evaluations on
a 9.7% subset. Chronologically ordered. Mean game length is 72.7 plies, giving
roughly 363M played moves.

The primary use is behaviour cloning: position to move, optionally conditioned on
rating. The inline-evaluation subset is a secondary, aligned source of value
targets.

## Provenance

Derived from the Lichess monthly database export for June 2026, truncated at
5,000,000 games. The rating filter applies to both sides: the minimum rating
observed across all 10,000,000 player slots is exactly 1800 (full pass), which
rules out a one-side-only filter.

Records are sorted by UTC start time.

| UTC date | Games |
|---|---:|
| 2026.06.01 | 974,456 |
| 2026.06.02 | 988,082 |
| 2026.06.03 | 982,775 |
| 2026.06.04 | 987,544 |
| 2026.06.05 | 961,000 |
| 2026.06.06 | 106,143 |

The last day is a partial truncation, not a drop in activity. It is the natural
holdout split.

## File facts

| Property | Value |
|---|---|
| Size | 12,606,157,374 bytes |
| Games | 5,000,000 (full pass) |
| Mean bytes per game | ~2,520 |
| Mean plies per game | 72.7 (sampled, n=383,747) |
| Ply distribution | min 0, p10 38, median 69, p90 112, max 445 (sampled, n=50,000) |
| Games with no moves | 0.21% (sampled, n=383,747) |
| Unique players | 280,876 (full pass) |

## Schema

Seven-tag roster plus Lichess extensions. Tag presence, sampled over 50,000
games:

| Tag | Presence |
|---|---:|
| `Event` `Site` `Date` `Round` `White` `Black` `Result` | 100% |
| `WhiteElo` `BlackElo` `ECO` `Opening` `TimeControl` | 100% |
| `UTCDate` `UTCTime` `Termination` | 100% |
| `WhiteRatingDiff` `BlackRatingDiff` | 99.8% |
| `WhiteTitle` `BlackTitle` | 2.0% each |
| `[%clk]` move annotation | 99.8% of games |
| `[%eval]` move annotation | 9.3% of games (9.7% on the wider sample) |

Example header and movetext:

```
[Event "Rated Blitz game"]
[Site "https://lichess.org/CimpAd9M"]
[White "Iminari"]
[Black "Lucas_Bortoli"]
[Result "0-1"]
[WhiteElo "1848"]
[BlackElo "1962"]
[ECO "D30"]
[Opening "Queen's Gambit Declined"]
[TimeControl "180+0"]
[Termination "Normal"]

1. d4 { [%eval 0.15] [%clk 0:03:00] } 1... d5 { [%eval 0.27] [%clk 0:03:00] }
```

### Parsing gotchas

- **Movetext is hard-wrapped mid-token.** A comment, a move number, or a clock
  value can be split across a newline. Line-oriented parsing will silently
  corrupt games. Use a real PGN tokeniser (`python-chess`'s `read_game`).
- **`Event` encodes the speed class, and for tournaments it also carries a
  URL**: `Rated Blitz tournament https://lichess.org/tournament/ToTiMeLS`.
  Classifying speed requires a prefix match, not equality. 3,435 distinct
  tournament and swiss event strings account for 400,567 games.
- **`[%eval]` is in pawns, signed from White's perspective**, and mate is written
  as `#N`, not as a number. It is absent on the final move of many games.

## Distributions

All tables in this section are full-pass unless noted.

### Result

| Result | Games | Share |
|---|---:|---:|
| 1-0 | 2,473,107 | 49.46% |
| 0-1 | 2,295,158 | 45.90% |
| 1/2-1/2 | 230,971 | 4.62% |
| `*` (unfinished) | 764 | 0.02% |

The 4.6% draw rate is a problem for any win/draw/loss head trained on game
results and needs class weighting or resampling.

### Speed class

| Class | Games | Share |
|---|---:|---:|
| Bullet | 2,325,359 | 46.51% |
| Blitz | 1,835,347 | 36.71% |
| Tournament / swiss (speed inside the event string) | 400,567 | 8.01% |
| Rapid | 392,412 | 7.85% |
| UltraBullet | 27,547 | 0.55% |
| Classical | 13,011 | 0.26% |
| Correspondence | 5,757 | 0.12% |

Nearly half the corpus is bullet. Move quality under a one-minute clock is not
the same distribution as considered play, and time-scramble moves are noisy
policy targets.

### Time control

387 distinct values.

| Control | Games | Share |
|---|---:|---:|
| 60+0 | 2,061,340 | 41.23% |
| 180+0 | 1,095,427 | 21.91% |
| 300+0 | 407,095 | 8.14% |
| 180+2 | 405,226 | 8.10% |
| 600+0 | 324,951 | 6.50% |
| 120+1 | 267,346 | 5.35% |
| 30+0 | 95,450 | 1.91% |
| 300+3 | 92,984 | 1.86% |
| 600+5 | 53,434 | 1.07% |
| 15+0 | 47,197 | 0.94% |

### Rating

Mean 2069.4, range 1800 to 3994, over 10,000,000 player slots.

| Band | Slots | Share |
|---|---:|---:|
| 1800-1899 | 2,236,300 | 22.36% |
| 1900-1999 | 2,313,177 | 23.13% |
| 2000-2099 | 1,854,903 | 18.55% |
| 2100-2199 | 1,326,752 | 13.27% |
| 2200-2299 | 907,918 | 9.08% |
| 2300-2399 | 587,604 | 5.88% |
| 2400-2499 | 346,200 | 3.46% |
| 2500-2599 | 192,607 | 1.93% |
| 2600-2699 | 107,637 | 1.08% |
| 2700-2799 | 58,198 | 0.58% |
| 2800-2899 | 31,082 | 0.31% |
| 2900-2999 | 17,820 | 0.18% |
| 3000-3099 | 13,040 | 0.13% |
| 3100+ | 6,972 | 0.07% |

45.5% of player slots are in the 1800-1999 band, 7.7% are 2400 or above. Rating
conditioning is well supported through the low 2000s and thin above 2600.

### Termination

| Termination | Games | Share |
|---|---:|---:|
| Normal | 3,189,399 | 63.79% |
| Time forfeit | 1,802,075 | 36.04% |
| Abandoned | 7,051 | 0.14% |
| Unterminated | 757 | 0.02% |
| Insufficient material | 648 | 0.01% |
| Rules infraction | 70 | <0.01% |

Over a third of games end on the clock. The final position of a time-forfeit
game says nothing about who was winning, so game-result value labels are
systematically wrong on a meaningful slice unless termination is used as a
filter or a feature.

### Titled players and bots

175,473 title tags across 10,000,000 player slots (1.75%).

| Title | Count | | Title | Count |
|---|---:|---|---|---:|
| FM | 49,835 | | WFM | 3,178 |
| BOT | 39,888 | | WIM | 1,449 |
| CM | 28,482 | | WCM | 1,348 |
| IM | 21,706 | | LM | 478 |
| NM | 19,385 | | WGM | 296 |
| GM | 9,288 | | WNM | 140 |

`BOT` is the important one: 39,888 player slots are engines, not humans. Behaviour
cloning that targets human play should exclude them.

### Openings

498 ECO codes, 2,948 named openings.

| Opening | Games |
|---|---:|
| Queen's Pawn Game | 117,058 |
| Caro-Kann Defense | 86,868 |
| Modern Defense | 79,520 |
| Pirc Defense | 78,566 |
| Van't Kruijs Opening | 75,436 |
| French Defense: Exchange Variation | 72,655 |
| Horwitz Defense | 67,684 |
| Benoni Defense: Old Benoni | 66,610 |
| Caro-Kann Defense: Exchange Variation | 52,544 |
| Scandinavian Defense | 50,532 |

The names come from Lichess's opening classifier, which labels the deepest
matching position, so broad names such as "Queen's Pawn Game" collect games that
left book early.

## The inline evaluation subset

9.7% of games carry `[%eval]` on their moves, roughly 485,000 games. Sampling
gives 7.00 evaluation annotations per game averaged across the whole corpus,
which scales to **about 35M labelled positions**. Coverage inside an annotated
game is essentially complete: 99.4% of plies carry an evaluation.

This is the only value-labelled data in the project that sits on the same
position distribution as the policy data, which makes it the natural training
set for a value head. See open question 11 in the [backlog](../README.md) before
relying on it: the subset exists because a user requested analysis, and that
selection may not be neutral.

## Quality and caveats

- 764 games have result `*` and no outcome.
- ~0.21% of games have zero moves.
- 36% of games end on time; the final position is not an outcome signal there.
- 39,888 player slots are bots.
- Bullet dominates the corpus at 46.5%.
- Six days of data means opening fashion, and the player population, are a
  snapshot rather than a sample across time.

## Open questions

- Is the `%eval` subset representative? (backlog 11)
- How does centipawn loss vary with rating and with remaining clock? (backlog 12)
- How much position duplication is there across ~363M positions? (backlog 14)
- Do bullet games degrade a policy net, or does volume outweigh noise? (backlog 13)

## Reproducing these numbers

```sh
# Full-pass header distributions, about seven minutes.
LC_ALL=C grep -a '^\[[A-Za-z]' data/raw/lichess_1800plus_5m.pgn \
  | LC_ALL=C awk -f research/data/_scripts/pgn_headers.awk

# Ply counts, tag presence, annotation coverage.
python3 research/data/_scripts/pgn_movetext_sample.py
```
