"""Shared path resolution for the data scripts."""
import os
import sys

PGN_NAME = "lichess_1800plus_5m.pgn"
EVALS_NAME = "lichess_evals_5m.jsonl"


def data_dir():
    """Raw data directory: first argv, else $CHESS_DATA, else <repo>/data/raw."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    env = os.environ.get("CHESS_DATA")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(repo, "data", "raw")


def pgn_path():
    return os.path.join(data_dir(), PGN_NAME)


def evals_path():
    return os.path.join(data_dir(), EVALS_NAME)


def quantiles(name, values):
    values = sorted(v for v in values if v is not None)
    n = len(values)
    print(
        "%s: n=%d min=%s p25=%s median=%s p75=%s p95=%s max=%s mean=%.1f"
        % (name, n, values[0], values[n // 4], values[n // 2],
           values[3 * n // 4], values[int(0.95 * n)], values[-1],
           sum(values) / n)
    )
