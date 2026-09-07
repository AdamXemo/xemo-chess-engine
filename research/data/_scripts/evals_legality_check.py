"""How many evaluation-database positions are not legally reachable.

A FEN parser is not enough: python-chess accepts positions with impossible
piece counts, so the piece census has to be explicit.  Requires python-chess.
"""
import collections
import json

import chess

from _common import evals_path

SAMPLE = 200000


def main():
    n = over32 = unparseable = shown = 0
    counts = collections.Counter()
    with open(evals_path()) as fh:
        for line in fh:
            rec = json.loads(line)
            n += 1
            fen = rec["fen"]
            placement = fen.split()[0]
            pieces = sum(c.isalpha() for c in placement)
            counts[placement.count("K") + placement.count("k")] += 1
            if pieces > 32:
                over32 += 1
                if shown < 5:
                    print("  impossible (%d pieces): %s" % (pieces, fen))
                    shown += 1
            try:
                chess.Board(fen + " 0 1")
            except ValueError as exc:
                unparseable += 1
                if unparseable <= 3:
                    print("  unparseable: %s (%s)" % (fen, exc))
            if n >= SAMPLE:
                break

    print("sampled positions:", n)
    print("more than 32 pieces: %d (%.4f%%)" % (over32, 100 * over32 / n))
    print("rejected by the FEN parser: %d" % unparseable)
    print("king counts:", dict(counts))


if __name__ == "__main__":
    main()
