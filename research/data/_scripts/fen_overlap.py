"""Overlap between game-archive positions and the evaluation database.

Reports the match rate by the ply at which a position first appeared, which is
what shows that the overlap is an opening-book artifact rather than usable
coverage.  Requires python-chess.
"""
import collections

import chess
import chess.pgn

from _common import evals_path, pgn_path

GAMES = 2000
BUCKET = 5
LAST_BUCKET = 40


def key(board):
    """Four-field FEN, matching the evaluation database's format."""
    return " ".join(board.fen().split()[:4])


def main():
    first_ply = {}
    games = 0
    with open(pgn_path(), errors="replace") as fh:
        while games < GAMES:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            games += 1
            board = game.board()
            first_ply.setdefault(key(board), 0)
            for ply, move in enumerate(game.mainline_moves(), start=1):
                board.push(move)
                first_ply.setdefault(key(board), ply)

    total = collections.Counter()
    for ply in first_ply.values():
        total[min(ply // BUCKET * BUCKET, LAST_BUCKET)] += 1

    hits = collections.Counter()
    matched = 0
    with open(evals_path()) as fh:
        for line in fh:
            # Cheaper than json.loads: "fen" is always the first key.
            fen = line[8:line.index('"', 8)]
            ply = first_ply.get(fen)
            if ply is not None:
                hits[min(ply // BUCKET * BUCKET, LAST_BUCKET)] += 1
                matched += 1

    print("games %d, unique positions %d, matched %d (%.2f%%)"
          % (games, len(first_ply), matched, 100 * matched / len(first_ply)))
    print("    ply | positions | matched | match rate")
    for band in sorted(total):
        label = "%d-%d" % (band, band + BUCKET - 1) if band < LAST_BUCKET else "40+"
        print("%7s | %9d | %7d | %5.2f%%"
              % (label, total[band], hits[band], 100 * hits[band] / total[band]))


if __name__ == "__main__":
    main()
