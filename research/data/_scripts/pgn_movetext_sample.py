"""Ply counts, tag presence, and annotation coverage in the game archive.

Two samples, because they answer different questions:
  - a head sample of 50,000 games, for tag presence and ply quantiles
  - a strided sample across seven file offsets, for corpus-wide means, since
    the file is time-ordered and any single region is not representative
"""
import collections
import subprocess

from _common import pgn_path

HEAD_GAMES = 50000
OFFSETS_MB = [500, 2000, 4000, 6000, 8000, 10000, 12000]
CHUNK_MB = 150


def head_sample(path):
    plies, tags, seen = [], collections.Counter(), set()
    games, cur, read = 0, 0, 0
    with open(path, errors="replace") as fh:
        for line in fh:
            read += len(line)
            # A header line is "[Tag ...".  Wrapped movetext can also begin
            # with "[", as in "[%clk 0:02:59] } 4. g3", so the second character
            # has to be checked or clock annotations are read as tags and the
            # moves on those lines are lost.
            if line.startswith("[") and line[1:2].isalpha():
                tag = line[1:].split(" ", 1)[0]
                if tag == "Event":
                    if games:
                        plies.append(cur)
                        for k in seen:
                            tags[k] += 1
                    seen, cur, games = set(), 0, games + 1
                    if games > HEAD_GAMES:
                        break
                seen.add(tag)
            else:
                cur += line.count("%clk")
                if "%eval" in line:
                    seen.add("%eval")
                if "%clk" in line:
                    seen.add("%clk")

    plies.sort()
    n = len(plies)
    print("head sample: %d games, %.1f MB, %.0f bytes/game"
          % (games - 1, read / 1e6, read / games))
    print("plies: min %d p10 %d median %d p90 %d max %d mean %.1f"
          % (plies[0], plies[n // 10], plies[n // 2], plies[9 * n // 10],
             plies[-1], sum(plies) / n))
    zero = sum(1 for p in plies if p == 0)
    print("games with no moves: %d (%.2f%%)" % (zero, 100 * zero / n))
    print("tag presence:")
    for k, v in tags.most_common():
        print("  %-18s %6d (%.1f%%)" % (k, v, 100 * v / (games - 1)))


def strided_sample(path):
    """Mean plies and eval coverage across the whole file."""
    awk = r'''
      /^\[Event/     { g++
                       if (g > 1) { tot += p; evtot += e; if (e) { ge++; getot += e } }
                       p = 0; e = 0; next }
      /^\[[A-Za-z]/ { next }   # header, not wrapped movetext beginning "[%clk"
                     { p += gsub(/%clk/, "x"); e += gsub(/%eval/, "x") }
      END            { n = g - 1
                       printf "strided sample: %d games\n", n
                       printf "  mean plies per game:            %.1f\n", tot / n
                       printf "  games with any eval:            %.1f%%\n", 100 * ge / n
                       printf "  mean evals per game (all):      %.2f\n", evtot / n
                       printf "  mean evals per annotated game:  %.1f\n", getot / ge
                       printf "  eval coverage within those:     %.1f%% of plies\n",
                              100 * getot / (ge * (tot / n)) }
    '''
    cmds = " ".join(
        "dd if=%s bs=1M skip=%d count=%d 2>/dev/null;" % (path, off, CHUNK_MB)
        for off in OFFSETS_MB
    )
    subprocess.run(["bash", "-c", "{ %s } | LC_ALL=C awk '%s'" % (cmds, awk)],
                   check=True)


if __name__ == "__main__":
    path = pgn_path()
    head_sample(path)
    strided_sample(path)
