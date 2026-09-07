"""Schema, score, and principal-variation shape of the evaluation database.

Samples at ten evenly spaced file offsets rather than reading the head, so the
numbers describe the whole file.
"""
import collections
import json
import os

from _common import evals_path, quantiles

PER_OFFSET = 20000
OFFSETS = 10


def main():
    path = evals_path()
    size = os.path.getsize(path)
    keys = collections.Counter()
    fen_fields, stm, castling = collections.Counter(), collections.Counter(), collections.Counter()
    pieces, per_pos, per_eval, depths, knodes, line_len, cps, mates = ([] for _ in range(8))
    cp_n = mate_n = n = 0

    for i in range(OFFSETS):
        with open(path, "rb") as fh:
            fh.seek(int(size * i / OFFSETS))
            fh.readline()  # discard the partial line
            for _ in range(PER_OFFSET):
                raw = fh.readline()
                if not raw:
                    break
                rec = json.loads(raw)
                n += 1
                keys.update(rec.keys())
                fields = rec["fen"].split()
                fen_fields[len(fields)] += 1
                stm[fields[1]] += 1
                castling[fields[2]] += 1
                pieces.append(sum(c.isalpha() for c in fields[0]))
                per_pos.append(len(rec["evals"]))
                for ev in rec["evals"]:
                    depths.append(ev["depth"])
                    knodes.append(ev["knodes"])
                    per_eval.append(len(ev["pvs"]))
                    for pv in ev["pvs"]:
                        line_len.append(len(pv["line"].split()))
                        if "cp" in pv:
                            cp_n += 1
                            cps.append(pv["cp"])
                        if "mate" in pv:
                            mate_n += 1
                            mates.append(pv["mate"])

    print("sampled positions:", n)
    print("top-level keys:", dict(keys))
    print("fen field counts:", dict(fen_fields))
    print("side to move:", dict(stm))
    print("castling rights:", castling.most_common(6))
    quantiles("pieces on board", pieces)
    quantiles("analyses per position", per_pos)
    quantiles("pvs per analysis", per_eval)
    quantiles("depth", depths)
    quantiles("knodes", knodes)
    quantiles("pv line length (plies)", line_len)
    quantiles("cp score", cps)
    quantiles("mate distance", mates)
    print("pv entries with cp: %d (%.1f%%), with mate: %d (%.1f%%)"
          % (cp_n, 100 * cp_n / (cp_n + mate_n), mate_n,
             100 * mate_n / (cp_n + mate_n)))
    print("mate distance concentration:",
          collections.Counter(abs(m) for m in mates).most_common(8))


if __name__ == "__main__":
    main()
