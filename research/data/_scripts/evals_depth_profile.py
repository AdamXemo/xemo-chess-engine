"""Best-available search depth per position, which is the filter that matters."""
import collections
import json

from _common import evals_path

SAMPLE = 500000


def main():
    n = d20 = d30 = any_mate = 0
    buckets = collections.Counter()
    with open(evals_path()) as fh:
        for line in fh:
            rec = json.loads(line)
            n += 1
            best = max(ev["depth"] for ev in rec["evals"])
            buckets[min(best // 10 * 10, 60)] += 1
            if best >= 20:
                d20 += 1
            if best >= 30:
                d30 += 1
            if any("mate" in pv for ev in rec["evals"] for pv in ev["pvs"]):
                any_mate += 1
            if n >= SAMPLE:
                break

    print("sampled positions:", n)
    print("best depth >=20: %.1f%%   >=30: %.1f%%" % (100 * d20 / n, 100 * d30 / n))
    print("positions containing a mate score: %.1f%%" % (100 * any_mate / n))
    print("depth band | positions | share")
    for band in sorted(buckets):
        label = "%d-%d" % (band, band + 9) if band < 60 else "60+"
        print("%10s | %9d | %5.2f%%" % (label, buckets[band], 100 * buckets[band] / n))


if __name__ == "__main__":
    main()
