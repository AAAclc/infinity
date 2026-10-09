"""Build the JPEG frame cache for all episodes/cameras (or a subset).

Usage:
    python scripts/build_frame_cache.py            # all
    python scripts/build_frame_cache.py 0 1 2      # specific episodes
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C
from src.decode_video import decode_episode, is_cached
from src.meta import episode_table


def main():
    rows, _ = episode_table()
    if len(sys.argv) > 1:
        want = set(int(x) for x in sys.argv[1:])
        rows = [r for r in rows if r["episode_index"] in want]
    t0 = time.time()
    done = skipped = 0
    for i, r in enumerate(rows):
        ep = r["episode_index"]
        for cam in C.CAMERAS:
            if is_cached(ep, cam, r["length"]):
                skipped += 1
                continue
            n = decode_episode(ep, cam)
            assert n == r["length"], "ep%d %s decoded %d != meta %d" % (ep, cam, n, r["length"])
            done += 1
        if (i + 1) % 25 == 0 or i == len(rows) - 1:
            print("[%d/%d] ep=%d  decoded=%d skipped=%d  %.1fs" % (i + 1, len(rows), ep, done, skipped, time.time() - t0), flush=True)
    print("DONE decoded=%d skipped=%d total=%.1fs" % (done, skipped, time.time() - t0))


if __name__ == "__main__":
    main()
