"""Reproduce and persist the stratified train/val split."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import split

if __name__ == "__main__":
    sp = split.make_split()
    p = split.save_split(sp)
    print("train=%d val=%d -> %s" % (len(sp["train"]), len(sp["val"]), p))
    for ti in sorted(sp["per_task"], key=int):
        d = sp["per_task"][ti]
        print("  task %s: total=%d train=%d val=%d" % (ti, d["total"], d["train"], d["val"]))
