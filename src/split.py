"""Stratified train/val split (tasks are interleaved in this dataset)."""
import json
import os
import random

from . import config as C
from .meta import episode_table


def make_split(val_fraction=None, seed=None):
    val_fraction = C.VAL_FRACTION if val_fraction is None else val_fraction
    seed = C.SEED if seed is None else seed
    rows, _ = episode_table()
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_index"], []).append(r["episode_index"])
    rng = random.Random(seed)
    train, val = [], []
    per_task = {}
    for ti in sorted(by_task):
        eps = sorted(by_task[ti])
        rng.shuffle(eps)
        nval = max(C.MIN_VAL_PER_TASK, int(round(val_fraction * len(eps))))
        val += eps[:nval]
        train += eps[nval:]
        per_task[ti] = dict(total=len(eps), val=nval, train=len(eps) - nval)
    train.sort()
    val.sort()
    return dict(train=train, val=val, seed=seed, val_fraction=val_fraction, per_task=per_task)


def split_path():
    return os.path.join(C.OUTPUT_DIR, "split.json")


def save_split(split, path=None):
    path = path or split_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return path


def load_split(path=None):
    path = path or split_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
