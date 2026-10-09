"""Metadata + parquet access for the LeRobot v2.1 LIBERO-Object dataset."""
import json

import numpy as np
import pyarrow.parquet as pq

from . import config as C


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_meta():
    with open(C.META_INFO, "r", encoding="utf-8") as f:
        info = json.load(f)
    tasks = load_jsonl(C.META_TASKS)
    episodes = load_jsonl(C.META_EPISODES)
    return info, tasks, episodes


def episode_table():
    """Return (rows, task_map). rows: list of dict(episode_index, task_index, task, length)."""
    info, tasks, episodes = load_meta()
    task_map = {t["task_index"]: t["task"] for t in tasks}
    text_to_idx = {v: k for k, v in task_map.items()}
    rows = []
    for e in episodes:
        txt = e["tasks"][0]
        rows.append(
            dict(
                episode_index=int(e["episode_index"]),
                task_index=int(text_to_idx[txt]),
                task=txt,
                length=int(e["length"]),
            )
        )
    rows.sort(key=lambda r: r["episode_index"])
    return rows, task_map


def load_episode(ep):
    """Load one episode's parquet into numpy arrays."""
    df = pq.read_table(C.PARQUET_TEMPLATE % ep).to_pandas()
    return dict(
        state=np.asarray(df["observation.state"].tolist(), dtype=np.float32),
        action=np.asarray(df["action"].tolist(), dtype=np.float32),
        timestamp=df["timestamp"].to_numpy(np.float32),
        frame_index=df["frame_index"].to_numpy(np.int64),
        task_index=df["task_index"].to_numpy(np.int64),
    )
