"""Numpy-level sample construction (no torch dependency)."""
import numpy as np
from PIL import Image

from . import config as C
from .decode_video import frame_path
from .meta import episode_table, load_episode

_EP_CACHE = {}
_ROWS = None


def rows_by_ep():
    global _ROWS
    if _ROWS is None:
        rows, _ = episode_table()
        _ROWS = {r["episode_index"]: r for r in rows}
    return _ROWS


def get_episode(ep):
    if ep not in _EP_CACHE:
        _EP_CACHE[ep] = load_episode(ep)
    return _EP_CACHE[ep]


def load_image(ep, cam, idx):
    with Image.open(frame_path(ep, cam, idx)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def get_action_chunk(ep, t, horizon):
    a = get_episode(ep)["action"]
    T = a.shape[0]
    chunk = np.empty((horizon, C.ACTION_DIM), dtype=np.float32)
    last = a[T - 1]
    for k in range(horizon):
        idx = t + k
        chunk[k] = a[idx] if idx < T else last
    return chunk


def get_state_hist(ep, t, k):
    s = get_episode(ep)["state"]
    out = np.empty((k, C.STATE_DIM), dtype=np.float32)
    for j in range(k):
        idx = max(t - (k - 1 - j), 0)
        out[j] = s[idx]
    return out


def build_sample(ep, t, horizon=16, state_hist=1, cameras=None):
    cameras = cameras or C.CAMERAS
    e = get_episode(ep)
    r = rows_by_ep()[ep]
    return dict(
        episode_index=int(ep),
        frame_index=int(t),
        task_index=int(r["task_index"]),
        task=r["task"],
        images={C.CAM_SHORT[c]: load_image(ep, c, t) for c in cameras},
        state=e["state"][t].astype(np.float32),
        state_hist=get_state_hist(ep, t, state_hist),
        action=e["action"][t].astype(np.float32),
        action_chunk=get_action_chunk(ep, t, horizon),
    )
