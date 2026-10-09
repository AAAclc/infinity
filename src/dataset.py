"""PyTorch Dataset wrapper. torch is imported lazily inside __getitem__
so the numpy core remains usable on machines without torch."""
import numpy as np

from . import augment as A
from . import config as C
from . import dataset_core as core
from .meta import episode_table
from .split import load_split


class LiberoObjectDataset:
    def __init__(self, split="train", horizon=16, state_hist=1, augment=False,
                 seed=0, cameras=None, normalize_image=True):
        sp = load_split()
        keep = set(sp["train"] if split == "train" else sp["val"])
        rows, _ = episode_table()
        self.items = []
        for r in rows:
            if r["episode_index"] in keep:
                for t in range(r["length"]):
                    self.items.append((r["episode_index"], t))
        self.horizon = horizon
        self.state_hist = state_hist
        self.augment = augment
        self.cameras = cameras or C.CAMERAS
        self.normalize_image = normalize_image
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        ep, t = self.items[i]
        s = core.build_sample(ep, t, horizon=self.horizon,
                              state_hist=self.state_hist, cameras=self.cameras)
        imgs = {}
        for k, v in s["images"].items():
            if self.augment:
                v = A.aug_image(v, self.rng)
            x = v.astype(np.float32)
            if self.normalize_image:
                x = x / 255.0
            imgs[k] = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        state = s["state"]
        if self.augment:
            state = A.aug_state(state, self.rng)
        return dict(
            images=imgs,
            state=torch.from_numpy(state),
            state_hist=torch.from_numpy(s["state_hist"]),
            action=torch.from_numpy(s["action"]),
            action_chunk=torch.from_numpy(s["action_chunk"]),
            task_index=torch.tensor(s["task_index"], dtype=torch.long),
            frame_index=torch.tensor(s["frame_index"], dtype=torch.long),
        )
