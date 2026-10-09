"""Model-agnostic policy interface + DP implementation.

step(obs) -> action7 in ENV space.
obs = dict(images={"main","wrist"} uint8 (H,W,3), state=(8,) float32, instruction=str)
Action chunks are re-planned every exec_horizon steps.
"""
import numpy as np
import torch

from . import action_adapter as AA
from . import task_match
from .model_dp import DiffusionPolicy


class BasePolicy:
    space = AA.LIBERO_RAW

    def reset(self):
        pass

    def step(self, obs):
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def step(self, obs):
        a = self.rng.uniform(-0.5, 0.5, 6).astype(np.float32)
        g = float(self.rng.integers(0, 2))
        return AA.to_space(np.concatenate([a, [g]]), self.space)


class DPPolicy(BasePolicy):
    def __init__(self, ckpt, device="cpu", horizon=16, exec_horizon=8, space=AA.LIBERO_RAW):
        self.device = torch.device(device)
        ck = torch.load(ckpt, map_location="cpu")
        self.model = DiffusionPolicy(horizon=ck.get("horizon", horizon))
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.horizon = ck.get("horizon", horizon)
        self.exec_horizon = exec_horizon
        self.space = space
        self._buffer = []

    def reset(self):
        self._buffer = []

    def _preprocess(self, obs):
        imgs = {}
        for k in ("main", "wrist"):
            x = np.asarray(obs["images"][k], dtype=np.float32) / 255.0
            imgs[k] = torch.from_numpy(x).permute(2, 0, 1)[None]
        state = torch.from_numpy(np.asarray(obs["state"], dtype=np.float32))[None]
        ti = torch.tensor([task_match.match_instruction(obs["instruction"])], dtype=torch.long)
        return ({k: v.to(self.device) for k, v in imgs.items()},
                state.to(self.device), ti.to(self.device))

    def step(self, obs):
        if not self._buffer:
            imgs, state, ti = self._preprocess(obs)
            with torch.no_grad():
                chunk = self.model.predict(imgs, state, ti)[0].cpu().numpy()
            self._buffer = [row for row in chunk[: self.exec_horizon]]
        a = np.asarray(self._buffer.pop(0), dtype=np.float32).copy()
        a[6] = 1.0 if a[6] > 0.5 else 0.0
        return AA.to_space(a, self.space)
