"""Smoke-test the torch data path + a tiny model on CPU.

Validates: torch Dataset __getitem__, DataLoader batching, tensor shapes,
and a minimal forward/backward. Runs on CPU (local GPU is sm_120 which the
restored cu124 torch cannot execute on; real GPU training happens on Iluvatar).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import LiberoObjectDataset

HORIZON = 16


class TinyPolicy(nn.Module):
    def __init__(self, horizon=HORIZON, hidden=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(16 * 64 * 64 * 2 + 8, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon * 7),
        )

    def forward(self, images, state):
        feats = [self.stem(images[k]) for k in sorted(images.keys())]
        x = torch.cat(feats + [state], dim=-1)
        return self.head(x).view(-1, HORIZON, 7)


def main():
    print("torch", torch.__version__, "| device: cpu (smoke test)")
    ds = LiberoObjectDataset(split="train", horizon=HORIZON, augment=True, seed=0)
    print("dataset samples:", len(ds))
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    model = TinyPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for i, batch in enumerate(dl):
        imgs = batch["images"]
        state = batch["state"]
        target = batch["action_chunk"]
        pred = model(imgs, state)
        loss = nn.functional.mse_loss(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if i == 0:
            print("  batch images:", {k: tuple(v.shape) for k, v in imgs.items()})
            print("  batch state :", tuple(state.shape), "| target:", tuple(target.shape), "| pred:", tuple(pred.shape))
        if i >= 4:
            break
    print("5 steps OK, final loss=%.4f, %.1fs" % (loss.item(), time.time() - t0))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
