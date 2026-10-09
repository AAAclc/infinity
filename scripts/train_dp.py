"""Train the Diffusion-Policy safety net (real training version).

    python scripts/train_dp.py --device cuda --epochs 60 --batch 64
    python scripts/train_dp.py --device cpu  --epochs 1 --max-steps 20   # smoke

Features: val-loss monitoring, best-ckpt saving, json log, resume.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.dataset import LiberoObjectDataset
from src.model_dp import DiffusionPolicy


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--resume", default="")
    p.add_argument("--ckpt", default=os.path.join("outputs", "dp.pt"))
    p.add_argument("--best", default=os.path.join("outputs", "dp_best.pt"))
    p.add_argument("--log", default=os.path.join("outputs", "train_log.json"))
    return p.parse_args()


def eval_loss(model, dl, device, max_batches=20):
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for i, b in enumerate(dl):
            if i >= max_batches:
                break
            imgs = {k: v.to(device) for k, v in b["images"].items()}
            tot += model.loss(imgs, b["state"].to(device), b["task_index"].to(device), b["action_chunk"].to(device)).item()
            n += 1
    model.train()
    return tot / max(n, 1)


def main():
    a = parse()
    device = torch.device(a.device)
    print("device:", device, "| cuda ok:", torch.cuda.is_available() if a.device != "cpu" else False)
    tr = LiberoObjectDataset(split="train", horizon=a.horizon, augment=True, seed=0)
    va = LiberoObjectDataset(split="val", horizon=a.horizon, augment=False, seed=1)
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers)
    model = DiffusionPolicy(horizon=a.horizon).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    start_ep = 0
    best = 1e9
    history = []
    if a.resume and os.path.isfile(a.resume):
        ck = torch.load(a.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        start_ep = ck.get("epoch", 0)
        best = ck.get("best", 1e9)
        history = ck.get("history", [])
        print("resumed from", a.resume, "epoch", start_ep)
    print("params %.2fM | train %d | val %d" % (sum(p.numel() for p in model.parameters()) / 1e6, len(tr), len(va)))

    step = 0
    t0 = time.time()
    for ep in range(start_ep, a.epochs):
        model.train()
        for b in dl:
            imgs = {k: v.to(device) for k, v in b["images"].items()}
            loss = model.loss(imgs, b["state"].to(device), b["task_index"].to(device), b["action_chunk"].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 100 == 0:
                print("  ep%d step%d loss=%.4f (%.0fs)" % (ep, step, loss.item(), time.time() - t0), flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        if (ep + 1) % a.val_every == 0 or ep == a.epochs - 1:
            v = eval_loss(model, vl, device)
            history.append(dict(epoch=ep, train=loss.item(), val=v, secs=round(time.time() - t0, 1)))
            print("EPOCH %d  train=%.4f  val=%.4f  best=%.4f  (%.0fs)" % (ep, loss.item(), v, best, time.time() - t0), flush=True)
            if v < best:
                best = v
                torch.save({"model": model.state_dict(), "horizon": a.horizon, "epoch": ep + 1, "best": best, "history": history}, a.best)
                print("   -> saved best", a.best)
        if a.max_steps and step >= a.max_steps:
            break
    torch.save({"model": model.state_dict(), "horizon": a.horizon, "epoch": a.epochs, "best": best, "history": history}, a.ckpt)
    with open(a.log, "w") as f:
        json.dump(history, f, indent=2)
    print("saved", a.ckpt, "| log", a.log)
    print("TRAIN DONE best_val=%.4f" % best)


if __name__ == "__main__":
    main()
