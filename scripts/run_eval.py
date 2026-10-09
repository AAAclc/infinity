"""Model-agnostic evaluation loop.

    python scripts/run_eval.py --env mock  --policy random --episodes 20
    python scripts/run_eval.py --env mock  --policy dp --ckpt outputs/dp.pt --episodes 2
    python scripts/run_eval.py --env libero --policy dp --ckpt dp.pt --episodes 50   # on platform
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.eval_env import MockEnv
from src.policy_wrapper import DPPolicy, RandomPolicy


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="mock", choices=["mock", "libero"])
    p.add_argument("--policy", default="random", choices=["random", "dp"])
    p.add_argument("--ckpt", default="outputs/dp.pt")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--device", default="cpu")
    p.add_argument("--exec-horizon", type=int, default=8)
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--dump-obs", action="store_true")
    p.add_argument("--out", default=os.path.join("outputs", "eval_results.json"))
    return p.parse_args()


def build_env(a):
    if a.env == "mock":
        return MockEnv(max_steps=a.max_steps, seed=0)
    from src.eval_libero import LiberoEnv
    return LiberoEnv(suite=a.suite, max_steps=a.max_steps)


def build_policy(a):
    if a.policy == "random":
        return RandomPolicy(seed=0)
    return DPPolicy(a.ckpt, device=a.device, exec_horizon=a.exec_horizon)


def main():
    a = parse()
    env = build_env(a)
    if a.dump_obs:
        env.dump_obs()
        return
    policy = build_policy(a)
    n_tasks = env.n_tasks
    results = {}
    t0 = time.time()
    for ti in range(n_tasks):
        succ = 0
        for ep in range(a.episodes):
            policy.reset()
            env.reset(ti, ep)
            done = False
            info = {}
            while not done:
                obs = env.get_obs()
                action = policy.step(obs)
                done, info = env.step(action)
            succ += int(info.get("success", False))
        rate = succ / a.episodes
        results[str(ti)] = dict(success=succ, episodes=a.episodes, rate=rate)
        print("task %d: %d/%d = %.1f%%  (%.0fs elapsed)" % (ti, succ, a.episodes, rate * 100, time.time() - t0), flush=True)
    mean = float(np.mean([r["rate"] for r in results.values()]))
    print("")
    print("MEAN SUCCESS RATE: %.1f%%" % (mean * 100))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(mean=mean, per_task=results, args=vars(a)), f, indent=2)
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
