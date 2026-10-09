"""Env interface + MockEnv so the eval loop is validated without LIBERO.

The real LIBERO adapter lives in eval_libero.py (needs libero installed; run on
the Iluvatar platform).
"""
import numpy as np

INSTRUCTIONS = [
    "pick up the orange juice and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the alphabet soup and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
]


class MockEnv:
    """Fake env: succeeds after succeed_after steps with probability p_success."""

    def __init__(self, n_tasks=10, max_steps=300, seed=0, p_success=0.5, succeed_after=50):
        self.n_tasks = n_tasks
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.p_success = p_success
        self.succeed_after = succeed_after
        self._t = 0
        self._will = False
        self.task_id = 0

    def reset(self, task_id, episode):
        self.task_id = task_id
        self._t = 0
        self._will = self.rng.random() < self.p_success
        return self._obs()

    def _obs(self):
        return dict(
            images={
                "main": self.rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
                "wrist": self.rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            },
            state=np.zeros(8, dtype=np.float32),
            instruction=INSTRUCTIONS[self.task_id],
        )

    def get_obs(self):
        return self._obs()

    def step(self, action):
        self._t += 1
        success = bool(self._will and self._t >= self.succeed_after)
        done = success or self._t >= self.max_steps
        return done, dict(success=success)
