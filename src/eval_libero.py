"""LIBERO env adapter (run on the Iluvatar platform where libero is installed).

Implements the same interface as eval_env.MockEnv:
    reset(task_id, episode) -> obs
    get_obs() -> obs
    step(action7) -> (done, info)
obs = dict(images={"main","wrist"} uint8 HWC, state=(8,) f32, instruction=str)

NOTE: observation key names / image orientation differ across libero versions.
Use `--dump-obs` in run_eval.py on the platform to print the raw obs dict once,
then adjust KEY_* / FLIP below accordingly.
"""
import numpy as np

KEY_MAIN = "agentview_image"
KEY_WRIST = "robot0_eye_in_hand_image"
KEY_EEF_POS = "robot0_eef_pos"
KEY_EEF_QUAT = "robot0_eef_quat"
KEY_GRIP = "robot0_gripper_qpos"
FLIP_IMAGES = True   # robosuite offscreen images are often vertically flipped


class LiberoEnv:
    def __init__(self, suite="libero_object", max_steps=520, seed=0, cameras=("main", "wrist")):
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
        self.benchmark = benchmark
        self.OffScreenRenderEnv = OffScreenRenderEnv
        self.suite = suite
        self.max_steps = max_steps
        self.seed = seed
        self.task_suite = benchmark.get_benchmark_dict()[suite]()
        self.n_tasks = self.task_suite.n_tasks
        self.env = None
        self._t = 0
        self._task_id = 0
        self._obs = None

    def _quat_to_axisangle(self, q):
        from robosuite.utils.transform_utils import quat2axisangle
        return np.asarray(quat2axisangle(np.asarray(q)), dtype=np.float32)

    def reset(self, task_id, episode):
        import robosuite.macros as macros  # noqa: F401
        task = self.task_suite.get_task(task_id)
        init_states = self.task_suite.get_task_init_states(task_id)
        if self.env is None:
            self.env = self.OffScreenRenderEnv(
                **dict(
                    bddl_file_name=task.bddl_file,
                    robots="Panda",
                    camera_names=("agentview", "robot0_eye_in_hand"),
                    camera_heights=256,
                    camera_widths=256,
                    has_offscreen_renderer=True,
                    render_camera="frontview",
                )
            )
        raw = self.env.reset()
        idx = episode % len(init_states)
        self.env.env.set_init_state(init_states[idx])
        for _ in range(5):
            raw = self.env.step(np.zeros(7))[0] if isinstance(self.env.step(np.zeros(7)), tuple) else raw
        self._task_id = task_id
        self._t = 0
        self._instruction = task.language
        self._obs = self._convert(raw)
        return self._obs

    def _convert(self, raw):
        o = raw[0] if isinstance(raw, tuple) else raw
        main = np.asarray(o[KEY_MAIN], dtype=np.uint8)
        wrist = np.asarray(o[KEY_WRIST], dtype=np.uint8)
        if FLIP_IMAGES:
            main = main[::-1].copy()
            wrist = wrist[::-1].copy()
        state = np.concatenate([
            np.asarray(o[KEY_EEF_POS], dtype=np.float32),
            self._quat_to_axisangle(o[KEY_EEF_QUAT]),
            np.asarray(o[KEY_GRIP], dtype=np.float32)[:2],
        ]).astype(np.float32)
        return dict(images={"main": main, "wrist": wrist}, state=state, instruction=self._instruction)

    def get_obs(self):
        return self._obs

    def step(self, action):
        raw = self.env.step(np.asarray(action, dtype=np.float32))
        o = raw[0] if isinstance(raw, tuple) else raw
        done = bool(raw[1]) if isinstance(raw, tuple) and raw[1] is not None else False
        info = raw[2] if isinstance(raw, tuple) and len(raw) > 2 and isinstance(raw[2], dict) else {}
        success = bool(info.get("success", False))
        self._t += 1
        self._obs = self._convert(raw)
        done = done or success or self._t >= self.max_steps
        return done, dict(success=success)

    def dump_obs(self, task_id=0):
        obs = self.reset(task_id, 0)
        print("instruction:", obs["instruction"])
        print("main:", obs["images"]["main"].shape, obs["images"]["main"].dtype)
        print("wrist:", obs["images"]["wrist"].shape)
        print("state:", np.round(obs["state"], 4))
