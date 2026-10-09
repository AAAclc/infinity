# LIBERO-Object dataset audit (measured, 2026-10-09)

Source: `libero_object.zip` (LeRobot v2.1). All numbers below were verified by
parsing the 454 parquet files and 908 mp4 files directly.

## Specs
| item | value |
|---|---|
| format | LeRobot v2.1, single chunk |
| robot | franka (OSC pose control) |
| tasks | 10 (LIBERO-Object suite), interleaved order |
| episodes | 454 |
| frames | 66,984 @ 20 fps |
| episode length | 114 / 147.5 / 254 (min/avg/max) |
| cameras | `observation.images.image` + `observation.images.wrist_image`, 256x256 |
| state | 8D `[x,y,z, ax,ay,az, grip1,grip2]` |
| action | 7D `[x,y,z, ax,ay,az, gripper]` |
| extra cols | timestamp/frame_index/episode_index/index/task_index (no reward, no done) |
| language | not in parquet; in `meta/tasks.jsonl` via task_index |

## Integrity: all pass
- 0 NaN / 0 Inf
- 908/908 mp4 containers valid (av1, timescale 10240); sample counts == episode lengths
- AV1 fully decodable (spot-checked 16 videos)
- timestamps strictly 0.05 s; frame_index contiguous; lengths match meta
- 454 distinct initial states; 454 distinct final states

## Action space (critical)
- Values lie on an exact uniform grid (post-processed, NOT raw LIBERO):
  - dims 0-2 (xyz): step = 3/1120 = 0.0026785714, range [-0.9375, +0.9375] (= +/-350 steps)
  - dims 3-5 (rot): step = 3/2800 = 0.0010714286
  - dim 6 (gripper): exactly {0.0, 1.0}
- gripper semantics: **1 = OPEN, 0 = CLOSE** (verified from temporal pattern)
- close event at ~34% of episode (mean); 408/454 episodes have exactly one close
- action is an incremental pose command; corr with (state[t+1]-state[t]) = 0.93/0.99/0.97 for x/y/z
- Official LIBERO/robosuite uses gripper in {-1,+1}; this dataset uses {0,1}
  => convention adapter required (src/action_adapter.py)

## Demonstration quality
- 0 episodes without a gripper-close event
- lift after close: mean 0.2197 m, min 0.1323 m  => all 454 are successful demos
- end-effector endpoints cluster at the basket (std 0.026/0.020/0.033 m)

## Diversity / generalization risk
- within-task initial-state spread is small: xyz spread 0.023/0.033/0.025 m (std ~0.006 m)
- hidden test runs 50 seeds per task => generalization gap is the main scoring risk

## Video gotchas
- codec av1 (decord cannot decode); moov AFTER mdat (streaming/pipe reads fail)
- use a local ffmpeg binary; frames cached as JPEG under cache/frames/
