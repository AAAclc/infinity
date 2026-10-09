# Competition plan (initial round, LIBERO-Object on Iluvatar CoreX)

## Objective
Maximize mean success rate over 10 tasks x 50 hidden seeds. Target >= 90% mean,
no single task below ~80%. Top 16 advance.

## Architecture: three decoupled layers
1. Data layer (model-agnostic): LeRobot -> frames + state + action + language,
   stratified split, augmentation.  [DONE: src/*]
2. Eval layer (model-agnostic): LIBERO sim + 50 seeds/task + success stats.
   Any model plugs in.  [TODO: this is the highest-value next build]
3. Model layer (swappable): ordered by expected success / portability.
   Action-convention adapter sits between model and env (src/action_adapter.py).

## Model choice
- Primary: pi0 (openpi, PyTorch). Best published LIBERO numbers (~96-99%),
  action-chunk + flow matching suits the 50-run robustness metric.
- Fallback 1: OpenVLA-7B LoRA (~95%).
- Safety net: Diffusion Policy (pure PyTorch, near-zero porting risk).
  Guarantees we always have a runnable, scorable submission.
Strategy: get DP through the full pipeline first (proves the chain), then swap
in pi0 to raise the score.

## Phases
- P0 unblock: platform GPU ticket; official interface doc (action convention).
- P1 env bring-up on Iluvatar: torch sees GPU; bf16 matmul + conv2d + flash_attn
  forward all pass; check /dev/shm; verify hf-mirror.
- P2 data pipeline: [DONE locally] frames cache, split, augment, adapter.
- P3 eval harness on Iluvatar (LIBERO+robosuite+MuJoCo, CPU-render): 10-min feedback loop.
- P4 baseline train: DP safety net -> first real number; then pi0.
- P5 score lifting, by ROI:
    1. extra demos via LIBERO sim + scripted policy across many seeds
       (directly attacks the small-init-randomization generalization gap)
    2. augmentation tuned to failure cases
    3. action-chunk + temporal ensembling to cut variance
    4. multi-seed ensembles
- P6 package per official interface; zip; submit.

## Key risks
| risk | mitigation |
|---|---|
| GPU never granted | ticket + support group; meanwhile local 5070Ti for DP-scale checks |
| pi0 won't port to CoreX | fallback OpenVLA; safety net DP |
| interface doc unavailable | adapter configurable; measured convention as default |
| generalization gap (50 seeds) | P5.1 extra demos + augmentation |
| huggingface blocked | hf-mirror / ModelScope; /share already has clip-vit-large-patch14 |
| /dev/shm too small | check in P1; else DataLoader num_workers=0 |
