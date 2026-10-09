# Iluvatar platform runbook

Copy this repo into the lease workdir, e.g. /mnt/localforge_leases/pl_5c4ee466e59a/repo,
then:

    bash scripts/platform_setup.sh          # GPU check, op smoke, shm, HF mirror, LIBERO install, obs dump

Then calibrate src/eval_libero.py KEY_* / FLIP_IMAGES using the --dump-obs output.

Train the safety net:
    python3 scripts/build_frame_cache.py    # decode AV1 -> JPEG cache
    python3 scripts/make_split.py
    python3 scripts/train_dp.py --device cuda --epochs 60 --batch 64

Evaluate:
    python3 scripts/run_eval.py --env libero --policy dp --ckpt outputs/dp.pt --episodes 50 --device cuda

Known platform facts (from audit):
    torch 2.7.1+corex.4.4.0, flash_attn 2.6.3+corex, transformers 4.57.6
    huggingface.co blocked -> use HF_ENDPOINT=https://hf-mirror.com or ModelScope
    /share has models--openai--clip-vit-large-patch14 (for the CLIP-text upgrade path)
    do NOT `source /usr/local/corex/enable` (it breaks the preinstalled torch env)
    persistent storage = the lease workdir, NOT /private or /share (both unmounted)
