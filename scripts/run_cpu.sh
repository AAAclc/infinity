#!/bin/bash
# CPU training path on the platform. bash scripts/run_cpu.sh
cd "$(dirname "$0")/.."
source ffmpeg_env.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
python3 scripts/build_frame_cache.py 2>&1 | tail -3
python3 scripts/make_split.py
python3 scripts/train_dp.py --device cpu --epochs ${EPOCHS:-40} --batch ${BATCH:-64} --workers 16 2>&1 | tail -40
echo RUN_CPU_DONE
