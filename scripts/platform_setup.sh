#!/usr/bin/env bash
# Platform bring-up + eval bootstrap for the Iluvatar (CoreX) box.
# Run inside the lease workdir after copying this repo there.
set -x
cd "$(dirname "$0")/.." || exit 1

echo "===== [1] GPU / torch ====="
ixsmi 2>&1 | head -20
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())"

echo "===== [2] operator smoke (bf16 matmul / conv2d / flash-attn) ====="
python3 - <<'PY'
import torch, time
a=torch.randn(2048,2048,device="cuda",dtype=torch.bfloat16); b=torch.randn_like(a)
torch.cuda.synchronize(); t=time.time(); c=a@b; torch.cuda.synchronize()
print("bf16 matmul OK %.1f ms"%((time.time()-t)*1e3))
import torch.nn as nn
print("conv2d OK", nn.Conv2d(3,32,3,padding=1).cuda()(torch.randn(8,3,224,224,device="cuda")).shape)
try:
    from flash_attn import flash_attn_func
    q=torch.randn(2,128,8,64,device="cuda",dtype=torch.bfloat16)
    print("flash_attn OK", flash_attn_func(q,q,q).shape)
except Exception as e:
    print("flash_attn FAIL", repr(e)[:150])
PY

echo "===== [3] shm ====="
df -h /dev/shm | tail -1

echo "===== [4] HF mirror ====="
export HF_ENDPOINT=https://hf-mirror.com
curl -sI -m 8 https://hf-mirror.com -o /dev/null -w "hf-mirror HTTP %{http_code}\n"

echo "===== [5] install LIBERO stack (CPU sim; no GPU needed) ====="
pip3 install --quiet mujoco robosuite 2>&1 | tail -2
pip3 install --quiet git+https://github.com/Lifelong-Robot-Learning/LIBERO.git 2>&1 | tail -2
python3 -c "import libero, robosuite, mujoco; print('libero stack OK')" 2>&1 | tail -2

echo "===== [6] dump raw obs keys (calibrate eval_libero KEY_*) ====="
python3 scripts/run_eval.py --env libero --dump-obs 2>&1 | tail -20

echo "===== [7] deps for our pipeline ====="
pip3 install --quiet pyarrow pandas pillow 2>&1 | tail -2
echo "SETUP DONE"
