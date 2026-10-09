#!/bin/bash
# One-shot platform bring-up (CPU path). Run from anywhere: bash scripts/setup_cpu.sh
set -e
cd "$(dirname "$0")/.."
echo "== download official dataset =="
[ -f libero_object.zip ] || curl -L -o libero_object.zip "https://openloong.obs.cn-east-3.myhuaweicloud.com/public/static/contests/other-1/libero_object.zip"
echo "== extract =="
[ -d libero_object/meta ] || python3 -c "import zipfile;zipfile.ZipFile('libero_object.zip').extractall('libero_object')"
echo "== deps =="
pip3 install --quiet pyarrow pandas pillow imageio-ffmpeg 2>&1 | tail -1
python3 -c "import imageio_ffmpeg as i;print('FFMPEG_PATH='+i.get_ffmpeg_exe())" > ffmpeg_env.sh
echo "== data smoke =="
python3 -c "import sys;sys.path.insert(0,'.');from src import meta;r,_=meta.episode_table();print('episodes',len(r))"
echo SETUP_OK
