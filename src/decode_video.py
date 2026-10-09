"""Decode AV1 mp4 episodes to a JPEG frame cache using a local ffmpeg.

Why: the dataset videos are AV1 with moov-after-mdat, which decord and many
streaming readers cannot handle. A local ffmpeg binary decodes them reliably.
Frames are cached as JPEG (quality 2) to keep the cache ~1-2 GB instead of
~27 GB of raw RGB.
"""
import os
import subprocess

from . import config as C


def frame_dir(ep, cam):
    return os.path.join(C.FRAMES_DIR, C.CAM_SHORT[cam], "ep%06d" % ep)


def frame_path(ep, cam, idx):
    return os.path.join(frame_dir(ep, cam), "f%05d.jpg" % idx)


def is_cached(ep, cam, length):
    d = frame_dir(ep, cam)
    if not os.path.isdir(d):
        return False
    return os.path.isfile(frame_path(ep, cam, 0)) and os.path.isfile(frame_path(ep, cam, length - 1))


def decode_episode(ep, cam, force=False, quality=2):
    """Decode one episode-camera to JPEGs. Returns number of frames written."""
    length = None
    outdir = frame_dir(ep, cam)
    if not force and os.path.isdir(outdir):
        n = len([f for f in os.listdir(outdir) if f.endswith(".jpg")])
        if n > 0:
            return n
    os.makedirs(outdir, exist_ok=True)
    mp4 = C.VIDEO_TEMPLATE % (cam, ep)
    cmd = [
        C.FFMPEG, "-v", "error", "-y", "-i", mp4,
        "-q:v", str(quality), "-start_number", "0",
        os.path.join(outdir, "f%05d.jpg"),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed for %s: %s" % (mp4, r.stderr.decode(errors="replace")[:300]))
    return len([f for f in os.listdir(outdir) if f.endswith(".jpg")])
