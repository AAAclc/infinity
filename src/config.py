"""Central configuration for the LIBERO-Object competition pipeline.

All paths are derived from this file's location so the project is relocatable.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(ROOT, "libero_object")
CACHE_DIR = os.path.join(ROOT, "cache")
FRAMES_DIR = os.path.join(CACHE_DIR, "frames")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

META_INFO = os.path.join(DATASET_DIR, "meta", "info.json")
META_TASKS = os.path.join(DATASET_DIR, "meta", "tasks.jsonl")
META_EPISODES = os.path.join(DATASET_DIR, "meta", "episodes.jsonl")

PARQUET_TEMPLATE = os.path.join(DATASET_DIR, "data", "chunk-000", "episode_%06d.parquet")
VIDEO_TEMPLATE = os.path.join(DATASET_DIR, "videos", "chunk-000", "%s", "episode_%06d.mp4")

CAMERAS = ("observation.images.image", "observation.images.wrist_image")
CAM_SHORT = {"observation.images.image": "main", "observation.images.wrist_image": "wrist"}

STATE_DIM = 8
ACTION_DIM = 7
FPS = 20
IMG_SIZE = 256

# Local ffmpeg that can decode AV1 (verified: gyan.dev 7.0 essentials build).
FFMPEG = os.environ.get("FFMPEG_PATH", r"D:\MC\overrides\ffmpeg.exe")

VAL_FRACTION = 0.15
MIN_VAL_PER_TASK = 3
SEED = 42
