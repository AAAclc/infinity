# VLA competition workspace (LIBERO-Object / Iluvatar CoreX)

Layout
- libero_object/      extracted official dataset (LeRobot v2.1)
- cache/frames/       decoded JPEG frame cache (built by scripts/build_frame_cache.py)
- src/                data pipeline (numpy core; torch wrapper lazy)
- scripts/            runnable entry points
- outputs/            split.json and other artifacts
- docs/               PLAN.md (strategy), data_audit.md (measured dataset facts)

Common commands (base Anaconda python has pyarrow/pandas/numpy/PIL; torch NOT required for the core)
    python scripts/make_split.py
    python scripts/build_frame_cache.py            # all episodes
    python scripts/build_frame_cache.py 0 1 2      # subset

Notes
- Videos are AV1 with moov-after-mdat; decoding uses the ffmpeg binary in src/config.py (FFMPEG).
- Action convention is configurable in src/action_adapter.py; flip it when the
  official interface doc arrives.
