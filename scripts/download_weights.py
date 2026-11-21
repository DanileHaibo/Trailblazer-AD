#!/usr/bin/env python3
"""Download the two teacher checkpoints. Token from HF_TOKEN only."""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not TOKEN:
    raise SystemExit("HF_TOKEN is not set")

JOBS = [
    ("OpenDriveVLA/OpenDriveVLA-0.5B", "/data2/hf_overlap_models/OpenDriveVLA-0.5B"),
    ("stanfordasl/nuscenes-waypoints-model", "/data2/hf_overlap_models/nuscenes-waypoints-model"),
]


def main():
    for repo, dest in JOBS:
        Path(dest).mkdir(parents=True, exist_ok=True)
        print(f"[download] {repo} -> {dest}", flush=True)
        snapshot_download(
            repo_id=repo,
            local_dir=dest,
            token=TOKEN,
            resume_download=True,
            max_workers=8,
        )
        print(f"[done] {repo}", flush=True)


if __name__ == "__main__":
    main()
