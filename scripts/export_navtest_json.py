#!/usr/bin/env python3
"""Export NAVSIM navtest nocot JSON from OpenScene logs (no 72B processor)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.navsim_data import JSON_DIR, LOG_DIR, iter_navtest_scenes, scene_to_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", type=Path, default=LOG_DIR)
    p.add_argument("--out", type=Path, default=JSON_DIR)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for _, window, current in iter_navtest_scenes(log_dir=args.log_dir):
        rec = scene_to_json(window, current)
        (args.out / f"{rec['token']}.json").write_text(json.dumps(rec))
        n += 1
        if n % 500 == 0:
            print(f"[export] {n} {rec['token']}", flush=True)
        if args.limit and n >= args.limit:
            break
    print(f"[done] wrote {n} json files to {args.out}", flush=True)


if __name__ == "__main__":
    main()
