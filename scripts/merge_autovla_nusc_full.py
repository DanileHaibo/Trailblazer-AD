#!/usr/bin/env python3
"""Merge AutoVLA full-val shards into OpenDrive token order (6019)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--od-cache", type=Path, default=ROOT / "results/cache_opendrive_nusc_val.pt")
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=ROOT / "results/cache_autovla_nusc_val_full.pt")
    args = p.parse_args()

    order = [r["token"] for r in torch.load(args.od_cache, map_location="cpu", weights_only=False)["rows"]]
    by_tok = {}
    hidden = 2048
    for sh in args.shards:
        blob = torch.load(sh, map_location="cpu", weights_only=False)
        hidden = int(blob.get("hidden_size", hidden))
        for r in blob["rows"]:
            by_tok[r["token"]] = r
    missing = [t for t in order if t not in by_tok]
    rows = [by_tok[t] for t in order if t in by_tok]
    torch.save({"rows": rows, "split": "val", "hidden_size": hidden, "horizon": 6}, args.out)
    print(f"[merge] {args.out} n={len(rows)}/{len(order)} missing={len(missing)}", flush=True)
    if missing[:8]:
        print("[missing sample]", missing[:8], flush=True)


if __name__ == "__main__":
    main()
