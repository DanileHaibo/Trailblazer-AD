#!/usr/bin/env python3
"""Merge shard caches from parallel GPU workers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("shards", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows, seen = [], set()
    hidden = None
    split = "val"
    for path in args.shards:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        hidden = blob.get("hidden_size", hidden)
        split = blob.get("split", split)
        for r in blob["rows"]:
            if r["token"] in seen:
                continue
            seen.add(r["token"])
            rows.append(r)
        print(f"[merge] {path} +{len(blob['rows'])} unique={len(rows)}", flush=True)
    out = Path(args.out)
    torch.save({"rows": rows, "split": split, "hidden_size": hidden}, out)
    print(f"[done] {out} n={len(rows)} hidden={hidden}", flush=True)


if __name__ == "__main__":
    main()
