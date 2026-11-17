#!/usr/bin/env python3
"""Cache AutoVLA nuScenes teacher trajectories + prefill hiddens (resumable)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.adapters.autovla import AutoVLARuntime
from trajspec.draft_head import pack_extra, pack_speed
from trajspec.nusc_data import TRAIN_PKL, VAL_PKL, load_infos, pack_sample, token_index


def iter_valid(infos, stride, limit, start=0):
    tok2i = token_index(infos)
    i = start
    n = 0
    while i < len(infos) and (limit <= 0 or n < limit):
        feat = pack_sample(infos, i, tok2i)
        if feat is not None:
            yield feat
            n += 1
            i += stride
        else:
            i += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="0 = all valid")
    p.add_argument("--start", type=int, default=0)
    args = p.parse_args()

    pkl = TRAIN_PKL if args.split == "train" else VAL_PKL
    infos = load_infos(pkl)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out.exists():
        blob = torch.load(out, map_location="cpu", weights_only=False)
        rows = blob["rows"]
        print(f"[resume] {len(rows)} already cached", flush=True)
    done = {r["token"] for r in rows}

    runner = AutoVLARuntime(device="cuda")
    n_new = 0
    for s in iter_valid(infos, args.stride, args.limit, args.start):
        if s["token"] in done:
            continue
        teacher = runner.predict(s)
        hidden = runner.prefill_hidden(s)
        rows.append(
            {
                "token": s["token"],
                "idx": s["idx"],
                "cmd": s["driving_command"],
                "hidden": hidden.cpu().float(),
                "teacher": torch.as_tensor(teacher[:6], dtype=torch.float32),
                "gt": torch.as_tensor(s["gt"][:6], dtype=torch.float32),
                "extra": pack_extra(s),
                "speed": pack_speed(s),
            }
        )
        done.add(s["token"])
        n_new += 1
        if n_new % 10 == 0 or n_new == 1:
            print(f"[{args.split}] cached {len(rows)} (+{n_new} this run) last={s['token'][:8]}", flush=True)
        if n_new % 20 == 0:
            torch.save({"rows": rows, "split": args.split, "hidden_size": runner.hidden_size}, out)
            print(f"[ckpt] {out} n={len(rows)}", flush=True)
    torch.save({"rows": rows, "split": args.split, "hidden_size": runner.hidden_size}, out)
    print(f"[done] {out} n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
