#!/usr/bin/env python3
"""Cache Stanford ASL nuScenes waypoint VLA teacher + prefill hiddens (resumable)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.adapters.rnb_waypoints import RnbWaypointsRuntime
from trajspec.draft_head import pack_extra, pack_speed
from trajspec.nusc_data import TRAIN_PKL, VAL_PKL, load_infos, pack_sample_rnb, token_index


def iter_valid(infos, stride, limit, start=0):
    """Shard by valid-sample index: keep those with n_valid % stride == start."""
    tok2i = token_index(infos)
    n_yield = 0
    n_valid = 0
    for i in range(len(infos)):
        feat = pack_sample_rnb(infos, i, tok2i)
        if feat is None:
            continue
        take = (n_valid % max(stride, 1)) == start
        n_valid += 1
        if not take:
            continue
        yield feat
        n_yield += 1
        if limit > 0 and n_yield >= limit:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--ckpt", default="/data2/hf_overlap_models/nuscenes-waypoints-model")
    p.add_argument("--device", default="cuda")
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

    n_plan = sum(1 for _ in iter_valid(infos, args.stride, args.limit, args.start))
    print(f"[plan] start={args.start} stride={args.stride} n_shard={n_plan} already={len(rows)}", flush=True)

    runner = RnbWaypointsRuntime(ckpt=args.ckpt, device=args.device)
    n_new = 0
    t0 = time.time()
    for s in iter_valid(infos, args.stride, args.limit, args.start):
        if s["token"] in done:
            continue
        try:
            teacher, hidden = runner.predict_and_hidden(s)
        except Exception as e:
            print(f"[skip] {s['token'][:12]} {type(e).__name__}: {e}", flush=True)
            continue
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
        if n_new % 5 == 0 or n_new == 1:
            dt = max(time.time() - t0, 1e-6)
            rate = n_new / dt
            remain = (n_plan - len(rows)) / max(rate, 1e-6)
            print(
                f"[{args.split}] cached {len(rows)}/{n_plan} (+{n_new} this run) "
                f"{rate:.2f} f/s eta={remain/60:.1f}m last={s['token'][:8]}",
                flush=True,
            )
        if n_new % 20 == 0:
            torch.save({"rows": rows, "split": args.split, "hidden_size": runner.hidden_size}, out)
            print(f"[ckpt] {out} n={len(rows)}", flush=True)
    torch.save({"rows": rows, "split": args.split, "hidden_size": runner.hidden_size}, out)
    print(f"[done] {out} n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
