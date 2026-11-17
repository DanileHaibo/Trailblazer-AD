#!/usr/bin/env python3
"""Cache AutoVLA teacher trajectories + prefill hiddens on NAVSIM navtest."""
from __future__ import annotations

import argparse
import json
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
from trajspec.navsim_data import JSON_DIR, SENSOR_DIR, json_to_autovla_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", type=Path, default=JSON_DIR)
    p.add_argument("--sensor-root", type=Path, default=SENSOR_DIR)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "cache_autovla_navtest.pt")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperature", type=float, default=0.01, help="official AutoVLA eval uses 0.01; 0 = greedy")
    p.add_argument("--official-nuplan-cams", action="store_true")
    args = p.parse_args()

    files = sorted(args.json_dir.glob("*.json"))
    files = files[args.start :: args.stride]
    if args.limit:
        files = files[: args.limit]
    print(f"[navtest] {len(files)} json files from {args.json_dir}", flush=True)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out.exists():
        blob = torch.load(out, map_location="cpu", weights_only=False)
        rows = blob["rows"]
        print(f"[resume] {len(rows)} already cached", flush=True)
    done = {r["token"] for r in rows}

    temp = None if args.temperature <= 0 else args.temperature
    runner = AutoVLARuntime(device=args.device, temperature=temp)
    n_new = 0
    missing_img = 0
    for fp in files:
        rec = json.loads(fp.read_text())
        tok = rec["token"]
        if tok in done:
            continue
        feat = json_to_autovla_features(
            rec, sensor_root=args.sensor_root, official_nuplan_cams=args.official_nuplan_cams
        )
        img0 = Path(feat["sensor_data_path"]) / feat["images"]["front_camera"][-1]
        if not img0.exists():
            missing_img += 1
            if missing_img <= 3:
                print(f"[skip missing image] {img0}", flush=True)
            continue
        teacher = runner.predict(feat)
        hidden = runner.prefill_hidden(feat)
        rows.append(
            {
                "token": tok,
                "cmd": feat["driving_command"],
                "hidden": hidden.cpu().float(),
                "teacher": torch.as_tensor(teacher[: runner.horizon], dtype=torch.float32),
                "gt": torch.as_tensor(feat["gt"][: runner.horizon], dtype=torch.float32),
                "extra": pack_extra(feat),
                "speed": pack_speed(feat),
                "log_name": rec.get("log_name"),
            }
        )
        done.add(tok)
        n_new += 1
        if n_new % 10 == 0 or n_new == 1:
            print(f"[navtest] cached {len(rows)} (+{n_new}) last={tok}", flush=True)
        if n_new % 20 == 0:
            torch.save({"rows": rows, "split": "navtest", "hidden_size": runner.hidden_size, "horizon": runner.horizon}, out)
            print(f"[ckpt] {out} n={len(rows)}", flush=True)
    torch.save({"rows": rows, "split": "navtest", "hidden_size": runner.hidden_size, "horizon": 10}, out)
    print(f"[done] {out} n={len(rows)} missing_img={missing_img}", flush=True)


if __name__ == "__main__":
    main()
