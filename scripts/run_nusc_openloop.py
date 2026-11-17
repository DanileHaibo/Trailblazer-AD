#!/usr/bin/env python3
"""nuScenes open-loop eval for AutoVLA + OpenDriveVLA (VAD 2Hz x 6 = 3s)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/xinhchen2/hhb/AutoVLA")

from trajspec.accept import AcceptConfig
from trajspec.adapters.autovla import AutoVLARuntime
from trajspec.adapters.opendrivevla import OpenDriveLMRuntime
from trajspec.b2d_data import l2_horizon
from trajspec.eval_loop import collect_and_train, eval_split
from trajspec.nusc_data import TRAIN_PKL, VAL_PKL, load_infos, subsample_valid

QWEN = Path("/data2/hf_overlap_models/Qwen2.5-VL-3B-Instruct")
AUTOVLA_CKPT = Path("/data2/hf_overlap_models/AutoVLA/AutoVLA_PDMS_89.ckpt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-n", type=int, default=80)
    p.add_argument("--val-n", type=int, default=200)
    p.add_argument("--train-stride", type=int, default=40)
    p.add_argument("--val-stride", type=int, default=30)
    p.add_argument("--train-steps", type=int, default=1200)
    p.add_argument("--lon-m", type=float, default=0.5)
    p.add_argument("--lat-m", type=float, default=0.3)
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--models", default="autovla,opendrive")
    p.add_argument("--out", default=str(ROOT / "results" / "nusc_openloop.json"))
    args = p.parse_args()

    print("[data] loading nuScenes VAD infos", flush=True)
    val_infos = load_infos(VAL_PKL)
    train_infos = load_infos(TRAIN_PKL)
    val_samples = subsample_valid(val_infos, args.val_stride, args.val_n, start=10)
    train_samples = subsample_valid(train_infos, args.train_stride, args.train_n, start=10)
    print(f"[data] nusc train={len(train_samples)} val={len(val_samples)} val_infos={len(val_infos)}", flush=True)
    if not val_samples:
        raise SystemExit("no valid nuScenes val samples")

    want = {x.strip() for x in args.models.split(",") if x.strip()}
    accept_cfg = AcceptConfig(lon_m=args.lon_m, lat_m=args.lat_m)
    reports = {}
    notes = [
        "nuScenes val, VAD temporal infos: 2 Hz x 6 waypoints = 3 s.",
        "ST-P3 L2 is mean-upto-t; UniAD L2 is displacement at exactly 1s/2s/3s.",
        "AutoVLA uses official 3-cam x 4-frame visual AR (PDMS checkpoint).",
        "OpenDriveVLA-0.5B here uses the official waypoint AR LM with nuScenes ego/history/command text from VAD infos. UniAD scene/track/map tokens are still not injected (needs the mmcv/LLaVA stack).",
        "GT is cumsum of VAD gt_ego_fut_trajs, mapped to +X forward / +Y left.",
    ]

    if args.smoke:
        sample = val_samples[0]
        print("[smoke] token", sample["token"], flush=True)
        print("[smoke] gt", np.round(sample["gt"], 3).tolist(), flush=True)
        print("[smoke] hist", np.round(sample["hist"], 3).tolist(), flush=True)
        print("[smoke] cmd", sample["mission_goal"], "v", sample["vehicle_velocity"], flush=True)
        if "autovla" in want and AUTOVLA_CKPT.exists():
            runner = AutoVLARuntime(device="cuda")
            t0 = time.time()
            pred = runner.predict(sample)
            print(f"[smoke] AutoVLA {time.time()-t0:.1f}s pred={np.round(pred[:6], 3).tolist()}", flush=True)
            print("[smoke] AutoVLA L2", l2_horizon(pred, sample["gt"]), flush=True)
            del runner
            torch.cuda.empty_cache()
        if "opendrive" in want:
            runner = OpenDriveLMRuntime(device="cuda")
            t0 = time.time()
            pred = runner.predict(sample)
            print(f"[smoke] OpenDrive {time.time()-t0:.1f}s pred={np.round(pred[:6], 3).tolist()}", flush=True)
            print("[smoke] OpenDrive L2", l2_horizon(pred, sample["gt"]), flush=True)
        return

    if "autovla" in want and AUTOVLA_CKPT.exists() and QWEN.exists():
        runner = AutoVLARuntime(device="cuda")
        head, losses = collect_and_train(runner, train_samples, steps=args.train_steps, supervise="teacher")
        torch.save(
            {"state_dict": head.state_dict(), "hidden_size": runner.hidden_size},
            ROOT / "results" / "draft_autovla_nusc.pt",
        )
        summary, rows = eval_split("autovla", runner, head, val_samples, accept_cfg)
        summary["final_train_loss"] = round(float(np.mean(losses[-50:])), 5)
        reports["AutoVLA"] = {"summary": summary, "samples": rows[:12]}
        del runner
        torch.cuda.empty_cache()

    if "opendrive" in want:
        runner = OpenDriveLMRuntime(device="cuda")
        head, losses = collect_and_train(runner, train_samples, steps=args.train_steps, supervise="teacher")
        torch.save(
            {"state_dict": head.state_dict(), "hidden_size": runner.hidden_size},
            ROOT / "results" / "draft_opendrive_nusc.pt",
        )
        summary, rows = eval_split("opendrivevla", runner, head, val_samples, accept_cfg)
        summary["final_train_loss"] = round(float(np.mean(losses[-50:])), 5)
        reports["OpenDriveVLA"] = {"summary": summary, "samples": rows[:12]}

    result = {
        "benchmark": "nuScenes open-loop (VAD protocol 2Hz x 6 = 3s)",
        "data": {
            "val_pkl": str(VAL_PKL),
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "n_val_infos": len(val_infos),
        },
        "accept": {"lon_m": args.lon_m, "lat_m": args.lat_m},
        "notes": notes,
        "reports": reports,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v.get("summary") for k, v in reports.items()}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
