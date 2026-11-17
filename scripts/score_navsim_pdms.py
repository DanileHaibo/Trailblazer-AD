#!/usr/bin/env python3
"""Score dumped AutoVLA NAVSIM trajectories with official PDMS."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/data2/navsim/maps")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/data2/navsim/exp")
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", "/home/xinhchen2/hhb/AutoVLA/navsim")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/data2/navsim")
sys.path.insert(0, os.environ["NAVSIM_DEVKIT_ROOT"])

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator


def xy_to_traj(xy: np.ndarray) -> Trajectory:
    xy = np.asarray(xy, dtype=np.float32)[:, :2]
    heading = np.zeros((xy.shape[0],), dtype=np.float32)
    if xy.shape[0] >= 2:
        d = np.diff(xy, axis=0)
        heading[1:] = np.arctan2(d[:, 1], d[:, 0] + 1e-6)
        heading[0] = heading[1]
    poses = np.concatenate([xy, heading[:, None]], axis=1)
    sampling = TrajectorySampling(time_horizon=5.0, interval_length=0.5)
    return Trajectory(poses=poses, trajectory_sampling=sampling)


def score_set(name: str, tokens, trajs, loader, simulator, scorer) -> dict:
    rows = []
    n_fail = 0
    for tok, xy in tqdm(list(zip(tokens, trajs)), desc=name):
        row = {"token": tok, "valid": True}
        try:
            cache = loader.get_from_token(tok)
            result = pdm_score(
                metric_cache=cache,
                model_trajectory=xy_to_traj(xy),
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            row.update(asdict(result))
        except Exception as exc:
            n_fail += 1
            row["valid"] = False
            row["error"] = str(exc)
        rows.append(row)
    df = pd.DataFrame(rows)
    valid = df[df["valid"] == True] if "valid" in df else df
    keys = [
        "score",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "driving_direction_compliance",
    ]
    summary = {"n": int(len(df)), "n_valid": int(len(valid)), "n_fail": n_fail}
    for k in keys:
        if k in valid.columns and len(valid):
            summary[k] = round(float(valid[k].mean()), 4)
    return {"summary": summary, "rows": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajs", default="/home/xinhchen2/hhb/TrajSpecDec/results/navsim_trajs.pt")
    p.add_argument("--metric-cache", default="/data2/navsim/navtest_metric_cache")
    p.add_argument("--out", default="/home/xinhchen2/hhb/TrajSpecDec/results/navsim_pdms.json")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    try:
        blob = torch.load(args.trajs, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(args.trajs, map_location="cpu")
    tokens = list(blob["tokens"])
    if args.limit:
        tokens = tokens[: args.limit]
    loader = MetricCacheLoader(Path(args.metric_cache))
    proposal = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling=proposal)
    scorer = PDMScorer(
        proposal_sampling=proposal,
        config=PDMScorerConfig(
            progress_weight=5.0,
            ttc_weight=5.0,
            comfortable_weight=2.0,
            driving_direction_weight=0.0,
        ),
    )
    out = {
        "paper_post_rft_pdms": 89.11,
        "n_metric_cache": len(loader.tokens),
        "n_trajs": len(tokens),
    }
    for name in ("teacher", "spec", "draft"):
        arr = blob[name].numpy()
        if args.limit:
            arr = arr[: args.limit]
        print(f"[pdms] scoring {name} n={len(tokens)}", flush=True)
        rec = score_set(name, tokens, arr, loader, simulator, scorer)
        rec.pop("rows")
        out[name] = rec
        print(f"[pdms] {name} {rec['summary']}", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
