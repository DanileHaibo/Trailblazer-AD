#!/usr/bin/env python3
"""Score OpenDriveVLA plan_conv.jsonl against official GPT-Driver GT (L2 + collision)."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

OD = Path("/home/xinhchen2/hhb/OpenDriveVLA")
sys.path.insert(0, str(OD / "drivevla"))
from eval_share.metric import PlanningMetric  # noqa: E402
from utils.trajectory_utils import retrieve_traj  # noqa: E402

GT_DIR = OD / "drivevla/eval_share/gt"


def load_preds(path: Path) -> dict:
    preds = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            token = row["id"].replace("_trajectory", "")
            traj = retrieve_traj(row["answer"][0] if isinstance(row["answer"], list) else row["answer"])
            preds[token] = np.asarray(traj, dtype=np.float32)
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--only-vehicle", action="store_true", default=True)
    args = p.parse_args()

    preds = load_preds(Path(args.pred))
    gt = pickle.load(open(GT_DIR / "gt_traj.pkl", "rb"))
    mask = pickle.load(open(GT_DIR / "gt_traj_mask.pkl", "rb"))
    occ_name = "planing_gt_segmentation_val" if args.only_vehicle else "vad_gt_seg.pkl"
    occ = pickle.load(open(GT_DIR / occ_name, "rb"))

    metric = PlanningMetric(6)
    n = 0
    skipped = 0
    l2s = []
    for token, g in tqdm(gt.items(), ncols=80):
        if token not in preds:
            skipped += 1
            continue
        g = torch.tensor(g).reshape(-1, 2)
        m = torch.tensor(mask[token]).reshape(-1, 2)
        pred = torch.tensor(preds[token]).reshape(-1, 2)
        occupancy = occ[token]
        if not isinstance(occupancy, torch.Tensor):
            occupancy = torch.tensor(occupancy)
        occupancy = torch.flip(occupancy, [-1])
        if occupancy.ndim == 4:
            occupancy = occupancy[0]
        if g.shape[0] % 2:
            g, m, pred = g[1:], m[1:], pred[1:]
            occupancy = occupancy[1:]
        g, m, pred = g[:6], m[:6], pred[:6]
        occupancy = occupancy[:6]
        metric(pred.unsqueeze(0), g.unsqueeze(0), occupancy.unsqueeze(0), token, m.unsqueeze(0))
        l2s.append(torch.linalg.norm(pred - g, dim=-1).numpy())
        n += 1

    l2s = np.stack(l2s) if l2s else np.zeros((0, 6))
    uni = l2s[:, [1, 3, 5]].mean(0) if n else [0, 0, 0]
    stp = np.array([l2s[:, :2].mean(), l2s[:, :4].mean(), l2s[:, :6].mean()]) if n else [0, 0, 0]
    scores = metric.compute()
    print(f"scored={n} skipped_missing_pred={skipped}")
    print(
        "UniAD L2 1/2/3s (ours): "
        f"{uni[0]:.3f} / {uni[1]:.3f} / {uni[2]:.3f}  avg={(uni[0]+uni[1]+uni[2])/3:.3f}"
    )
    print(
        "ST-P3 L2 1/2/3s (ours): "
        f"{stp[0]:.3f} / {stp[1]:.3f} / {stp[2]:.3f}  avg={stp.mean():.3f}"
    )
    print(
        "official-style UniAD L2[1,3,5]: "
        f"{scores['L2'][1]:.3f} / {scores['L2'][3]:.3f} / {scores['L2'][5]:.3f}"
    )
    print(
        "col box 1/2/3s: "
        f"{100*scores['obj_box_col'][1]:.2f} / {100*scores['obj_box_col'][3]:.2f} / {100*scores['obj_box_col'][5]:.2f}"
    )


if __name__ == "__main__":
    main()
