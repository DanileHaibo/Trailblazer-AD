#!/usr/bin/env python3
"""Score a native or spec plan_conv jsonl with the official OpenDriveVLA metric.

Same protocol as scripts/eval_opendrive_nusc.py (PlanningMetric, official GT,
occupancy mask), but works on either a generate() jsonl (native) or a jsonl we
build from a spec run, so the two arms are scored identically.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

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
            token = str(row["id"]).replace("_trajectory", "")
            if "native_xy" in row:
                traj = row["native_xy"]
            elif "spec_xy" in row:
                traj = row["spec_xy"]
            else:
                ans = row["answer"]
                traj = retrieve_traj(ans[0] if isinstance(ans, list) else ans)
            preds[token] = np.asarray(traj, dtype=np.float32)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--occupancy", default="vad_gt_seg")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    preds = load_preds(Path(args.pred))
    print(f"[load] {len(preds)} predictions", flush=True)
    gt = pickle.load(open(GT_DIR / "gt_traj.pkl", "rb"))
    mask = pickle.load(open(GT_DIR / "gt_traj_mask.pkl", "rb"))
    occ = pickle.load(open(GT_DIR / f"{args.occupancy}.pkl", "rb"))

    metric = PlanningMetric(6)
    l2s, n, skipped = [], 0, 0
    for token, g in gt.items():
        if token not in preds:
            skipped += 1
            continue
        g = torch.tensor(g).reshape(-1, 2)
        m = torch.tensor(mask[token]).reshape(-1, 2)
        pred = torch.tensor(preds[token]).reshape(-1, 2)
        occupancy = occ[token]
        if not isinstance(occupancy, torch.Tensor):
            occupancy = torch.tensor(occupancy)
        if occupancy.ndim == 4:
            occupancy = occupancy[0]
        if g.shape[0] % 2:
            g, m, pred = g[1:], m[1:], pred[1:]
            occupancy = occupancy[1:]
        g, m, pred = g[:6], m[:6], pred[:6]
        occupancy = occupancy[:6]
        if pred.shape[0] < g.shape[0]:
            continue
        metric(pred.unsqueeze(0), g.unsqueeze(0), occupancy.unsqueeze(0), token, m.unsqueeze(0))
        l2s.append(torch.linalg.norm(pred - g, dim=-1).numpy())
        n += 1

    l2s = np.stack(l2s)
    uni = l2s[:, [1, 3, 5]].mean(0)
    stp = np.array([l2s[:, :2].mean(), l2s[:, :4].mean(), l2s[:, :6].mean()])
    scores = metric.compute()
    tag = args.tag or Path(args.pred).stem
    print(f"\n===== {tag} =====")
    print(f"scored={n} skipped={skipped}")
    print(f"UniAD L2  1/2/3s: {uni[0]:.3f} / {uni[1]:.3f} / {uni[2]:.3f}  avg={uni.mean():.3f}")
    print(f"ST-P3 L2  1/2/3s: {stp[0]:.3f} / {stp[1]:.3f} / {stp[2]:.3f}  avg={stp.mean():.3f}")
    print(
        f"official-style UniAD L2[1,3,5]: "
        f"{scores['L2'][1]:.3f} / {scores['L2'][3]:.3f} / {scores['L2'][5]:.3f}"
    )
    print(
        f"col box 1/2/3s: "
        f"{100*scores['obj_box_col'][1]:.2f} / {100*scores['obj_box_col'][3]:.2f} / "
        f"{100*scores['obj_box_col'][5]:.2f}"
    )
    out = {
        "tag": tag,
        "n": int(n),
        "skipped": int(skipped),
        "uniad_l2": [round(float(x), 4) for x in uni],
        "stp3_l2": [round(float(x), 4) for x in stp],
        "official_l2_135": [round(float(scores["L2"][k]), 4) for k in (1, 3, 5)],
        "col_box_135": [round(float(100 * scores["obj_box_col"][k]), 3) for k in (1, 3, 5)],
    }
    p = Path(args.pred).with_suffix(".score.json")
    p.write_text(json.dumps(out, indent=2))
    print(f"[write] {p}")


if __name__ == "__main__":
    main()
