#!/usr/bin/env python3
"""Score cached RnB teacher on official UniAD (OpenDrive only_vehicle + Agent-Driver metric_uniad)."""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OD = Path(os.environ.get("TRAJSPEC_OPENDRIVE_REPO", "/home/xinhchen2/hhb/OpenDriveVLA")) / "drivevla"
AD = Path(os.environ.get("TRAJSPEC_AGENTDRIVER_REPO", "/tmp/RnB-EnCoRe/Agent-Driver"))
GT = OD / "eval_share/gt"
CACHE = Path(
    os.environ.get("TRAJSPEC_RNB_CACHE", str(ROOT / "results/cache_rnb_nusc_val.pt"))
)
OUT_PKL = ROOT / "results/rnb_teacher_uniad_pred.pkl"
OUT_JSON = ROOT / "results/rnb_uniad_eval.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OD))
sys.path.insert(0, str(AD))


def ego_to_od(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    return np.stack([-xy[:, 1], xy[:, 0]], axis=-1)


def dump_teacher():
    blob = torch.load(CACHE, map_location="cpu", weights_only=False)
    pred = {}
    for r in blob["rows"]:
        pred[r["token"]] = ego_to_od(r["teacher"].numpy())
    pickle.dump(pred, open(OUT_PKL, "wb"))
    print(f"[dump] {OUT_PKL} n={len(pred)}", flush=True)
    return pred


def uniad_from_scores(l2, col):
    l2 = np.asarray(l2)
    col = np.asarray(col)
    u_l2 = [float(l2[1]), float(l2[3]), float(l2[5])]
    u_col = [float(col[1] * 100), float(col[3] * 100), float(col[5] * 100)]
    return {
        "l2": [round(x, 4) for x in u_l2],
        "l2_avg": round(sum(u_l2) / 3, 4),
        "col": [round(x, 4) for x in u_col],
        "col_avg": round(sum(u_col) / 3, 4),
    }


def main():
    pred = dump_teacher()
    paper = {
        "source": "R&B-EnCoRe Table IV No Reasoning (UniAD)",
        "l2": [0.22, 0.62, 1.34],
        "l2_avg": 0.72,
        "col": [0.10, 0.25, 1.11],
        "col_avg": 0.49,
    }

    print("\n=== OpenDrive official UniAD (only_vehicle occupancy) ===", flush=True)
    from eval_share.evaluation import planning_evaluation

    od_pred = {f"{k}_trajectory": v for k, v in pred.items()}
    planning_evaluation(od_pred, subset=None, only_vehicle=True)

    print("\n=== Agent-Driver --metric uniad (OpenDrive GT as uniad_gt_seg) ===", flush=True)
    import types
    from eval_share.metric import Metric as _Metric

    pl = types.ModuleType("pytorch_lightning")
    plm = types.ModuleType("pytorch_lightning.metrics")
    plmm = types.ModuleType("pytorch_lightning.metrics.metric")
    plmm.Metric = _Metric
    sys.modules.setdefault("pytorch_lightning", pl)
    sys.modules.setdefault("pytorch_lightning.metrics", plm)
    sys.modules["pytorch_lightning.metrics.metric"] = plmm
    metrics = ROOT / "results/agentdriver_metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    mapping = {
        "gt_traj.pkl": GT / "gt_traj.pkl",
        "gt_traj_mask.pkl": GT / "gt_traj_mask.pkl",
        "uniad_gt_seg.pkl": GT / "planing_gt_segmentation_val",
        "stp3_gt_seg.pkl": GT / "vad_gt_seg.pkl",
    }
    for name, src in mapping.items():
        dst = metrics / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

    from agentdriver.evaluation.evaluation import planning_evaluation as ad_eval

    class C:
        metric = "uniad"
        gt_folder = str(metrics)
        method = "RnB-trajonly"

    ad_eval(pred, C())

    from eval_share.metric import PlanningMetric
    import pickle as pkl

    gt = pkl.load(open(GT / "gt_traj.pkl", "rb"))
    mask = pkl.load(open(GT / "gt_traj_mask.pkl", "rb"))
    occ = pkl.load(open(GT / "planing_gt_segmentation_val", "rb"))
    metric = PlanningMetric(6)
    for token, g in gt.items():
        occupancy = occ[token]
        if not isinstance(occupancy, torch.Tensor):
            occupancy = torch.tensor(occupancy)
        occupancy = torch.flip(occupancy, [-1])
        gt_trajectory = torch.as_tensor(g, dtype=torch.float32)
        gt_traj_mask = torch.as_tensor(mask[token], dtype=torch.float32)
        output = torch.as_tensor(pred[token], dtype=torch.float32).view(1, -1, 2)
        if occupancy.ndim == 3:
            occupancy = occupancy.unsqueeze(0)
        if gt_trajectory.ndim == 2:
            gt_trajectory = gt_trajectory.unsqueeze(0)
        if gt_traj_mask.ndim == 2:
            gt_traj_mask = gt_traj_mask.unsqueeze(0)
        metric(output[:, :6], gt_trajectory[:, :6], occupancy[:, :6], token, gt_traj_mask[:, :6])
    scores = metric.compute()
    l2 = scores["L2"].detach().cpu().numpy()
    col = scores["obj_box_col"].detach().cpu().numpy()
    ours = uniad_from_scores(l2, col)
    out = {"paper_no_reasoning_uniad": paper, "ours_teacher_uniad_opendrive_gt": ours, "n": len(pred)}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("\n[compare]", json.dumps(out, indent=2), flush=True)
    print("[write]", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
