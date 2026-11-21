"""Trajectory metrics shared by the OpenDrive / R&B eval scripts."""

from __future__ import annotations

import numpy as np


def l2_horizon(pred: np.ndarray, gt: np.ndarray) -> dict:
    """ST-P3 style L2 at 1s / 2s / 3s over a 2 Hz, 6-waypoint horizon."""
    n = min(len(pred), len(gt), 6)
    if n == 0:
        return {"l2_1s": None, "l2_2s": None, "l2_3s": None, "ade": None}
    d = np.linalg.norm(pred[:n, :2] - gt[:n, :2], axis=-1)

    def mean_upto(t_pts):
        k = min(t_pts, n)
        return float(d[:k].mean()) if k else None

    return {
        "l2_1s": mean_upto(2),
        "l2_2s": mean_upto(4),
        "l2_3s": mean_upto(6),
        "ade": float(d.mean()),
        "l2_at": d.tolist(),
    }
