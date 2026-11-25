"""Trajectory metrics shared by the OpenDrive / R&B eval scripts.

L2 matches OpenDriveVLA's official planning eval: invalid future steps
(`gt_ego_fut_masks == 0`) contribute **zero** error instead of comparing
against a padded / zero-filled GT (which previously produced ~10 m "dirty"
outliers on ~15% of nuScenes val).
"""

from __future__ import annotations

import numpy as np


def _point_mask(gt_mask: np.ndarray | None, n: int) -> np.ndarray:
    """Per-waypoint validity in {0,1}, length n."""
    if gt_mask is None:
        return np.ones(n, dtype=np.float32)
    m = np.asarray(gt_mask, dtype=np.float32).reshape(-1)
    if m.size < n:
        m = np.pad(m, (0, n - m.size))
    m = m[:n]
    # Official OpenDrive mask is often (T, 2); treat a step valid if any dim > 0.
    if m.ndim > 1:
        m = m.reshape(n, -1).max(axis=-1)
    return (m > 0.5).astype(np.float32)


def l2_horizon(
    pred: np.ndarray,
    gt: np.ndarray,
    gt_mask: np.ndarray | None = None,
) -> dict:
    """ST-P3 style L2 at 1s / 2s / 3s over a 2 Hz, 6-waypoint horizon.

    Invalid steps (mask=0) contribute 0 to the distance, matching
    `OpenDriveVLA/drivevla/eval_share/metric.py:compute_L2`.
    """
    n = min(len(pred), len(gt), 6)
    if n == 0:
        return {"l2_1s": None, "l2_2s": None, "l2_3s": None, "ade": None, "l2_at": []}
    d = np.linalg.norm(pred[:n, :2] - gt[:n, :2], axis=-1)
    m = _point_mask(gt_mask, n)
    d = d * m

    def mean_upto(t_pts):
        k = min(t_pts, n)
        return float(d[:k].mean()) if k else None

    return {
        "l2_1s": mean_upto(2),
        "l2_2s": mean_upto(4),
        "l2_3s": mean_upto(6),
        "ade": float(d.mean()),
        "l2_at": d.tolist(),
        "gt_mask": m.tolist(),
    }


def l2_uniad_at(
    pred: np.ndarray,
    gt: np.ndarray,
    gt_mask: np.ndarray | None = None,
) -> dict:
    """UniAD-style L2 at exactly 1s/2s/3s (2 Hz waypoints index 1, 3, 5)."""
    n = min(len(pred), len(gt), 6)
    if n == 0:
        return {"l2_at_1s": None, "l2_at_2s": None, "l2_at_3s": None}
    d = np.linalg.norm(pred[:n, :2] - gt[:n, :2], axis=-1)
    m = _point_mask(gt_mask, n)
    d = d * m

    def at(i):
        return float(d[i]) if i < n else None

    return {"l2_at_1s": at(1), "l2_at_2s": at(3), "l2_at_3s": at(5)}
