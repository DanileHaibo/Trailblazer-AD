"""Waypoint-level accept/reject. Continuous analog of speculative decoding.

Token-level exact match is the wrong granularity for trajectories: a JSON/BPE
mismatch would kill an otherwise valid prefix. We accept the longest prefix
whose longitudinal/lateral error stays inside explicit metre thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AcceptConfig:
    """Per-axis thresholds in metres. x is treated as longitudinal, y lateral."""

    lon_m: float = 0.35
    lat_m: float = 0.20
    # If True, also require heading error when traj has a third column (rad).
    heading_rad: float = 0.15
    use_heading: bool = False


def longest_prefix_accept(
    draft: torch.Tensor,
    teacher: torch.Tensor,
    cfg: AcceptConfig | None = None,
) -> int:
    """Return n such that draft[:n] is accepted against teacher[:n].

    Shapes: [K, 2] or [K, 3]. Comparison is elementwise on the overlapping
    length. The first violating index is rejected; everything after it is too.
    """
    cfg = cfg or AcceptConfig()
    if draft.ndim != 2 or teacher.ndim != 2:
        raise ValueError(f"expected [K, C], got {tuple(draft.shape)} vs {tuple(teacher.shape)}")
    k = min(int(draft.shape[0]), int(teacher.shape[0]))
    if k == 0:
        return 0
    d = draft[:k].float()
    t = teacher[:k].float()
    lon_ok = (d[:, 0] - t[:, 0]).abs() <= cfg.lon_m
    lat_ok = (d[:, 1] - t[:, 1]).abs() <= cfg.lat_m
    ok = lon_ok & lat_ok
    if cfg.use_heading and d.shape[1] >= 3 and t.shape[1] >= 3:
        head_err = (d[:, 2] - t[:, 2]).abs()
        head_err = torch.minimum(head_err, (2 * torch.pi - head_err).abs())
        ok = ok & (head_err <= cfg.heading_rad)
    # First False -> reject from there. All True -> accept k.
    bad = (~ok).nonzero(as_tuple=False)
    if bad.numel() == 0:
        return k
    return int(bad[0, 0].item())
