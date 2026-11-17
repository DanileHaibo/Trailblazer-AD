"""Map AutoVLA continuous waypoints to official codebook action ids.

Each codebook entry is a (6, 4, 2) vehicle-contour snippet. Rollout matches
`ActionTokenizer.rollout`: next pose is the mean of the last contour, heading
from the last contour's front/rear corners.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import torch

DEFAULT_CODEBOOK = Path(
    os.environ.get("TRAJSPEC_AUTOVLA_CODEBOOK", "/home/xinhchen2/hhb/AutoVLA/codebook_cache/agent_vocab.pkl")
)


def load_codebook(path: str | Path = DEFAULT_CODEBOOK) -> torch.Tensor:
    blob = pickle.load(open(path, "rb"))
    return torch.as_tensor(blob["token_all"]["veh"], dtype=torch.float32)  # [V, 6, 4, 2]


def _rotate(pts: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """pts [..., 2], head [] or [1] radians. Matches ActionTokenizer.transform_to_global."""
    c, s = torch.cos(head).to(pts.dtype), torch.sin(head).to(pts.dtype)
    x, y = pts[..., 0], pts[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y], dim=-1)


def one_step_all(
    codebook: torch.Tensor,
    pos: torch.Tensor,
    head: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply every codebook token from (pos, head). Returns next_pos [V,2], next_head [V]."""
    v = codebook.shape[0]
    local = codebook.reshape(v, -1, 2)  # [V, 24, 2]
    glob = _rotate(local, head) + pos.view(1, 1, 2)
    last = glob.view(v, 6, 4, 2)[:, -1]
    pos_next = last.mean(dim=1)
    diff = last[:, 0] - last[:, 3]
    head_next = torch.atan2(diff[:, 1], diff[:, 0])
    return pos_next, head_next


def xy_to_action_ids(
    xy: torch.Tensor,
    codebook: torch.Tensor,
) -> list[int]:
    """Nearest codebook token per absolute waypoint, rolling pose from origin."""
    xy = xy.reshape(-1, 2).float()
    cb = codebook.float()
    pos = xy.new_zeros(2)
    head = xy.new_zeros(())
    ids: list[int] = []
    for k in range(xy.shape[0]):
        pos_next, head_next = one_step_all(cb, pos, head)
        err = (pos_next - xy[k]).pow(2).sum(dim=-1)
        idx = int(err.argmin().item())
        ids.append(idx)
        pos = pos_next[idx]
        head = head_next[idx]
    return ids


def action_ids_to_xy(ids: list[int], codebook: torch.Tensor) -> torch.Tensor:
    """Roll ids from origin; drop the t=0 pose like AutoVLA decode."""
    cb = codebook.float()
    pos = cb.new_zeros(2)
    head = cb.new_zeros(())
    pts = [pos.clone()]
    for i in ids:
        pos_next, head_next = one_step_all(cb, pos, head)
        pos = pos_next[int(i)]
        head = head_next[int(i)]
        pts.append(pos.clone())
    traj = torch.stack(pts, dim=0)
    return traj[1:, :2] if traj.shape[0] > 1 else traj.new_zeros(0, 2)
