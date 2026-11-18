"""Map AutoVLA continuous waypoints to official codebook action ids.

Each codebook entry is a (6, 4, 2) vehicle-contour snippet. Rollout matches
`ActionTokenizer.rollout`: next pose is the mean of the last contour, heading
from the last contour's front/rear corners.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import torch

DEFAULT_CODEBOOK = Path("/home/xinhchen2/hhb/AutoVLA/codebook_cache/agent_vocab.pkl")


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
    """Roll ids from origin; drop the t=0 pose like AutoVLA decode.

    Only the selected token per step is transformed, not the whole codebook:
    the previous version rotated all 2048 entries at every step (~6.8 ms per
    10-waypoint path), which dominated the speculative-decoding bookkeeping.
    """
    cb = codebook.float()
    if not len(ids):
        return cb.new_zeros(0, 2)
    sel = cb[torch.as_tensor(list(ids), dtype=torch.long, device=cb.device)]  # [K,6,4,2]
    pos = cb.new_zeros(2)
    head = cb.new_zeros(())
    pts = []
    for k in range(sel.shape[0]):
        local = sel[k].reshape(-1, 2)  # [24,2]
        c, s = torch.cos(head), torch.sin(head)
        glob = torch.stack([c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1]], -1) + pos
        last = glob.view(6, 4, 2)[-1]
        pos = last.mean(dim=0)
        diff = last[0] - last[3]
        head = torch.atan2(diff[1], diff[0])
        pts.append(pos.clone())
    return torch.stack(pts, dim=0)[:, :2]
