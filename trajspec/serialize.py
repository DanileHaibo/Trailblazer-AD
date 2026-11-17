"""Serialize / parse trajectories in the two AR VLA tokenizations.

OpenDriveVLA: textual waypoints
    <traj_start>[(x1,y1),(x2,y2),...,(x6,y6)]<traj_end>

AutoVLA: physical action tokens
    each <action_i> is a codebook entry; rollout concatenates poses.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

import torch

TRAJ_START = "<traj_start>"
TRAJ_END = "<traj_end>"


def format_opendrive_traj(traj: torch.Tensor | Sequence[Sequence[float]]) -> str:
    """Match OpenDriveVLA's GPT answer: [(x,y), ...] with 2 decimals."""
    if torch.is_tensor(traj):
        pts = traj.detach().float().cpu().tolist()
    else:
        pts = [list(p) for p in traj]
    inner = ",".join(f"({x:.2f},{y:.2f})" for x, y in (p[:2] for p in pts))
    return f"{TRAJ_START}[{inner}]{TRAJ_END}"


def parse_opendrive_traj(text: str, horizon: int = 6) -> torch.Tensor:
    """Port of OpenDriveVLA retrieve_traj, returning [K, 2]."""
    text = re.sub(r"[a-zA-Z]", "", text)
    text = re.sub(r"[\u4e00-\u9fff]", "", text)
    text = re.sub(r"[<>]", "", text)
    text = re.sub(r"(\d+\.\d+)\s+(\d+\.\d+)(?=\s*[,\)])", r"\1, \2", text)
    text = re.sub(r"\s+", "", text)
    pairs = re.findall(r"[\[\(]([-\deE.+]+),\s*([-\deE.+]+)[\]\)]", text)
    coords = [(float(x), float(y)) for x, y in pairs]
    if not coords:
        return torch.zeros(horizon, 2)
    if len(coords) < horizon:
        coords = coords + [coords[-1]] * (horizon - len(coords))
    return torch.tensor(coords[:horizon], dtype=torch.float32)


def extract_opendrive_points(text: str) -> list[tuple[float, float]]:
    """Like parse_opendrive_traj but no pad/truncate — used to stop serial generate."""
    text = re.sub(r"[a-zA-Z]", "", text)
    text = re.sub(r"[\u4e00-\u9fff]", "", text)
    text = re.sub(r"[<>]", "", text)
    text = re.sub(r"(\d+\.\d+)\s+(\d+\.\d+)(?=\s*[,\)])", r"\1, \2", text)
    text = re.sub(r"\s+", "", text)
    pairs = re.findall(r"[\[\(]([-\deE.+]+),\s*([-\deE.+]+)[\]\)]", text)
    return [(float(x), float(y)) for x, y in pairs]


def format_opendrive_answer(traj: torch.Tensor | Sequence[Sequence[float]]) -> str:
    """Official generate string: [(x,y),...] with 2 decimals, no traj_start wrapper."""
    if torch.is_tensor(traj):
        pts = traj.detach().float().cpu().tolist()
    else:
        pts = [list(p) for p in traj]
    inner = ",".join(f"({x:.2f},{y:.2f})" for x, y in (p[:2] for p in pts))
    return f"[{inner}]"


def ego_to_od_xy(xy: torch.Tensor) -> torch.Tensor:
    """Ego (+X forward, +Y left) → OpenDrive/UniAD lidar (x right, y forward)."""
    xy = xy.reshape(-1, 2).float()
    return torch.stack([-xy[:, 1], xy[:, 0]], dim=-1)


def od_to_ego_xy(xy: torch.Tensor) -> torch.Tensor:
    """OpenDrive/UniAD lidar (x right, y forward) → ego (+X forward, +Y left)."""
    xy = xy.reshape(-1, 2).float()
    return torch.stack([xy[:, 1], -xy[:, 0]], dim=-1)


class ActionCodebook:
    """AutoVLA-style discrete motion snippets.

    Official AutoVLA stores (n_bins, 6, 4, 2) vehicle patches and rolls them
    out token-by-token. For the adapter/demo we keep a simpler equivalent:
    each token is a (dx, dy) ego-frame displacement, so one token == one pose.
    When an official pickle is available, `from_official_pickle` is used.
    """

    def __init__(self, table: torch.Tensor):
        # table: [V, 2] displacements in metres
        if table.ndim != 2 or table.shape[1] != 2:
            raise ValueError(f"expected [V, 2] codebook, got {tuple(table.shape)}")
        self.table = table.float()

    @property
    def vocab_size(self) -> int:
        return int(self.table.shape[0])

    @classmethod
    def kinematic(cls, n_bins: int = 256, max_speed: float = 15.0, dt: float = 0.5) -> "ActionCodebook":
        """Cover a driving envelope: speed x heading-rate grid."""
        n_speed = int(math.sqrt(n_bins))
        n_yaw = max(n_bins // n_speed, 1)
        speeds = torch.linspace(0.0, max_speed, n_speed)
        yaws = torch.linspace(-0.6, 0.6, n_yaw)
        rows = []
        for v in speeds:
            for yaw in yaws:
                dx = float(v) * dt * math.cos(float(yaw))
                dy = float(v) * dt * math.sin(float(yaw))
                rows.append((dx, dy))
        table = torch.tensor(rows[:n_bins], dtype=torch.float32)
        if table.shape[0] < n_bins:
            pad = table[-1:].repeat(n_bins - table.shape[0], 1)
            table = torch.cat([table, pad], dim=0)
        return cls(table)

    def encode(self, traj: torch.Tensor) -> list[int]:
        """Nearest-neighbour quantize a [K, 2] absolute traj into action ids.

        Absolute waypoints are converted to per-step displacements first.
        """
        if traj.ndim != 2:
            raise ValueError(traj.shape)
        prev = torch.zeros(2, dtype=traj.dtype, device=traj.device)
        ids = []
        table = self.table.to(device=traj.device, dtype=traj.dtype)
        for k in range(traj.shape[0]):
            delta = traj[k, :2] - prev
            dist = ((table - delta) ** 2).sum(dim=-1)
            ids.append(int(dist.argmin().item()))
            prev = traj[k, :2]
        return ids

    def decode(self, ids: Sequence[int]) -> torch.Tensor:
        """Roll displacements into an absolute [K, 2] trajectory from origin."""
        pos = torch.zeros(2)
        pts = []
        for i in ids:
            idx = max(0, min(int(i), self.vocab_size - 1))
            pos = pos + self.table[idx]
            pts.append(pos.clone())
        if not pts:
            return torch.zeros(0, 2)
        return torch.stack(pts, dim=0)

    def format_tokens(self, ids: Sequence[int]) -> str:
        return "".join(f"<action_{i}>" for i in ids)

    def parse_tokens(self, text: str) -> list[int]:
        return [int(x) for x in re.findall(r"<action_(\d+)>", text)]
