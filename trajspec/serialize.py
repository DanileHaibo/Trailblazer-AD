"""Serialize / parse OpenDriveVLA's textual waypoint tokenization.

    <traj_start>[(x1,y1),(x2,y2),...,(x6,y6)]<traj_end>

RnB/stanfordasl uses the same textual form without the wrapper tags, so the
same parser covers both teachers.
"""

from __future__ import annotations

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
