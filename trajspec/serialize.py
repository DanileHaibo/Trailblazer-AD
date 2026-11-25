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

# Strict float token: rejects broken teacher outputs like '-0..0' that the
# looser `[-\deE.+]+` pattern in OpenDriveVLA's retrieve_traj still matches.
_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PAIR_RE = re.compile(rf"[\[\(]({_FLOAT}),\s*({_FLOAT})[\]\)]")


def _clean_traj_text(text: str) -> str:
    """Mirror OpenDriveVLA `retrieve_traj` cleanups, then strip whitespace.

    Critical for live eval: the teacher occasionally emits consecutive dots
    (`-0..0`) or glued decimals; without these fixes float() raises and the
    whole 6k-frame run aborts.
    """
    text = re.sub(r"[a-zA-Z]", "", text)
    text = re.sub(r"[\u4e00-\u9fff]", "", text)
    # Keep only the first decimal in runs of dots: '0..' → '0.'
    text = re.sub(r"(\d+)\.+", r"\1.", text)
    text = re.sub(r"[<>]", "", text)
    text = re.sub(r"(\d+\.\d+)\s+(\d+\.\d+)(?=\s*[,\)])", r"\1, \2", text)
    text = re.sub(r"(\d+\.\d+)-(\d+\.\d+)", r"\1, \2", text)
    text = re.sub(
        r"(\(\s*-?\d+\.\d+,\s*-?\d+\.\d+),\s*-?\d+\.\d+(\s*\))",
        r"\1\2",
        text,
    )
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(\d+\.\d+)\.(\d+)", r"\1\2", text)
    text = re.sub(r"(\d+)\.\.(\d+)", r"\1.\2", text)
    text = re.sub(r"(\d+)\.\-(\d+)", r"\1.\2", text)
    return text


def _parse_xy_pairs(text: str) -> list[tuple[float, float]]:
    text = _clean_traj_text(text)
    out: list[tuple[float, float]] = []
    for x, y in _PAIR_RE.findall(text):
        try:
            out.append((float(x), float(y)))
        except ValueError:
            continue
    return out


def format_opendrive_traj(traj: torch.Tensor | Sequence[Sequence[float]]) -> str:
    """Match OpenDriveVLA's GPT answer: [(x,y), ...] with 2 decimals."""
    if torch.is_tensor(traj):
        pts = traj.detach().float().cpu().tolist()
    else:
        pts = [list(p) for p in traj]
    inner = ",".join(f"({x:.2f},{y:.2f})" for x, y in (p[:2] for p in pts))
    return f"{TRAJ_START}[{inner}]{TRAJ_END}"


def parse_opendrive_traj(text: str, horizon: int = 6) -> torch.Tensor:
    """Port of OpenDriveVLA retrieve_traj, returning [K, 2]. Never raises on junk."""
    coords = _parse_xy_pairs(text)
    if not coords:
        return torch.zeros(horizon, 2)
    if len(coords) < horizon:
        coords = coords + [coords[-1]] * (horizon - len(coords))
    return torch.tensor(coords[:horizon], dtype=torch.float32)


def extract_opendrive_points(text: str) -> list[tuple[float, float]]:
    """Like parse_opendrive_traj but no pad/truncate — used to stop serial generate."""
    return _parse_xy_pairs(text)


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
