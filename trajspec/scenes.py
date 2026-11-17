"""Scene generators used by toy teachers and draft-head training."""

from __future__ import annotations

import math

import torch


def kinematic_trajectory(
    speed: float,
    yaw_rate: float,
    horizon: int,
    dt: float = 0.5,
) -> torch.Tensor:
    """Constant-speed, constant-yaw-rate bicycle, origin at (0,0), heading 0."""
    x = y = heading = 0.0
    pts = []
    for _ in range(horizon):
        heading += yaw_rate * dt
        x += speed * dt * math.cos(heading)
        y += speed * dt * math.sin(heading)
        pts.append((x, y))
    return torch.tensor(pts, dtype=torch.float32)


def sample_scenes(n: int, horizon: int, seed: int = 0) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    scenes = []
    for _ in range(n):
        speed = float(torch.rand(1, generator=g) * 12.0 + 0.5)  # 0.5..12.5 m/s
        yaw = float((torch.rand(1, generator=g) - 0.5) * 0.5)  # +-0.25 rad/s
        cmd = int(torch.randint(0, 3, (1,), generator=g))  # 0 L / 1 F / 2 R
        if cmd == 0:
            yaw = -abs(yaw) - 0.05
        elif cmd == 2:
            yaw = abs(yaw) + 0.05
        else:
            yaw *= 0.15
        gt = kinematic_trajectory(speed, yaw, horizon)
        feat = torch.tensor(
            [speed / 15.0, yaw / 0.5, float(cmd) / 2.0, 1.0],
            dtype=torch.float32,
        )
        scenes.append(
            {
                "speed": speed,
                "yaw_rate": yaw,
                "cmd": cmd,
                "gt": gt,
                "feat": feat,
            }
        )
    return scenes
