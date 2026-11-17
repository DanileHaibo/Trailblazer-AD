"""Toy AR teachers that use the *same serialization* as the two real VLAs.

They are not the official weights. They exist so the speculative loop can be
trained and timed without waiting on gated checkpoints / heavy conda stacks.
Real adapters live in autovla.py / opendrivevla.py and implement the same
ARTeacher protocol.
"""

from __future__ import annotations

import torch
from torch import nn

from trajspec.scenes import kinematic_trajectory
from trajspec.serialize import ActionCodebook, format_opendrive_traj, parse_opendrive_traj


class _ARKinematic(nn.Module):
    """Small AR waypoint predictor conditioned on a scene feature.

    Each decode step sees (feat, last_xy) -> next_xy. This is the sequential
    bottleneck we are trying to skip with the draft head.
    """

    def __init__(self, feat_dim: int = 4, hidden: int = 64, noise: float = 0.0):
        super().__init__()
        self.noise = noise
        self.step = nn.Sequential(
            nn.Linear(feat_dim + 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        # Bias toward "keep going forward a bit" so untrained steps are not NaN.
        nn.init.zeros_(self.step[-1].weight)
        nn.init.constant_(self.step[-1].bias[0], 0.8)
        nn.init.zeros_(self.step[-1].bias[1])

    def decode_n(
        self,
        feat: torch.Tensor,
        n: int,
        prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        last = torch.zeros(2, device=feat.device, dtype=torch.float32)
        pts = []
        if prefix is not None and prefix.numel() > 0:
            for i in range(prefix.shape[0]):
                pts.append(prefix[i, :2].float())
            last = prefix[-1, :2].float()
        for _ in range(n):
            inp = torch.cat([feat.float(), last], dim=-1)
            delta = self.step(inp)
            last = last + delta
            if self.noise > 0:
                last = last + torch.randn_like(last) * self.noise
            pts.append(last)
        return torch.stack(pts[-n:], dim=0) if n > 0 else torch.zeros(0, 2)


class OpenDriveStyleTeacher(nn.Module):
    """AR teacher that emits OpenDriveVLA textual waypoints: [(x,y), ...] x6."""

    def __init__(self, horizon: int = 6, hidden_size: int = 4, noise: float = 0.04):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.ar = _ARKinematic(feat_dim=4, hidden=64, noise=noise)

    def prefill(self, scene: dict) -> tuple[torch.Tensor, object]:
        # Prefill analog: a compact scene state. The real VLAs replace this
        # with last-layer hidden after visual+prompt prefill.
        feat = scene["feat"]
        return feat.float(), {"feat": feat}

    def teacher_waypoints(self, scene: dict, kv: object, hidden: torch.Tensor) -> torch.Tensor:
        # Match official OpenDriveVLA: serialize then parse, so verification
        # sees the same 2-decimal rounding the real model would emit.
        raw = kinematic_trajectory(scene["speed"], scene["yaw_rate"], self.horizon)
        text = format_opendrive_traj(raw)
        return parse_opendrive_traj(text, horizon=self.horizon)

    def continue_decode(
        self,
        scene: dict,
        kv: object,
        accepted: torch.Tensor,
        remaining: int,
    ) -> torch.Tensor:
        raw = kinematic_trajectory(scene["speed"], scene["yaw_rate"], self.horizon)
        text = format_opendrive_traj(raw)
        full = parse_opendrive_traj(text, horizon=self.horizon)
        start = int(accepted.shape[0])
        return full[start : start + remaining]


class AutoVLAStyleTeacher(nn.Module):
    """AR teacher that emits AutoVLA <action_i> tokens, one pose per token."""

    def __init__(self, horizon: int = 10, hidden_size: int = 4, n_bins: int = 256):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.codebook = ActionCodebook.kinematic(n_bins=n_bins)

    def prefill(self, scene: dict) -> tuple[torch.Tensor, object]:
        feat = scene["feat"]
        return feat.float(), {"feat": feat}

    def _quantized(self, scene: dict) -> torch.Tensor:
        raw = kinematic_trajectory(scene["speed"], scene["yaw_rate"], self.horizon)
        ids = self.codebook.encode(raw)
        return self.codebook.decode(ids)

    def teacher_waypoints(self, scene: dict, kv: object, hidden: torch.Tensor) -> torch.Tensor:
        return self._quantized(scene)

    def continue_decode(
        self,
        scene: dict,
        kv: object,
        accepted: torch.Tensor,
        remaining: int,
    ) -> torch.Tensor:
        full = self._quantized(scene)
        start = int(accepted.shape[0])
        return full[start : start + remaining]
