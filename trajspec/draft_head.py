"""Orion-style one-shot draft: prefill hidden + ego extras -> K waypoints."""

from __future__ import annotations

import math

import torch
from torch import nn


EXTRA_SIZE = 16


def _cmd_onehots(sample: dict) -> tuple[float, float, float]:
    cmd = sample.get("driving_command")
    if cmd is None:
        raw = sample.get("cmd", 1)
        if isinstance(raw, (int, float)):
            cr = 1.0 if int(raw) == 2 else 0.0
            cl = 1.0 if int(raw) == 0 else 0.0
            return cr, cl, 1.0 if cr == 0.0 and cl == 0.0 else 0.0
        cmd = str(raw)
    cmd = str(cmd).lower()
    cr = 1.0 if "right" in cmd else 0.0
    cl = 1.0 if "left" in cmd else 0.0
    return cr, cl, 1.0 if cr == 0.0 and cl == 0.0 else 0.0


def _speed_ms(sample: dict) -> float:
    v = sample.get("vehicle_velocity")
    if isinstance(v, (list, tuple)) and len(v):
        return float(v[0])
    if v is not None and not isinstance(v, (list, tuple)):
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return float(sample.get("speed", 0.0) or 0.0)


def pack_extra(sample: dict) -> torch.Tensor:
    """Ego kinematics for the draft prior. Layout (16):

    v/15, cmd_right, cmd_left, cmd_straight,
    v_fwd/15, v_left/15, can_ax/3, can_ay/3, yaw_rate, kappa/3,
    hist_x/10, hist_y/10, hist_x2/10, hist_y2/10, hist_speed/15, hist_yaw
    """
    v0 = _speed_ms(sample)
    cr, cl, cs = _cmd_onehots(sample)
    lcf = sample.get("lcf")
    v_fwd = v0
    v_left = 0.0
    can_ax = can_ay = yaw_rate = kappa = 0.0
    if lcf is not None and len(lcf) >= 9:
        # lidar: +X right, +Y forward → ego: +X forward, +Y left
        v_fwd = float(lcf[1])
        v_left = -float(lcf[0])
        can_ax = float(lcf[2])
        can_ay = float(lcf[3])
        yaw_rate = float(lcf[4])
        kappa = float(lcf[8])
        if abs(v0) < 1e-6:
            v0 = float(lcf[7]) if abs(float(lcf[7])) > 1e-6 else v_fwd

    hx = hy = hx2 = hy2 = hist_speed = hist_yaw = 0.0
    hist = sample.get("hist")
    if hist is not None and len(hist):
        hx, hy = float(hist[-1][0]), float(hist[-1][1])
        if len(hist) >= 2:
            hx2, hy2 = float(hist[-2][0]), float(hist[-2][1])
            mx, my = hx - hx2, hy - hy2
        else:
            mx, my = -hx, -hy
        hist_speed = math.sqrt(mx * mx + my * my) / 0.5
        hist_yaw = math.atan2(my, mx + 1e-6)

    return torch.tensor(
        [
            v0 / 15.0,
            cr,
            cl,
            cs,
            v_fwd / 15.0,
            v_left / 15.0,
            can_ax / 3.0,
            can_ay / 3.0,
            yaw_rate,
            kappa / 3.0,
            hx / 10.0,
            hy / 10.0,
            hx2 / 10.0,
            hy2 / 10.0,
            hist_speed / 15.0,
            hist_yaw,
        ],
        dtype=torch.float32,
    )


def pack_speed(sample: dict) -> torch.Tensor:
    return torch.tensor([_speed_ms(sample)], dtype=torch.float32)


def bicycle_waypoints(
    speed: torch.Tensor,
    accel: torch.Tensor,
    omega: torch.Tensor,
    horizon: int,
    dt: float = 0.5,
) -> torch.Tensor:
    """Constant accel + yaw-rate bicycle. speed/accel/omega: [B] → [B, K, 2]."""
    b = speed.shape[0]
    steps = torch.arange(1, horizon + 1, device=speed.device, dtype=torch.float32)
    t = steps * dt
    v = (speed.view(b, 1) + accel.view(b, 1) * t).clamp(min=0.0)
    w = omega.view(b, 1)
    th = w * t
    w_abs = w.abs()
    straight = w_abs < 1e-3
    w_safe = torch.where(straight, torch.ones_like(w), w)
    x_arc = v / w_safe * torch.sin(th)
    y_arc = v / w_safe * (1.0 - torch.cos(th))
    x_lin = v * t
    y_lin = 0.5 * speed.view(b, 1).clamp(min=0.0) * w * (t ** 2)
    x = torch.where(straight, x_lin, x_arc)
    y = torch.where(straight, y_lin, y_arc)
    return torch.stack([x, y], dim=-1)


class DraftHead(nn.Module):
    """Maps last-layer hidden + ego extras to a short-horizon trajectory.

    The backbone is a bicycle prior (speed, accel, yaw-rate). A small residual
    MLP corrects leftover error without having to re-learn "drive forward".
    """

    extra_size = EXTRA_SIZE

    def __init__(
        self,
        hidden_size: int,
        horizon: int = 6,
        dim: int = 2,
        width: int | None = None,
        extra_size: int = EXTRA_SIZE,
        dt: float = 0.5,
    ):
        super().__init__()
        self.horizon = horizon
        self.dim = dim
        self.dt = dt
        self.extra_size = extra_size
        width = width or 512
        self.ln = nn.LayerNorm(hidden_size)
        self.extra_mlp = nn.Sequential(
            nn.Linear(extra_size, 96),
            nn.GELU(),
            nn.Linear(96, 96),
        )
        fused = hidden_size + 96
        self.kin = nn.Sequential(
            nn.Linear(fused, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4),  # dv, accel, omega_res, speed_scale
        )
        self.net = nn.Sequential(
            nn.Linear(fused, width),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(width // 2, horizon * dim),
        )
        nn.init.zeros_(self.kin[-1].weight)
        nn.init.zeros_(self.kin[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.02)
        nn.init.zeros_(self.net[-1].bias)
        # command / kappa / measured yaw-rate → omega before the learned residual
        self.cmd_omega = nn.Parameter(torch.tensor(0.12))
        self.kappa_omega = nn.Parameter(torch.tensor(0.08))
        self.meas_omega = nn.Parameter(torch.tensor(0.35))

    def _pad_extra(self, extra: torch.Tensor) -> torch.Tensor:
        n = extra.shape[-1]
        if n < self.extra_size:
            pad = extra.new_zeros(*extra.shape[:-1], self.extra_size - n)
            extra = torch.cat([extra, pad], dim=-1)
        elif n > self.extra_size:
            extra = extra[..., : self.extra_size]
        return extra

    def forward(
        self,
        hidden: torch.Tensor,
        extra: torch.Tensor | None = None,
        speed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """hidden: [B, D] or [D] -> [B, K, 2] absolute waypoints."""
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        b = hidden.shape[0]
        if extra is None:
            extra = torch.zeros(b, self.extra_size, device=hidden.device, dtype=torch.float32)
        elif extra.ndim == 1:
            extra = extra.unsqueeze(0)
        extra = self._pad_extra(extra.to(device=hidden.device, dtype=torch.float32))
        if speed is None:
            speed = extra[:, 0] * 15.0
        elif speed.ndim > 1:
            speed = speed.view(b, -1)[:, 0]
        speed = speed.to(device=hidden.device, dtype=torch.float32)

        h = self.ln(hidden.float())
        e = self.extra_mlp(extra)
        fused = torch.cat([h, e], dim=-1)
        kin = self.kin(fused)
        dv = 2.0 * torch.tanh(kin[:, 0])
        accel = 3.0 * torch.tanh(kin[:, 1])
        omega_res = 0.25 * torch.tanh(kin[:, 2])
        v_scale = 1.0 + 0.35 * torch.tanh(kin[:, 3])

        cr, cl = extra[:, 1], extra[:, 2]
        yaw_rate = extra[:, 8]
        kappa = extra[:, 9] * 3.0
        omega_bias = self.cmd_omega * (cl - cr) + self.kappa_omega * torch.tanh(kappa) + self.meas_omega * yaw_rate
        omega = omega_bias + omega_res
        v = (speed.float() + dv).clamp(min=0.0) * v_scale
        prior = bicycle_waypoints(v, accel, omega, self.horizon, dt=self.dt)
        residual = self.net(fused).view(b, self.horizon, self.dim)
        return prior + residual
