#!/usr/bin/env python3
"""Motivation: cheap predictors already match the VLA on a near continuous prefix.

Lightweight proposals vs frozen teacher (not vs GT):
  1) constant velocity along +X
  2) closed-form bicycle (speed / accel / yaw-rate)
  3) linear history extrapolation
  4) extras-only residual MLP on top of the bicycle (no VLA hidden)

The MLP is not a tiny planner. It never sees prefill hidden and is not trained
on GT. Last-layer residual is zero-init, so it starts as the bicycle and only
learns ego-state corrections (command / kappa / history) toward the teacher.
"""
from __future__ import annotations

import os

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.accept import AcceptConfig, longest_prefix_accept
from trajspec.draft_head import bicycle_waypoints
from trajspec.nusc_data import extras_for_tokens

HORIZON = 6
DT = 0.5
TIMES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
CFG = AcceptConfig(lon_m=0.50, lat_m=0.30)
RESULTS = ROOT / "results"


def ego_to_od(xy: torch.Tensor) -> torch.Tensor:
    """ego (+X forward, +Y left) -> UniAD/OpenDrive (x right, y forward)."""
    return torch.stack([-xy[..., 1], xy[..., 0]], dim=-1)


def rebuild(tokens: list[str]):
    found = extras_for_tokens(tokens)
    extra, speed = [], []
    hit = 0
    for t in tokens:
        if t in found:
            e, s, _ = found[t]
            extra.append(e.float().view(-1)[:16])
            speed.append(s.view(-1)[:1].float())
            hit += 1
        else:
            extra.append(torch.zeros(16))
            speed.append(torch.zeros(1))
    extra = [torch.nn.functional.pad(e, (0, 16 - e.numel()))[:16] for e in extra]
    print(f"[extra] rebuilt {hit}/{len(tokens)}", flush=True)
    return torch.stack(extra).float(), torch.stack(speed).float().view(-1)


def load_slice(cache: Path, indices: list[int]):
    blob = torch.load(cache, map_location="cpu", weights_only=False)
    rows = blob["rows"]
    tokens = [rows[i]["token"] for i in indices]
    teacher = torch.stack([rows[i]["teacher"][:6] for i in indices]).float()
    extra, speed = rebuild(tokens)
    return tokens, extra, speed, teacher


def const_vel(speed: torch.Tensor) -> torch.Tensor:
    t = torch.arange(1, HORIZON + 1, dtype=torch.float32) * DT
    x = speed.view(-1, 1).clamp(min=0.0) * t
    y = torch.zeros_like(x)
    return torch.stack([x, y], dim=-1)


def bicycle(extra: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
    v = speed.float()
    v_fwd = extra[:, 4] * 15.0
    use = torch.where(v.abs() > 1e-3, v, v_fwd)
    accel = extra[:, 6] * 3.0
    omega = extra[:, 8]
    return bicycle_waypoints(use, accel, omega, HORIZON, dt=DT)


def hist_extrap(extra: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
    hx, hy = extra[:, 10] * 10.0, extra[:, 11] * 10.0
    hx2, hy2 = extra[:, 12] * 10.0, extra[:, 13] * 10.0
    # hist[-1] is 0.5 s ago in the current ego frame (origin = now).
    vx_now = -hx / DT
    vy_now = -hy / DT
    vx_h = (hx - hx2) / DT
    vy_h = (hy - hy2) / DT
    has_hist = (hx.abs() + hy.abs()) > 0.05
    vx = torch.where(has_hist, vx_now, speed.clamp(min=0.0))
    vy = torch.where(has_hist, vy_now, torch.zeros_like(vx))
    # If two-step history looks consistent, blend.
    two = (hx2.abs() + hy2.abs()) > 0.05
    vx = torch.where(two, 0.5 * (vx + vx_h), vx)
    vy = torch.where(two, 0.5 * (vy + vy_h), vy)
    t = torch.arange(1, HORIZON + 1, dtype=torch.float32) * DT
    return torch.stack([vx.unsqueeze(1) * t, vy.unsqueeze(1) * t], dim=-1)


class EgoResidualMLP(nn.Module):
    """Bicycle + extras-only residual. Zero last layer => starts as bicycle."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, HORIZON * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, extra: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        return prior + self.net(extra.float()).view(-1, HORIZON, 2)


def train_residual(extra, prior, y, device, steps=600, bs=128):
    model = EgoResidualMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    n = extra.shape[0]
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),))
        pred = model(extra[idx].to(device), prior[idx].to(device))
        loss = (pred - y[idx].to(device)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_res(model, extra, prior, device):
    out = []
    for i in range(0, extra.shape[0], 256):
        out.append(model(extra[i : i + 256].to(device), prior[i : i + 256].to(device)).cpu())
    return torch.cat(out, dim=0)


def stats_vs_teacher(pred: torch.Tensor, teacher: torch.Tensor) -> dict:
    p = pred[:, :HORIZON, :2].float()
    t = teacher[:, :HORIZON, :2].float()
    d = (p - t).pow(2).sum(-1).sqrt()
    lon = (p[..., 0] - t[..., 0]).abs()
    lat = (p[..., 1] - t[..., 1]).abs()
    in_box = (lon <= CFG.lon_m) & (lat <= CFG.lat_m)
    accs = []
    for i in range(p.shape[0]):
        accs.append(int(longest_prefix_accept(p[i], t[i], CFG)))
    accs_t = torch.tensor(accs)
    prefix = [(accs_t >= k).float().mean().item() for k in range(1, HORIZON + 1)]
    return {
        "mean_l2_m": [round(float(d[:, k].mean()), 4) for k in range(HORIZON)],
        "median_l2_m": [round(float(d[:, k].median()), 4) for k in range(HORIZON)],
        "frac_in_box": [round(float(in_box[:, k].float().mean()), 4) for k in range(HORIZON)],
        "frac_prefix_ge": [round(float(x), 4) for x in prefix],
        "mean_n_accept": round(float(accs_t.float().mean()), 3),
        "frac_ge2": round(float((accs_t >= 2).float().mean()), 4),
        "frac_ge4": round(float((accs_t >= 4).float().mean()), 4),
        "frac_eq6": round(float((accs_t >= 6).float().mean()), 4),
    }


def maybe_od(xy: torch.Tensor, to_od: bool) -> torch.Tensor:
    return ego_to_od(xy) if to_od else xy


def run_one(name: str, cache: Path, ev_idx: list[int], tr_idx: list[int], to_od: bool, device):
    print(f"=== {name} eval={len(ev_idx)} train={len(tr_idx)} to_od={to_od} ===", flush=True)
    _, extra, speed, teacher = load_slice(cache, ev_idx)
    _, extra_tr, speed_tr, teacher_tr = load_slice(cache, tr_idx)

    cv = maybe_od(const_vel(speed), to_od)
    kin = maybe_od(bicycle(extra, speed), to_od)
    hist = maybe_od(hist_extrap(extra, speed), to_od)
    kin_tr = maybe_od(bicycle(extra_tr, speed_tr), to_od)

    print(f"[{name}] train extras-only residual MLP on teacher", flush=True)
    mlp = train_residual(extra_tr, kin_tr, teacher_tr, device)
    fused = predict_res(mlp, extra, kin, device)

    return {
        "model": name,
        "n": int(teacher.shape[0]),
        "horizon_s": TIMES,
        "accept_box_m": {"lon": CFG.lon_m, "lat": CFG.lat_m},
        "const_vel": stats_vs_teacher(cv, teacher),
        "bicycle": stats_vs_teacher(kin, teacher),
        "hist_extrap": stats_vs_teacher(hist, teacher),
        "ego_residual_mlp": stats_vs_teacher(fused, teacher),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    od_cache = RESULTS / "cache_opendrive_nusc_val.pt"
    rnb_cache = Path(os.environ.get("TRAJSPEC_RNB_CACHE", "/home/xinhchen2/hhb/TrajSpecDec/results/cache_rnb_nusc_val.pt"))
    n_od = len(torch.load(od_cache, map_location="cpu", weights_only=False)["rows"])
    od = run_one(
        "OpenDriveVLA-0.5B",
        od_cache,
        list(range(1500, min(2500, n_od))),
        list(range(2500, min(5500, n_od))),
        to_od=False,  # cache[1500:] is already ego; mixed lidar prefix is skipped
        device=device,
    )
    rnb = None
    if rnb_cache.exists():
        n_rnb = len(torch.load(rnb_cache, map_location="cpu", weights_only=False)["rows"])
        rnb = run_one(
            "R&B-EnCoRe-4B",
            rnb_cache,
            list(range(1000)),
            list(range(1000, min(5000, n_rnb))),
            to_od=False,
            device=device,
        )
    out = {
        "note": (
            "Lightweight predictors vs frozen VLA teacher on 1000 frames. "
            "Accept = longest prefix inside 0.50 m lon / 0.30 m lat. "
            "MLP is extras-only residual on the bicycle, trained on teacher "
            "(disjoint split), no prefill hidden, not vs GT."
        ),
        "opendrive": od,
    }
    if rnb is not None:
        out["rnb"] = rnb
    path = RESULTS / "motivation_near_prefix.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    slim = {
        k: {
            p: {kk: v[p][kk] for kk in ("mean_l2_m", "frac_in_box", "frac_prefix_ge", "mean_n_accept", "frac_ge2")}
            for p in ("const_vel", "bicycle", "hist_extrap", "ego_residual_mlp")
        }
        for k, v in [(("opendrive", od))] + ([("rnb", rnb)] if rnb is not None else [])
    }
    print(json.dumps(slim, indent=2), flush=True)
    print("[write]", path, flush=True)


if __name__ == "__main__":
    main()
