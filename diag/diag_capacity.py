#!/usr/bin/env python3
"""Is the ceiling capacity or information?

Train several head variants on the *train* split and report both train-split and
held-out accept. Oracle is 6.0 (verified in diag_headroom.py), so any gap is the
regressor.

  train_accept >> val_accept  -> capacity/regularisation, not enough data
  train_accept ~= val_accept    -> the last hidden simply does not carry the
                                  teacher's 3 s intent; need better inputs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/xinhchen2/hhb/TrajSpecDec")
sys.path.insert(0, str(ROOT))

from scripts.run_full_specdec import load_cache  # noqa: E402
from trajspec.accept import AcceptConfig, longest_prefix_accept  # noqa: E402
from trajspec.draft_head import DraftHead, bicycle_waypoints  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WideHead(nn.Module):
    """Same bicycle prior, but a much larger residual trunk with residual blocks.

    Kept compatible with the existing training loop signature
    (forward(hidden, extra=, speed=)).
    """

    extra_size = 16

    def __init__(self, hidden_size, horizon=6, dim=2, width=1024, depth=4, extra_size=16, dt=0.5):
        super().__init__()
        self.horizon, self.dim, self.dt, self.extra_size = horizon, dim, dt, extra_size
        self.ln = nn.LayerNorm(hidden_size)
        self.extra_mlp = nn.Sequential(nn.Linear(extra_size, 128), nn.GELU(), nn.Linear(128, 128))
        fused = hidden_size + 128
        self.proj = nn.Linear(fused, width)

        blocks = []
        for _ in range(depth):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, width * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(width * 2, width),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.kin = nn.Sequential(nn.Linear(fused, 128), nn.GELU(), nn.Linear(128, 4))
        self.out = nn.Linear(width, horizon * dim)
        self.cmd_omega = nn.Parameter(torch.tensor(0.12))
        self.kappa_omega = nn.Parameter(torch.tensor(0.08))
        self.meas_omega = nn.Parameter(torch.tensor(0.35))
        nn.init.zeros_(self.kin[-1].weight)
        nn.init.zeros_(self.kin[-1].bias)
        nn.init.xavier_uniform_(self.out.weight, gain=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, hidden, extra=None, speed=None):
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        b = hidden.shape[0]
        if extra is None:
            extra = torch.zeros(b, self.extra_size, device=hidden.device)
        elif extra.ndim == 1:
            extra = extra.unsqueeze(0)
        n = extra.shape[-1]
        if n < self.extra_size:
            extra = torch.cat([extra, extra.new_zeros(*extra.shape[:-1], self.extra_size - n)], -1)
        extra = extra[..., : self.extra_size].to(device=hidden.device, dtype=torch.float32)
        if speed is None:
            speed = extra[:, 0] * 15.0
        elif speed.ndim > 1:
            speed = speed.view(b, -1)[:, 0]
        speed = speed.to(device=hidden.device, dtype=torch.float32)

        h = self.ln(hidden.float())
        e = self.extra_mlp(extra)
        fused = torch.cat([h, e], -1)
        x = self.proj(fused)
        for blk in self.blocks:
            x = x + blk(x)

        kin = self.kin(fused)
        dv = 2.0 * torch.tanh(kin[:, 0])
        accel = 3.0 * torch.tanh(kin[:, 1])
        omega_res = 0.25 * torch.tanh(kin[:, 2])
        v_scale = 1.0 + 0.35 * torch.tanh(kin[:, 3])
        cr, cl = extra[:, 1], extra[:, 2]
        yaw_rate = extra[:, 8]
        kappa = extra[:, 9] * 3.0
        omega = (
            self.cmd_omega * (cl - cr)
            + self.kappa_omega * torch.tanh(kappa)
            + self.meas_omega * yaw_rate
            + omega_res
        )
        v = (speed + dv).clamp(min=0.0) * v_scale
        prior = bicycle_waypoints(v, accel, omega, self.horizon, dt=self.dt)
        return prior + self.out(x).view(b, self.horizon, self.dim)


def build(kind, hidden_size, width):
    if kind == "base":
        return DraftHead(hidden_size, horizon=6, width=width).to(DEV)
    if kind == "wide":
        return WideHead(hidden_size, horizon=6, width=width, depth=4).to(DEV)
    raise ValueError(kind)


def train_step_fn(head, hidden, target, opt, extra=None, speed=None):
    pred = head(hidden, extra=extra, speed=speed)
    k = min(pred.shape[1], target.shape[1])
    tgt = target[:, :k].float()
    w_list = [2.4, 2.0, 1.6, 1.2, 0.9, 0.7][:k]
    w_t = torch.tensor(w_list, device=pred.device).view(1, k, 1)
    w_xy = torch.tensor([1.15, 2.0], device=pred.device).view(1, 1, 2)
    err = nn.functional.smooth_l1_loss(pred[:, :k], tgt, beta=0.2, reduction="none")
    loss = (err * w_t * w_xy).mean()
    box = (pred[:, :k] - tgt).abs()
    hinge = torch.relu(box[:, :, 0] - 0.5) + 1.4 * torch.relu(box[:, :, 1] - 0.3)
    loss = loss + 1.6 * (hinge * w_t.view(1, k)).mean()
    d_pred = pred[:, 1:k] - pred[:, : k - 1]
    d_tgt = tgt[:, 1:k] - tgt[:, : k - 1]
    loss = loss + 0.55 * nn.functional.smooth_l1_loss(d_pred, d_tgt, beta=0.15)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(head, hs, extra, speed, ys, cfg, bs=512):
    head.eval()
    accs = []
    for i in range(0, hs.shape[0], bs):
        pred = head(
            hs[i : i + bs].to(DEV), extra=extra[i : i + bs].to(DEV), speed=speed[i : i + bs].to(DEV)
        ).cpu()
        for j in range(pred.shape[0]):
            accs.append(longest_prefix_accept(pred[j], ys[i + j], cfg))
    return np.asarray(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="auto", choices=["auto", "od"])
    ap.add_argument("--kind", default="wide")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.which == "auto":
        cache = ROOT / "results/cache_autovla_nusc_train.pt"
        vcache = ROOT / "results/cache_autovla_nusc_val.pt"
    else:
        cache = ROOT / "results/cache_opendrive_nusc_val.pt"
        vcache = None

    tr_tok, hs, extra, speed, ys, gt, hidden_size = load_cache(cache, refresh_extra=True)
    n = hs.shape[0]

    if vcache is not None:
        _, vhs, vextra, vspeed, vys, vgt, _ = load_cache(vcache, refresh_extra=True)
        tr_sel = torch.arange(n)  # full train cache is train data
        va_h, va_e, va_s, va_y = vhs, vextra, vspeed, vys
    else:
        g = torch.Generator().manual_seed(17)
        perm = torch.randperm(n, generator=g)
        nv = max(64, int(round(n * 0.15)))
        va_idx, tr_sel = perm[:nv], perm[nv:]
        va_h, va_e, va_s, va_y = hs[va_idx], extra[va_idx], speed[va_idx], ys[va_idx]

    cfg = AcceptConfig(lon_m=0.5, lat_m=0.3)
    head = build(args.kind, hidden_size, args.width)
    nparam = sum(p.numel() for p in head.parameters())
    print(
        f"[{args.which}/{args.kind}{args.tag}] trainable params={nparam/1e6:.2f}M "
        f"train={tr_sel.numel()} val={va_h.shape[0]} hidden={hidden_size}",
        flush=True,
    )

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=2e-5)

    # turn oversampling, same idea as the original loop
    e_tr = extra[tr_sel]
    turn = (e_tr[:, 1] + e_tr[:, 2]).clamp(0, 1)
    w = 1.0 + 3.5 * turn
    w = (w / w.sum()).cpu()

    t0 = time.time()
    best_val, best_state = -1.0, None
    for step in range(args.steps):
        idx = tr_sel[torch.multinomial(w, args.batch, replacement=True)]
        loss = train_step_fn(
            head,
            hs[idx].to(DEV),
            ys[idx].to(DEV),
            opt,
            extra=extra[idx].to(DEV),
            speed=speed[idx].to(DEV),
        )
        sched.step()
        if (step + 1) % 2000 == 0:
            a_tr = evaluate(head, hs[tr_sel[:2000]], extra[tr_sel[:2000]], speed[tr_sel[:2000]], ys[tr_sel[:2000]], cfg)
            a_va = evaluate(head, va_h, va_e, va_s, va_y, cfg)
            print(
                f"  step {step+1:>6} loss={loss:.4f} train_acc={a_tr.mean():.3f} "
                f"val_acc={a_va.mean():.3f} val_full6={np.mean(a_va>=6):.3f} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
            if a_va.mean() > best_val:
                best_val = float(a_va.mean())
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    a_tr = evaluate(head, hs[tr_sel[:2000]], extra[tr_sel[:2000]], speed[tr_sel[:2000]], ys[tr_sel[:2000]], cfg)
    a_va = evaluate(head, va_h, va_e, va_s, va_y, cfg)
    print(
        f"[FINAL {args.which}/{args.kind}{args.tag}] train={a_tr.mean():.3f} "
        f"val={a_va.mean():.3f} full6={np.mean(a_va>=6):.3f}",
        flush=True,
    )
    out = {
        "which": args.which,
        "kind": args.kind,
        "width": args.width,
        "params_M": round(nparam / 1e6, 3),
        "steps": args.steps,
        "train_accept": round(float(a_tr.mean()), 4),
        "val_accept": round(float(a_va.mean()), 4),
        "val_full6": round(float(np.mean(a_va >= 6)), 4),
        "val_dist": np.bincount(a_va, minlength=7).tolist(),
    }
    Path(ROOT / f"results/diag_capacity_{args.which}_{args.kind}{args.tag}.json").write_text(json.dumps(out, indent=2))
    torch.save(
        {"state_dict": head.state_dict(), "hidden_size": hidden_size, "kind": args.kind, "width": args.width},
        ROOT / f"results/diag_capacity_{args.which}_{args.kind}{args.tag}.pt",
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
