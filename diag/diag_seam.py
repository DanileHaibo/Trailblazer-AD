#!/usr/bin/env python3
"""Seam-correction study for the splice point.

Current policy (paper): out = draft[:n] + teacher[n] + teacher[n+1:]
At the splice the path jumps from the draft's path onto the teacher's, which is
a C0/C1 discontinuity. This script compares candidate repairs on held-out data:

  paper      : draft[:n] + teacher[n:]                       (baseline)
  teacher    : teacher[:]                                    (upper ref for L2)
  hold_last  : draft[:n] + draft[n-1] + teacher[n+1:]         (skip the bad point)
  blend_v    : draft prefix extrapolated by its own velocity, linearly blended
               into the teacher suffix (alpha 0 -> 1 over the tail)
  offset_decay: teacher suffix shifted by the seam offset, offset decaying to 0
               at the horizon so the endpoint stays the teacher's

Reported: L2 vs GT (only GT-valid frames), mean/max step discontinuity, and how
often the repaired trajectory is better than the baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/xinhchen2/hhb/TrajSpecDec")
sys.path.insert(0, str(ROOT))

from scripts.run_full_specdec import load_cache  # noqa: E402
from trajspec.accept import AcceptConfig, longest_prefix_accept  # noqa: E402
from trajspec.draft_head import DraftHead  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def splice_paper(d, t, n, k=6):
    parts = []
    if n > 0:
        parts.append(d[:n])
    if n < k:
        parts.append(t[n : n + 1])
        if n + 1 < k:
            parts.append(t[n + 1 : k])
    return torch.cat(parts, 0)[:k]


def splice_hold_last(d, t, n, k=6):
    """Keep the accepted prefix, repeat its last point, then the teacher suffix."""
    if n <= 0:
        return t[:k].clone()
    parts = [d[:n]]
    if n < k:
        parts.append(d[n - 1 : n])
    if n + 1 < k:
        parts.append(t[n + 1 : k])
    return torch.cat(parts, 0)[:k]


def splice_blend_v(d, t, n, k=6):
    """Extrapolate the draft prefix by its own velocity, blend into teacher.

    At step j>=n: alpha_j = (j-n+1)/(k-n), out = (1-a)*extrap + a*teacher.
    Endpoint is exactly the teacher's last point.
    """
    if n <= 0:
        return t[:k].clone()
    if n >= k:
        return d[:k].clone()
    out = torch.empty(k, 2, dtype=d.dtype)
    out[:n] = d[:n]
    if n >= 2:
        v = d[n - 1] - d[n - 2]
    else:
        v = d[0] - torch.zeros(2, dtype=d.dtype)
    span = max(k - n, 1)
    for j in range(n, k):
        a = (j - n + 1) / span
        extrap = d[n - 1] + v * (j - n + 1)
        out[j] = (1 - a) * extrap + a * t[j]
    return out


def splice_offset_decay(d, t, n, k=6):
    """Shift the teacher suffix so it starts continuous, decay the shift to 0."""
    if n <= 0:
        return t[:k].clone()
    if n >= k:
        return d[:k].clone()
    out = torch.empty(k, 2, dtype=d.dtype)
    out[:n] = d[:n]
    offset = d[n - 1] - t[n - 1]  # how far the draft sits from the teacher here
    span = max(k - n, 1)
    for j in range(n, k):
        w = 1.0 - (j - n + 1) / span  # 1 -> 0
        out[j] = t[j] + offset * w
    return out


STRATS = {
    "paper": splice_paper,
    "hold_last": splice_hold_last,
    "blend_v": splice_blend_v,
    "offset_decay": splice_offset_decay,
}


def disc(spec):
    """Max single-step jump, as a smoothness proxy."""
    if spec.shape[0] < 3:
        return 0.0
    steps = (spec[1:] - spec[:-1]).norm(dim=-1)
    return float((steps[1:] - steps[:-1]).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["auto", "od", "both"])
    ap.add_argument("--lon", type=float, default=0.5)
    ap.add_argument("--lat", type=float, default=0.3)
    args = ap.parse_args()

    out = {}
    for which in (["auto", "od"] if args.which == "both" else [args.which]):
        if which == "auto":
            cache = ROOT / "results/cache_autovla_nusc_val.pt"
            ckpt = ROOT / "results/repro2_specdec_autovla.pt"
            name = "AutoVLA"
        else:
            cache = ROOT / "results/cache_opendrive_nusc_val.pt"
            ckpt = ROOT / "results/repro2_specdec_opendrive.pt"
            name = "OpenDriveVLA"

        _, hs, extra, speed, ys, gt, _ = load_cache(cache, refresh_extra=True)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        head = DraftHead(int(blob["hidden_size"]), horizon=6, width=512).to(DEV)
        head.load_state_dict(blob["state_dict"])
        head.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, hs.shape[0], 512):
                preds.append(
                    head(hs[i : i + 512].to(DEV), extra=extra[i : i + 512].to(DEV), speed=speed[i : i + 512].to(DEV)).cpu()
                )
        pred = torch.cat(preds)

        cfg = AcceptConfig(lon_m=args.lon, lat_m=args.lat)
        has_gt = torch.linalg.norm(gt, dim=-1).max(-1).values > 0.2
        n_acc = np.array([longest_prefix_accept(pred[i], ys[i], cfg) for i in range(pred.shape[0])])

        # restrict the comparison to frames where the seam actually matters
        sel = np.where(has_gt.numpy() & (n_acc < 6) & (n_acc > 0))[0]
        print(f"\n===== {name}  seam-relevant frames (0<n_accept<6): {len(sel)} / {pred.shape[0]} =====")

        l2_teacher = np.mean([float(torch.linalg.norm(ys[i] - gt[i], dim=-1).mean()) for i in sel])
        print(f"  teacher-only L2 = {l2_teacher:.4f}")

        res = {}
        base_l2 = None
        for tag, fn in STRATS.items():
            l2, dsc, better = [], [], 0
            for i in sel:
                sp = fn(pred[i], ys[i], int(n_acc[i]))
                v = float(torch.linalg.norm(sp - gt[i], dim=-1).mean())
                l2.append(v)
                dsc.append(disc(sp))
                if base_l2 is not None and v < base_l2[len(l2) - 1][1] - 1e-9:
                    better += 1
            if tag == "paper":
                base_l2 = list(zip(range(len(l2)), l2))
            res[tag] = {
                "l2": round(float(np.mean(l2)), 4),
                "disc": round(float(np.mean(dsc)), 4),
                "better_than_paper": round(better / max(len(sel), 1), 3) if base_l2 else None,
            }
            print(f"  {tag:<13} L2={res[tag]['l2']:.4f}  disc={res[tag]['disc']:.4f}  better={res[tag]['better_than_paper']}")

        # full-set view (all GT-valid frames) for the winner
        print(f"  --- full GT-valid set (n={int(has_gt.sum())}) ---")
        full = {}
        for tag, fn in STRATS.items():
            l2 = [float(torch.linalg.norm(fn(pred[i], ys[i], int(n_acc[i])) - gt[i], dim=-1).mean()) for i in np.where(has_gt.numpy())[0]]
            full[tag] = round(float(np.mean(l2)), 4)
            print(f"  {tag:<13} L2={full[tag]:.4f}")
        out[which] = {"seam_frames": len(sel), "subset": res, "full": full}

    Path(ROOT / "results/diag_seam.json").write_text(json.dumps(out, indent=2))
    print("\n[write] results/diag_seam.json")


if __name__ == "__main__":
    main()
