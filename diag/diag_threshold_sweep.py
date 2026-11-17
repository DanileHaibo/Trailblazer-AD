#!/usr/bin/env python3
"""Vectorised threshold sweep for both teachers.

Computes accept for many (lon, lat) scales in one pass by precomputing the
per-waypoint |dx|/|dy| error matrix, so a full sweep over 6019 frames takes
seconds instead of minutes.
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
from trajspec.draft_head import DraftHead  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accept_from_errors(ex, ey, lon, lat):
    """ex, ey: [N,6] abs errors. Returns [N] longest-prefix accept."""
    ok = (ex <= lon) & (ey <= lat)  # [N,6]
    # first index where not ok -> that count; if all ok -> 6
    first_bad = np.where(~ok, np.arange(6)[None, :], 6).min(axis=1)
    return first_bad


def sweep(name, cache, ckpt, scales):
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
    pred = torch.cat(preds).numpy()
    tgt = ys.numpy()
    gt_np = gt.numpy()
    ex = np.abs(pred[..., 0] - tgt[..., 0])
    ey = np.abs(pred[..., 1] - tgt[..., 1])
    has_gt = np.linalg.norm(gt_np, axis=-1).max(-1) > 0.2

    print(f"\n===== {name}  n={len(pred)} =====  (lon/lat scaled together from 0.50/0.30)")
    rows = []
    for s in scales:
        lon, lat = 0.5 * s, 0.3 * s
        acc = accept_from_errors(ex, ey, lon, lat)
        steps = 1 + np.maximum(6 - acc - 1, 0)
        # spliced trajectory = draft prefix + teacher correction + teacher suffix
        l2 = []
        for i in np.where(has_gt)[0]:
            n = int(acc[i])
            parts = []
            if n:
                parts.append(pred[i, :n])
            if n < 6:
                parts.append(tgt[i, n : n + 1])
                if n + 1 < 6:
                    parts.append(tgt[i, n + 1 : 6])
            sp = np.concatenate(parts, 0)[:6]
            l2.append(np.linalg.norm(sp - gt_np[i], axis=-1).mean())
        rec = {
            "scale": round(float(s), 2),
            "lon_m": round(lon, 3),
            "lat_m": round(lat, 3),
            "mean_accept": round(float(acc.mean()), 4),
            "full6": round(float((acc >= 6).mean()), 4),
            "ge2": round(float((acc >= 2).mean()), 4),
            "decode_speedup": round(float((6.0 / steps).mean()), 3),
            "spec_ade_m": round(float(np.mean(l2)), 4),
        }
        rows.append(rec)
        print(json.dumps(rec))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["auto", "od", "both"])
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.5])
    args = ap.parse_args()

    out = {}
    if args.which in ("auto", "both"):
        out["autovla"] = sweep(
            "AutoVLA",
            ROOT / "results/cache_autovla_nusc_val.pt",
            ROOT / "results/repro2_specdec_autovla.pt",
            args.scales,
        )
    if args.which in ("od", "both"):
        out["opendrive"] = sweep(
            "OpenDriveVLA",
            ROOT / "results/cache_opendrive_nusc_val.pt",
            ROOT / "results/repro2_specdec_opendrive.pt",
            args.scales,
        )
    Path(ROOT / "results/diag_threshold_sweep.json").write_text(json.dumps(out, indent=2))
    print("\n[write] results/diag_threshold_sweep.json")


if __name__ == "__main__":
    main()
