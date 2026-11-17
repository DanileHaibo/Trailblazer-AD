#!/usr/bin/env python3
"""Where does the draft lose acceptance? Per-index / per-command / per-speed.

Loads the cached teacher hidden + rebuilt 16-d ego extras, then reports the
first-failure index distribution and the residual error per waypoint, so we know
whether the bottleneck is the prior (early points) or the horizon (late points).
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


def command_of(extra: torch.Tensor) -> np.ndarray:
    """extra layout: [v/15, cmd_right, cmd_left, cmd_straight, ...]"""
    e = extra.cpu()
    cr, cl = e[:, 1].numpy(), e[:, 2].numpy()
    out = np.full(len(e), "straight", dtype=object)
    out[cl > 0.5] = "left"
    out[cr > 0.5] = "right"
    return out


def diag(name: str, cache: Path, ckpt: Path, lon=0.5, lat=0.3):
    tokens, hs, extra, speed, ys, gt, hidden_size = load_cache(cache, refresh_extra=True)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    head = DraftHead(int(blob["hidden_size"]), horizon=6, width=512).to(DEV)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    cfg = AcceptConfig(lon_m=lon, lat_m=lat)

    n = hs.shape[0]
    accs, per_idx_err, per_idx_ok = [], np.zeros((n, 6, 2)), np.zeros((n, 6))
    with torch.no_grad():
        for i in range(0, n, 512):
            pred = head(
                hs[i : i + 512].to(DEV),
                extra=extra[i : i + 512].to(DEV),
                speed=speed[i : i + 512].to(DEV),
            ).cpu()
            tgt = ys[i : i + 512]
            per_idx_err[i : i + 512] = (pred - tgt).abs().numpy()
            per_idx_ok[i : i + 512, :] = (
                (per_idx_err[i : i + 512, :, 0] <= lon) & (per_idx_err[i : i + 512, :, 1] <= lat)
            )
            for j in range(pred.shape[0]):
                accs.append(longest_prefix_accept(pred[j], ys[i + j], cfg))

    accs = np.asarray(accs)
    cmd = command_of(extra)
    v = (extra[:, 0] * 15.0).numpy()

    # per-waypoint "would-be-valid" rate and prefix survival
    prefix_survive = np.array([(accs > k).mean() for k in range(6)])

    print(f"\n===== {name} =====")
    print(f"n={n}  mean_accept={accs.mean():.3f}  full6={np.mean(accs >= 6):.3f}")
    print(f"accept dist: {dict(zip(*np.unique(accs, return_counts=True)))}")
    print("\nper-index:  [point]  local_ok_rate  prefix_survive  mean|dx|  mean|dy|")
    for k in range(6):
        print(
            f"  k={k}     ok={per_idx_ok[:, k].mean():.3f}    survive={prefix_survive[k]:.3f}"
            f"        dx={per_idx_err[:, k, 0].mean():.3f}  dy={per_idx_err[:, k, 1].mean():.3f}"
        )

    print("\nby command:")
    for c in ("straight", "left", "right"):
        m = cmd == c
        if m.sum() == 0:
            continue
        print(f"  {c:<9} n={m.sum():<5} accept={accs[m].mean():.3f}  full6={np.mean(accs[m] >= 6):.3f}")
    # lateral error by command at each index
    print("\nby command, mean|dy| per index:")
    for c in ("straight", "left", "right"):
        m = cmd == c
        if m.sum() == 0:
            continue
        print(f"  {c:<9} " + "  ".join(f"{per_idx_err[m, k, 1].mean():.3f}" for k in range(6)))

    print("\nby speed bucket:")
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 99)):
        m = (v >= lo) & (v < hi)
        if m.sum() == 0:
            continue
        print(f"  v in [{lo},{hi}) n={m.sum():<5} accept={accs[m].mean():.3f} full6={np.mean(accs[m] >= 6):.3f}")

    return {
        "n": int(n),
        "mean_accept": float(accs.mean()),
        "full6": float(np.mean(accs >= 6)),
        "prefix_survive": prefix_survive.tolist(),
        "local_ok": per_idx_ok.mean(0).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["auto", "od", "both"])
    ap.add_argument("--lon", type=float, default=0.5)
    ap.add_argument("--lat", type=float, default=0.3)
    args = ap.parse_args()

    out = {}
    if args.which in ("auto", "both"):
        out["autovla"] = diag(
            "AutoVLA",
            ROOT / "results/cache_autovla_nusc_val.pt",
            ROOT / "results/repro2_specdec_autovla.pt",
            args.lon,
            args.lat,
        )
    if args.which in ("od", "both"):
        out["opendrive"] = diag(
            "OpenDriveVLA",
            ROOT / "results/cache_opendrive_nusc_val.pt",
            ROOT / "results/repro2_specdec_opendrive.pt",
            args.lon,
            args.lat,
        )
    print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
