#!/usr/bin/env python3
"""Headroom analysis: how much accept is lost to quantization vs. the regressor.

1. oracle      : draft = teacher, but after the same serialize/quantize path
                 the real pipeline uses. This is the ceiling of *any* draft.
2. const-vel   : pure kinematic prior, no learning. Reference floor.
3. current     : the trained DraftHead checkpoint.

If oracle ~= 6.0 the tolerance budget is not eaten by quantization and the only
way up is a better regressor. If oracle is well below 6.0, quantization itself
is the bottleneck and the draft must be trained quantize-aware.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/xinhchen2/hhb/TrajSpecDec")
sys.path.insert(0, str(ROOT))

from scripts.run_full_specdec import load_cache  # noqa: E402
from trajspec.accept import AcceptConfig, longest_prefix_accept  # noqa: E402
from trajspec.draft_head import DraftHead, bicycle_waypoints  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def summarize(name, accs):
    a = np.asarray(accs)
    print(f"  {name:<26} mean={a.mean():.3f}  full6={np.mean(a >= 6):.3f}  dist={np.bincount(a, minlength=7) / len(a)}")
    return a.mean()


def autovla_quant_roundtrip(xy: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    from trajspec.action_quant import action_ids_to_xy, xy_to_action_ids

    ids = xy_to_action_ids(xy.cpu(), codebook)
    back = action_ids_to_xy(ids, codebook)
    if back.shape[0] < xy.shape[0]:
        pad = back[-1:].repeat(xy.shape[0] - back.shape[0], 1)
        back = torch.cat([back, pad], 0)
    return back[: xy.shape[0]].float()


def od_quant_roundtrip(xy: torch.Tensor) -> torch.Tensor:
    """Mirror the 2-decimal text round trip used by the OpenDrive serializer."""
    return torch.round(xy * 100.0) / 100.0


def run(name, cache, ckpt, quantizer, lon, lat, extra_refresh=True):
    tokens, hs, extra, speed, ys, gt, hidden_size = load_cache(cache, refresh_extra=extra_refresh)
    cfg = AcceptConfig(lon_m=lon, lat_m=lat)
    n = hs.shape[0]
    print(f"\n===== {name}  n={n}  lon={lon} lat={lat} =====")

    # ---- 1. oracle under quantization ----
    accs_oracle_raw, accs_oracle_q = [], []
    q_err = []
    for i in range(n):
        t = ys[i]
        if quantizer is not None:
            tq = quantizer(t)
            q_err.append(float((tq - t).abs().mean()))
        else:
            tq = t
        # perfect draft, no quantization
        accs_oracle_raw.append(longest_prefix_accept(t, t, cfg))
        # perfect draft, after the quantization the pipeline applies
        accs_oracle_q.append(longest_prefix_accept(tq, t, cfg))
    if quantizer is not None:
        print(f"  quantization mean|err| per coord: {np.mean(q_err):.4f} m")
        summarize("oracle (no quant)", accs_oracle_raw)
        summarize("oracle (quantized)", accs_oracle_q)
    else:
        summarize("oracle", accs_oracle_raw)

    # ---- 2. constant-velocity prior, no learning ----
    v = speed.view(-1).float()
    accs_cv = []
    for i in range(n):
        pred = bicycle_waypoints(
            v[i : i + 1], torch.zeros(1), torch.zeros(1), 6, dt=0.5
        )[0]
        accs_cv.append(longest_prefix_accept(pred, ys[i], cfg))
    summarize("const-velocity prior", accs_cv)

    # ---- 3. current checkpoint ----
    if ckpt is not None and Path(ckpt).exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        head = DraftHead(int(blob["hidden_size"]), horizon=6, width=512).to(DEV)
        head.load_state_dict(blob["state_dict"])
        head.eval()
        accs = []
        with torch.no_grad():
            for i in range(0, n, 512):
                pred = head(
                    hs[i : i + 512].to(DEV), extra=extra[i : i + 512].to(DEV), speed=speed[i : i + 512].to(DEV)
                ).cpu()
                for j in range(pred.shape[0]):
                    accs.append(longest_prefix_accept(pred[j], ys[i + j], cfg))
        summarize("current DraftHead", accs)

        # current head + quantization (what the live path actually sees)
        if quantizer is not None:
            accs_q = []
            with torch.no_grad():
                for i in range(0, n, 512):
                    pred = head(
                        hs[i : i + 512].to(DEV),
                        extra=extra[i : i + 512].to(DEV),
                        speed=speed[i : i + 512].to(DEV),
                    ).cpu()
                    for j in range(pred.shape[0]):
                        accs_q.append(longest_prefix_accept(quantizer(pred[j]), ys[i + j], cfg))
            summarize("current DraftHead (quant)", accs_q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["auto", "od", "both"])
    ap.add_argument("--lon", type=float, default=0.5)
    ap.add_argument("--lat", type=float, default=0.3)
    args = ap.parse_args()

    if args.which in ("auto", "both"):
        from trajspec.action_quant import load_codebook

        cb = load_codebook()
        run(
            "AutoVLA",
            ROOT / "results/cache_autovla_nusc_val.pt",
            ROOT / "results/repro2_specdec_autovla.pt",
            lambda xy: autovla_quant_roundtrip(xy, cb),
            args.lon,
            args.lat,
        )
    if args.which in ("od", "both"):
        run(
            "OpenDriveVLA",
            ROOT / "results/cache_opendrive_nusc_val.pt",
            ROOT / "results/repro2_specdec_opendrive.pt",
            od_quant_roundtrip,
            args.lon,
            args.lat,
        )


if __name__ == "__main__":
    main()
