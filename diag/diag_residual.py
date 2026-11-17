#!/usr/bin/env python3
"""Is the residual error structured (bias/scale) or noise?

If the draft is systematically biased (e.g. under-predicting speed, or lateral
drift that grows linearly), a closed-form correction can recover acceptance for
free. If the residual is noise, only better inputs help.

Reports, per waypoint index:
  - mean signed error dx/dy  (bias)
  - std of error             (noise)
  - best-fit linear scale on the longitudinal axis
  - the accept we would get after removing that bias
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="auto", choices=["auto", "od"])
    args = ap.parse_args()

    cfg = AcceptConfig(lon_m=0.5, lat_m=0.3)
    if args.which == "auto":
        tr = load_cache(ROOT / "results/cache_autovla_nusc_train.pt", refresh_extra=True)
        va = load_cache(ROOT / "results/cache_autovla_nusc_val.pt", refresh_extra=True)
        ckpt = ROOT / "results/repro2_specdec_autovla.pt"
    else:
        tr = load_cache(ROOT / "results/cache_opendrive_nusc_val.pt", refresh_extra=True)
        va = None
        ckpt = ROOT / "results/repro2_specdec_opendrive.pt"

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    head = DraftHead(int(blob["hidden_size"]), horizon=6, width=512).to(DEV)
    head.load_state_dict(blob["state_dict"])
    head.eval()

    preds, tgts = [], []
    with torch.no_grad():
        for pack in (tr, va):
            if pack is None:
                continue
            _, hs, extra, speed, ys, _, _ = pack
            for i in range(0, hs.shape[0], 512):
                p = head(
                    hs[i : i + 512].to(DEV), extra=extra[i : i + 512].to(DEV), speed=speed[i : i + 512].to(DEV)
                ).cpu()
                preds.append(p)
                tgts.append(ys[i : i + 512])
    pred = torch.cat(preds).numpy()
    tgt = torch.cat(tgts).numpy()
    err = pred - tgt

    print(f"===== {args.which}  n={pred.shape[0]} (train+val pooled) =====")
    print("idx   mean dx   mean dy   std dx   std dy   |dx|    |dy|")
    for k in range(6):
        print(
            f" {k}   {err[:, k, 0].mean():+8.4f}  {err[:, k, 1].mean():+8.4f}  "
            f"{err[:, k, 0].std():7.4f}  {err[:, k, 1].std():7.4f}  "
            f"{np.abs(err[:, k, 0]).mean():6.3f}  {np.abs(err[:, k, 1]).mean():6.3f}"
        )

    print("\nlongitudinal scale fit (tgt_x ~ a * pred_x):")
    for k in range(6):
        a = float((pred[:, k, 0] * tgt[:, k, 0]).sum() / max((pred[:, k, 0] ** 2).sum(), 1e-9))
        print(f"  k={k}  a={a:.4f}")

    # how much does removing the per-index bias buy?
    for name, adj in (
        ("raw", np.zeros((6, 2))),
        ("minus bias", -err.mean(axis=0)),
    ):
        accs = [longest_prefix_accept(torch.tensor(pred[i] + adj), torch.tensor(tgt[i]), cfg) for i in range(pred.shape[0])]
        print(f"\n  accept after '{name}': {np.mean(accs):.4f}  full6={np.mean(np.array(accs)>=6):.4f}")

    # best global longitudinal scale on val
    print("\n  per-index scale-corrected (tgt_x ~ a_k * pred_x):")
    adj = np.zeros((6, 2))
    for k in range(6):
        a = (pred[:, k, 0] * tgt[:, k, 0]).sum() / max((pred[:, k, 0] ** 2).sum(), 1e-9)
        adj[k, 0] = 0.0
        pred[:, k, 0] *= float(a)
    accs = [longest_prefix_accept(torch.tensor(pred[i]), torch.tensor(tgt[i]), cfg) for i in range(pred.shape[0])]
    print(f"  accept after scale: {np.mean(accs):.4f} full6={np.mean(np.array(accs)>=6):.4f}")


if __name__ == "__main__":
    main()
