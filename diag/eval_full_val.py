#!/usr/bin/env python3
"""Full nuScenes-val evaluation (6019 frames) for both teachers.

Uses the teacher hidden caches that already exist, so no 3B / UniAD rerun:
  AutoVLA      results/cache_autovla_nusc_val_full.pt   (6019)
  OpenDriveVLA results/cache_opendrive_nusc_val.pt       (6019)

Reports the paper's Table-1 quantities on the full val split, and compares the
hard splice against the seam-corrected one. Accept/speedup are unchanged by the
splice, so only L2 moves.
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
from trajspec.b2d_data import l2_horizon  # noqa: E402
from trajspec.draft_head import DraftHead  # noqa: E402
from trajspec.eval_loop import l2_uniad_at  # noqa: E402
from trajspec.sd_baselines import seam_blend  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def splice_hard(d, t, n, k=6):
    parts = []
    if n > 0:
        parts.append(d[:n])
    if n < k:
        parts.append(t[n : n + 1])
        if n + 1 < k:
            parts.append(t[n + 1 : k])
    return torch.cat(parts, 0)[:k]


def run(name, cache, ckpt, cfg):
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
    n = pred.shape[0]

    acc = np.array([longest_prefix_accept(pred[i], ys[i], cfg) for i in range(n)])
    steps = 1 + np.maximum(6 - acc - 1, 0)
    has_gt = torch.linalg.norm(gt, dim=-1).max(-1).values > 0.2
    idx = np.where(has_gt.numpy())[0]

    def collect(fn):
        st, un = [], []
        for i in idx:
            sp = fn(pred[i], ys[i], int(acc[i]))
            s = sp.numpy()
            g = gt[i].numpy()
            st.append(l2_horizon(s, g))
            un.append(l2_uniad_at(s, g))
        return (
            {k: round(float(np.mean([x[k] for x in st])), 4) for k in ("l2_1s", "l2_2s", "l2_3s", "ade")},
            {k: round(float(np.mean([x[k] for x in un])), 4) for k in ("l2_at_1s", "l2_at_2s", "l2_at_3s")},
        )

    t_st = [l2_horizon(ys[i].numpy(), gt[i].numpy()) for i in idx]
    t_un = [l2_uniad_at(ys[i].numpy(), gt[i].numpy()) for i in idx]
    teacher = (
        {k: round(float(np.mean([x[k] for x in t_st])), 4) for k in ("l2_1s", "l2_2s", "l2_3s", "ade")},
        {k: round(float(np.mean([x[k] for x in t_un])), 4) for k in ("l2_at_1s", "l2_at_2s", "l2_at_3s")},
    )

    hard = collect(lambda d, t, k: splice_hard(d, t, k))
    blen = collect(lambda d, t, k: seam_blend(d, t, k))

    res = {
        "name": name,
        "n": int(n),
        "n_with_gt": int(len(idx)),
        "mean_n_accept": round(float(acc.mean()), 3),
        "decode_speedup": round(float((6.0 / steps).mean()), 3),
        "frac_accept_ge2": round(float((acc >= 2).mean()), 3),
        "frac_accept_ge4": round(float((acc >= 4).mean()), 3),
        "frac_accept_6": round(float((acc == 6).mean()), 3),
        "accept_dist": {str(k): round(float((acc == k).mean()), 3) for k in range(7)},
        "teacher": {"stp3": teacher[0], "uniad": teacher[1]},
        "spec_hard": {"stp3": hard[0], "uniad": hard[1]},
        "spec_blend": {"stp3": blen[0], "uniad": blen[1]},
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lon", type=float, default=0.5)
    ap.add_argument("--lat", type=float, default=0.3)
    ap.add_argument("--out", default=str(ROOT / "results/full_val_6019.json"))
    args = ap.parse_args()
    cfg = AcceptConfig(lon_m=args.lon, lat_m=args.lat)

    res = {}
    res["autovla"] = run(
        "AutoVLA-3B",
        ROOT / "results/cache_autovla_nusc_val_full.pt",
        ROOT / "results/repro2_specdec_autovla.pt",
        cfg,
    )
    res["opendrive"] = run(
        "OpenDriveVLA-0.5B",
        ROOT / "results/cache_opendrive_nusc_val.pt",
        ROOT / "results/repro2_specdec_opendrive.pt",
        cfg,
    )

    hdr = f"{'teacher':<16}{'method':<12}{'accept':>8}{'spd':>7}{'full6':>8}{'ST-P3 1/2/3s':>26}{'UniAD 1/2/3s':>24}"
    print(hdr)
    print("-" * len(hdr))
    for key, r in res.items():
        print(f"{key:<16}n={r['n']}  gt_valid={r['n_with_gt']}")
        rows = [("teacher", r["teacher"]), ("spec-hard", r["spec_hard"]), ("spec-blend", r["spec_blend"])]
        for tag, d in rows:
            st, un = d["stp3"], d["uniad"]
            acc = "-" if tag == "teacher" else f"{r['mean_n_accept']:.3f}"
            spd = "-" if tag == "teacher" else f"{r['decode_speedup']:.2f}"
            f6 = "-" if tag == "teacher" else f"{r['frac_accept_6']:.3f}"
            print(
                f"{'':<16}{tag:<12}{acc:>8}{spd:>7}{f6:>8}"
                f"{st['l2_1s']:>9.3f}/{st['l2_2s']:.3f}/{st['l2_3s']:<7.3f}"
                f"{un['l2_at_1s']:>8.3f}/{un['l2_at_2s']:.3f}/{un['l2_at_3s']:<7.3f}"
            )
        print("-" * len(hdr))

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
