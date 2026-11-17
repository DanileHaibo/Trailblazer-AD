#!/usr/bin/env python3
"""Run the full paper experiment suite with the seam-corrected splice.

Outputs one JSON per (teacher, seam) plus a combined summary, covering:
  main     : accept / speedup / ST-P3 + UniAD L2 vs GT     (paper Table 1)
  seam     : hard vs blend, to isolate the splice contribution
  per-index, per-command, threshold sweep are already in diag/.

Usage
  python diag/run_full_experiments.py            # both teachers, both seams
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
from trajspec.eval_loop import l2_uniad_at  # noqa: E402
from trajspec.sd_baselines import seam_blend  # noqa: E402
from trajspec.b2d_data import l2_horizon  # noqa: E402

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


def evaluate(name, cache, ckpt, cfg):
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

    n_acc = np.array([longest_prefix_accept(pred[i], ys[i], cfg) for i in range(pred.shape[0])])
    has_gt = torch.linalg.norm(gt, dim=-1).max(-1).values > 0.2

    out = {"name": name, "n": int(pred.shape[0]), "n_with_gt": int(has_gt.sum()), "seams": {}}
    for seam in ("hard", "blend"):
        rows = []
        for i in range(pred.shape[0]):
            n = int(n_acc[i])
            sp = splice_hard(pred[i], ys[i], n) if seam == "hard" else seam_blend(pred[i], ys[i], n)
            rec = {"n_accept": n, "spec_decode_steps": 1 + max(6 - n - 1, 0)}
            if has_gt[i]:
                g = gt[i].numpy()
                s = sp.numpy()
                rec.update({f"spec_{k}": v for k, v in l2_horizon(s, g).items()})
                rec.update({f"spec_uniad_{k}": v for k, v in l2_uniad_at(s, g).items()})
            rows.append(rec)

        gtr = [r for r in rows if "spec_ade" in r]
        accs = np.array([r["n_accept"] for r in rows])

        def m(key):
            return round(float(np.mean([r[key] for r in gtr])), 4) if gtr else None

        out["seams"][seam] = {
            "mean_n_accept": round(float(accs.mean()), 3),
            "decode_speedup": round(float(np.mean([6.0 / r["spec_decode_steps"] for r in rows])), 3),
            "frac_accept_ge2": round(float((accs >= 2).mean()), 3),
            "frac_accept_ge4": round(float((accs >= 4).mean()), 3),
            "frac_accept_6": round(float((accs == 6).mean()), 3),
            "accept_dist": {str(k): round(float((accs == k).mean()), 3) for k in range(7)},
            "spec_stp3": {k: m(f"spec_{k}") for k in ("l2_1s", "l2_2s", "l2_3s", "ade")},
            "spec_uniad": {k: m(f"spec_uniad_{k}") for k in ("l2_at_1s", "l2_at_2s", "l2_at_3s")},
        }

    # teacher reference (same frames, same metric)
    gtr_idx = np.where(has_gt.numpy())[0]
    t_s = [l2_horizon(ys[i].numpy(), gt[i].numpy()) for i in gtr_idx]
    t_u = [l2_uniad_at(ys[i].numpy(), gt[i].numpy()) for i in gtr_idx]
    out["teacher_stp3"] = {k: round(float(np.mean([x[k] for x in t_s])), 4) for k in ("l2_1s", "l2_2s", "l2_3s", "ade")}
    out["teacher_uniad"] = {k: round(float(np.mean([x[k] for x in t_u])), 4) for k in ("l2_at_1s", "l2_at_2s", "l2_at_3s")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lon", type=float, default=0.5)
    ap.add_argument("--lat", type=float, default=0.3)
    ap.add_argument("--out", default=str(ROOT / "results/full_experiments.json"))
    args = ap.parse_args()
    cfg = AcceptConfig(lon_m=args.lon, lat_m=args.lat)

    res = {}
    res["autovla"] = evaluate(
        "AutoVLA-3B", ROOT / "results/cache_autovla_nusc_val.pt", ROOT / "results/repro2_specdec_autovla.pt", cfg
    )
    res["opendrive"] = evaluate(
        "OpenDriveVLA-0.5B", ROOT / "results/cache_opendrive_nusc_val.pt", ROOT / "results/repro2_specdec_opendrive.pt", cfg
    )

    print(f"\n{'teacher':<20}{'seam':<8}{'accept':>8}{'spd':>7}{'full6':>8}{'ST-P3 1/2/3s':>26}{'UniAD 1/2/3s':>24}")
    print("-" * 105)
    for key, r in res.items():
        ts, tu = r["teacher_stp3"], r["teacher_uniad"]
        print(
            f"{key:<20}{'teacher':<8}{'-':>8}{'-':>7}{'-':>8}"
            f"{ts['l2_1s']:>9.3f}/{ts['l2_2s']:.3f}/{ts['l2_3s']:<7.3f}"
            f"{tu['l2_at_1s']:>8.3f}/{tu['l2_at_2s']:.3f}/{tu['l2_at_3s']:<7.3f}"
        )
        for seam in ("hard", "blend"):
            s = r["seams"][seam]
            st, su = s["spec_stp3"], s["spec_uniad"]
            print(
                f"{key:<20}{seam:<8}{s['mean_n_accept']:>8.3f}{s['decode_speedup']:>7.2f}{s['frac_accept_6']:>8.3f}"
                f"{st['l2_1s']:>9.3f}/{st['l2_2s']:.3f}/{st['l2_3s']:<7.3f}"
                f"{su['l2_at_1s']:>8.3f}/{su['l2_at_2s']:.3f}/{su['l2_at_3s']:<7.3f}"
            )
        print("-" * 105)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
