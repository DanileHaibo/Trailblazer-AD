#!/usr/bin/env python3
"""Re-run teacher / draft / spec ablation and explain accept vs GT L2.

Accept is vs the frozen teacher (0.50 m lon / 0.30 m lat), not vs GT.
ST-P3 L2/col are vs official occupancy GT.
"""
from __future__ import annotations

import os

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.accept import AcceptConfig, longest_prefix_accept
from trajspec.eval_loop import spec_from_draft

RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import eval_official_planning as ev  # noqa: E402


def stp3_l2(pred, ref):
    d = np.linalg.norm(pred[:6, :2] - ref[:6, :2], axis=-1)
    return {
        "l2_1s": float(d[:2].mean()),
        "l2_2s": float(d[:4].mean()),
        "l2_3s": float(d[:6].mean()),
        "at": [float(x) for x in d],
    }


def summarize_vs(pred_map, ref_map, tokens):
    rows = [stp3_l2(pred_map[t], ref_map[t]) for t in tokens if t in pred_map and t in ref_map]
    return {
        "n": len(rows),
        "l2_1s": round(float(np.mean([r["l2_1s"] for r in rows])), 4),
        "l2_2s": round(float(np.mean([r["l2_2s"] for r in rows])), 4),
        "l2_3s": round(float(np.mean([r["l2_3s"] for r in rows])), 4),
        "per_waypoint": [round(float(np.mean([r["at"][k] for r in rows])), 4) for k in range(6)],
    }


def box_stats(drafts, teachers, cfg):
    accs = []
    in_box = [[] for _ in range(6)]
    lon_err = [[] for _ in range(6)]
    lat_err = [[] for _ in range(6)]
    for d, t in zip(drafts, teachers):
        d = np.asarray(d)[:6, :2]
        t = np.asarray(t)[:6, :2]
        n = longest_prefix_accept(torch.as_tensor(d), torch.as_tensor(t), cfg)
        accs.append(int(n))
        lon = np.abs(d[:, 0] - t[:, 0])
        lat = np.abs(d[:, 1] - t[:, 1])
        ok = (lon <= cfg.lon_m) & (lat <= cfg.lat_m)
        for k in range(6):
            in_box[k].append(bool(ok[k]))
            lon_err[k].append(float(lon[k]))
            lat_err[k].append(float(lat[k]))
    n = len(accs)
    return {
        "mean_n_accept": round(float(np.mean(accs)), 3),
        "frac_ge2": round(float(np.mean([a >= 2 for a in accs])), 4),
        "frac_ge4": round(float(np.mean([a >= 4 for a in accs])), 4),
        "frac_eq6": round(float(np.mean([a >= 6 for a in accs])), 4),
        "accept_dist": {str(k): round(accs.count(k) / n, 4) for k in range(7)},
        "frac_in_box_per_waypoint": [round(float(np.mean(in_box[k])), 4) for k in range(6)],
        "mean_lon_err_per_waypoint": [round(float(np.mean(lon_err[k])), 4) for k in range(6)],
        "mean_lat_err_per_waypoint": [round(float(np.mean(lat_err[k])), 4) for k in range(6)],
        "p90_lon_err_1s": round(float(np.percentile(np.maximum(lon_err[0], lon_err[1]), 90)), 4),
        "p90_lat_err_1s": round(float(np.percentile(np.maximum(lat_err[0], lat_err[1]), 90)), 4),
    }


def run_one(name, cache, ckpt, to_od, device):
    print(f"=== dump {name} ===", flush=True)
    tokens, teacher, spec, draft, accept = ev.dump_from_cache(
        cache, ckpt, device, refresh_extra=True, to_od=to_od
    )
    # dump_from_cache maps to OD coords when to_od=True; accept was computed in native cache coords.
    cfg = AcceptConfig(lon_m=0.50, lat_m=0.30)
    drafts, teachers = [], []
    import importlib.util

    specm = importlib.util.spec_from_file_location("run_full_specdec", ROOT / "scripts" / "run_full_specdec.py")
    mod = importlib.util.module_from_spec(specm)
    specm.loader.exec_module(mod)
    toks, hs, extra, speed, ys, _, _ = mod.load_cache(cache, refresh_extra=False)
    head = ev.load_head(ckpt, device)
    with torch.no_grad():
        for i in range(hs.shape[0]):
            d = (
                head(hs[i : i + 1].to(device), extra=extra[i : i + 1].to(device), speed=speed[i : i + 1].to(device))[0]
                .cpu()
                .numpy()
            )
            drafts.append(d[:6])
            teachers.append(ys[i].numpy()[:6])
    vs_teacher = summarize_vs(
        {toks[i]: drafts[i] for i in range(len(toks))},
        {toks[i]: teachers[i] for i in range(len(toks))},
        toks,
    )
    boxes = box_stats(drafts, teachers, cfg)
    print(f"[occ] scoring {name} teacher/draft/spec", flush=True)
    occ = {
        "teacher": ev.round_report(ev.official_pair(teacher)),
        "spec": ev.round_report(ev.official_pair(spec)),
        "draft": ev.round_report(ev.official_pair(draft)),
    }
    return {
        "name": name,
        "n": len(tokens),
        "accept_from_dump": accept,
        "accept_vs_teacher": boxes,
        "draft_vs_teacher_stp3": vs_teacher,
        "occupancy_stp3": {
            k: {
                "l2": v["stp3_l2"],
                "l2_avg": v["stp3_l2_avg"],
                "col": v["stp3_col"],
                "col_avg": v["stp3_col_avg"],
                "n": v["n"],
            }
            for k, v in occ.items()
        },
        "note": (
            "occupancy L2/col is vs GT. accept and draft_vs_teacher are vs frozen teacher. "
            "ST-P3 1s is the mean of the first two 2 Hz waypoints (0.5s and 1.0s)."
        ),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    out["opendrive"] = run_one(
        "OpenDriveVLA-0.5B",
        RESULTS / "cache_opendrive_nusc_val.pt",
        RESULTS / "specdec_opendrive_heldout.pt",
        to_od=False,
        device=device,
    )
    rnb_cache = Path(os.environ.get("TRAJSPEC_RNB_CACHE", "/home/xinhchen2/hhb/TrajSpecDec/results/cache_rnb_nusc_val.pt"))
    rnb_ckpt = Path(os.environ.get("TRAJSPEC_RNB_DRAFT", "/home/xinhchen2/hhb/TrajSpecDec/results/specdec_rnb_full_retrain.pt"))
    if rnb_cache.exists() and rnb_ckpt.exists():
        out["rnb"] = run_one(
            "R&B-EnCoRe-4B",
            rnb_cache,
            rnb_ckpt,
            to_od=False,
            device=device,
        )
    path = RESULTS / "ablation_teacher_draft_rerun.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    slim = {
        k: {
            "occupancy_stp3": v["occupancy_stp3"],
            "accept_vs_teacher": {
                kk: v["accept_vs_teacher"][kk]
                for kk in (
                    "mean_n_accept",
                    "frac_ge2",
                    "frac_ge4",
                    "frac_eq6",
                    "frac_in_box_per_waypoint",
                    "mean_lon_err_per_waypoint",
                    "mean_lat_err_per_waypoint",
                )
            },
            "draft_vs_teacher_stp3": v["draft_vs_teacher_stp3"],
        }
        for k, v in out.items()
    }
    print(json.dumps(slim, indent=2), flush=True)
    print("[write]", path, flush=True)


if __name__ == "__main__":
    main()
