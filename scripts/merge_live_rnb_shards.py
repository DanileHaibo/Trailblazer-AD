#!/usr/bin/env python3
"""Merge dual-GPU live RnB shard JSONs into one full-val summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 2),
        "p50": round(float(np.median(a)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for p in args.shards:
        blob = json.loads(Path(p).read_text())
        rows.extend(blob.get("rows") or [])
    # stable unique by token (keep first)
    seen = set()
    uniq = []
    for r in rows:
        tok = r["token"]
        if tok in seen:
            continue
        seen.add(tok)
        uniq.append(r)
    rows = uniq

    off_ms = [r["t_official_generate_ms"] for r in rows]
    spec_ms = [r["spec"]["t_spec_e2e_ms"] for r in rows]
    accs = [r["n_accept"] for r in rows]
    mean_off = float(np.mean(off_ms))
    mean_spec = float(np.mean(spec_ms))

    def mean_l2(side, key):
        return round(float(np.mean([r[side][key] for r in rows])), 4)

    out = {
        "model": "stanfordasl/nuscenes-waypoints-model (Qwen3-VL-4B, traj-only)",
        "n": len(rows),
        "mean_n_accept": round(float(np.mean(accs)), 3),
        "accept_dist": {str(k): round(accs.count(k) / len(accs), 3) for k in range(7)},
        "frac_accept_6": round(accs.count(6) / len(accs), 3),
        "official_generate_ms": summarize(off_ms),
        "spec_e2e_ms": summarize(spec_ms),
        "prefill_ms": summarize([r["spec"]["t_prefill_ms"] for r in rows]),
        "verify_ms": summarize([r["spec"]["t_verify_ms"] for r in rows]),
        "tail_ms": summarize([r["spec"]["t_tail_ms"] for r in rows]),
        "draft_ms": summarize([r["spec"]["t_draft_ms"] for r in rows]),
        "speedup_vs_official": round(mean_off / max(mean_spec, 1e-6), 3),
        "official_stp3": {
            "l2_1s": mean_l2("official", "l2_1s"),
            "l2_2s": mean_l2("official", "l2_2s"),
            "l2_3s": mean_l2("official", "l2_3s"),
            "ade": mean_l2("official", "ade"),
        },
        "spec_stp3": {
            "l2_1s": mean_l2("spec", "l2_1s"),
            "l2_2s": mean_l2("spec", "l2_2s"),
            "l2_3s": mean_l2("spec", "l2_3s"),
            "ade": mean_l2("spec", "ade"),
        },
        "official_uniad_at": {
            "l2_at_1s": mean_l2("official", "l2_at_1s"),
            "l2_at_2s": mean_l2("official", "l2_at_2s"),
            "l2_at_3s": mean_l2("official", "l2_at_3s"),
        },
        "spec_uniad_at": {
            "l2_at_1s": mean_l2("spec", "l2_at_1s"),
            "l2_at_2s": mean_l2("spec", "l2_at_2s"),
            "l2_at_3s": mean_l2("spec", "l2_at_3s"),
        },
        "ade_spec_vs_official": round(float(np.mean([r["ade_spec_vs_official"] for r in rows])), 4),
        "shards": args.shards,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "rows"}, indent=2))
    print("[write]", args.out)


if __name__ == "__main__":
    main()
