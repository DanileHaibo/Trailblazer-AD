#!/usr/bin/env python3
"""Merge dual-GPU OpenDrive live shard JSONs into one full-val summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(xs):
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
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
    seen = set()
    uniq = []
    for r in rows:
        tok = r["token"]
        if tok in seen:
            continue
        seen.add(tok)
        uniq.append(r)
    rows = uniq

    gen_t = np.array([r["t_official_generate_ms"] for r in rows], dtype=np.float64)
    spec_t = np.array([r["t_spec_e2e_ms"] for r in rows], dtype=np.float64)
    nacc = np.array([r["n_accept"] for r in rows], dtype=np.float64)

    def mean_side(side, key):
        return round(float(np.mean([r[side][key] for r in rows])), 4)

    out = {
        "model": "OpenDriveVLA-0.5B+UniAD",
        "mode": "live_teacher_force_verify",
        "n": len(rows),
        "start_index": 0,
        "num_shards": len(args.shards),
        "accept": {
            "lon_m": 0.5,
            "lat_m": 0.3,
            "mean_n_accept": round(float(nacc.mean()), 3) if len(rows) else 0.0,
        },
        "accept_hist": {str(i): int((nacc == i).sum()) for i in range(7)},
        "measured_ms": {
            "official_generate": summarize(gen_t),
            "spec_e2e": summarize(spec_t),
            "prefill": summarize([r["t_prefill_ms"] for r in rows]),
            "verify": summarize([r["t_verify_ms"] for r in rows]),
            "tail": summarize([r["t_tail_ms"] for r in rows]),
            "draft": summarize([r["t_draft_ms"] for r in rows]),
        },
        "speedup_vs_official_generate": round(float(gen_t.mean() / spec_t.mean()), 3)
        if len(rows) and spec_t.mean() > 0
        else 0.0,
        "mean_ade_spec_vs_teacher": round(float(np.mean([r["ade_spec_vs_teacher"] for r in rows])), 4)
        if rows
        else 0.0,
        "teacher_l2": {
            k: mean_side("teacher", k)
            for k in ("l2_1s", "l2_2s", "l2_3s", "ade", "l2_at_1s", "l2_at_2s", "l2_at_3s")
        },
        "spec_l2": {
            k: mean_side("spec", k)
            for k in ("l2_1s", "l2_2s", "l2_3s", "ade", "l2_at_1s", "l2_at_2s", "l2_at_3s")
        },
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print("[write]", args.out, f"n={out['n']}")


if __name__ == "__main__":
    main()
