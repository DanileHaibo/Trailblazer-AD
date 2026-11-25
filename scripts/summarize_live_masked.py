#!/usr/bin/env python3
"""Re-score an existing live JSON with OpenDrive-style GT masks (no re-inference)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajspec.nusc_data import VAL_PKL, load_infos


def _mask(tok: str, tok2: dict) -> np.ndarray:
    s = tok2[tok]
    m = np.asarray(s.get("gt_ego_fut_masks", np.ones(6)), dtype=np.float32).reshape(-1)
    if m.size < 6:
        m = np.pad(m, (0, 6 - m.size))
    return m[:6]


def remask_side(rows, side_key: str, tok2: dict) -> dict:
    keys = ("l2_1s", "l2_2s", "l2_3s", "ade", "l2_at_1s", "l2_at_2s", "l2_at_3s")
    acc = {k: [] for k in keys}
    for r in rows:
        side = r[side_key]
        d = np.asarray(side.get("l2_at", []), dtype=np.float64)
        m = _mask(r["token"], tok2)
        if len(d) < 6:
            continue
        dm = d[:6] * m
        acc["l2_1s"].append(float(dm[:2].mean()))
        acc["l2_2s"].append(float(dm[:4].mean()))
        acc["l2_3s"].append(float(dm[:6].mean()))
        acc["ade"].append(float(dm.mean()))
        acc["l2_at_1s"].append(float(dm[1]))
        acc["l2_at_2s"].append(float(dm[3]))
        acc["l2_at_3s"].append(float(dm[5]))
    return {k: round(float(np.mean(v)), 4) if v else None for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("live_json")
    ap.add_argument("--teacher-key", default="teacher")
    ap.add_argument("--spec-key", default="spec")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.loads(Path(args.live_json).read_text())
    rows = d["rows"]
    tok2 = {s["token"]: s for s in load_infos(VAL_PKL)}
    if "official" in (rows[0] if rows else {}):
        args.teacher_key = "official"

    teacher = remask_side(rows, args.teacher_key, tok2)
    spec = remask_side(rows, args.spec_key, tok2)
    if "measured_ms" in d:
        off = d["measured_ms"]["official_generate"]["mean"]
        sp = d["measured_ms"]["spec_e2e"]["mean"]
        spd = d.get("speedup_vs_official_generate")
        acc = d.get("accept", {}).get("mean_n_accept")
    else:
        off = d["official_generate_ms"]["mean"]
        sp = d["spec_e2e_ms"]["mean"]
        spd = d.get("speedup_vs_official")
        acc = d.get("mean_n_accept")

    out = {
        "source": args.live_json,
        "n": len(rows),
        "metric": "L2 × gt_ego_fut_masks (invalid steps → 0; OpenDrive official style)",
        "speedup": spd,
        "mean_n_accept": acc,
        "official_generate_ms": off,
        "spec_e2e_ms": sp,
        "teacher_l2_masked": teacher,
        "spec_l2_masked": spec,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
