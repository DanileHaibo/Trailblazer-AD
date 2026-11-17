#!/usr/bin/env python3
"""Paper-vs-repro table for TrajSpecDec.

Reads the result JSONs committed under ``results/`` (relative to this repo) and
compares them against the paper numbers.

Usage:
    python repro/compare_to_paper.py

Expected files in results/:
    specdec_autovla_heldout.json     paper, offline AutoVLA
    specdec_opendrive_heldout.json   paper, offline OpenDriveVLA
    repro2_specdec_autovla.json      repro, offline AutoVLA
    repro2_specdec_opendrive.json    repro, offline OpenDriveVLA
    repro_live_autovla.json          repro, live AutoVLA
    repro_live_opendrive.json        repro, live OpenDriveVLA
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

REF = {
    "autovla": RESULTS / "specdec_autovla_heldout.json",
    "opendrive": RESULTS / "specdec_opendrive_heldout.json",
}
REPRO_OFFLINE = {
    "autovla": RESULTS / "repro2_specdec_autovla.json",
    "opendrive": RESULTS / "repro2_specdec_opendrive.json",
}
REPRO_LIVE = {
    "autovla": RESULTS / "repro_live_autovla.json",
    "opendrive": RESULTS / "repro_live_opendrive.json",
}


def jload(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def offline_metrics(blob):
    if blob is None:
        return None
    s = blob["summary"]
    return {
        "n": s.get("n"),
        "mean_n_accept": s.get("mean_n_accept"),
        "frac_accept_ge2": s.get("frac_accept_ge2"),
        "frac_accept_6": (s.get("accept_dist") or {}).get("6"),
        "decode_speedup": s.get("mean_decode_speedup"),
        "spec_l2": [s.get("spec_stp3", {}).get(k) for k in ("l2_1s", "l2_2s", "l2_3s")],
    }


def live_metrics(blob, speed_key):
    if blob is None:
        return None
    return {
        "n": blob.get("n"),
        "mean_n_accept": (blob.get("accept") or {}).get("mean_n_accept"),
        "speedup": blob.get(speed_key),
    }


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        return "/".join("-" if x is None else f"{x:.4f}" for x in v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_table(title, rows, keys, hdr):
    print(f"=== {title} ===")
    print(hdr)
    print("-" * len(hdr))
    for model, (ref, got) in rows.items():
        if ref is None:
            print(f"(missing paper reference for {model})")
            continue
        got = got or {}
        for key in keys:
            a, b = ref.get(key), got.get(key)
            delta = "-"
            if isinstance(a, float) and isinstance(b, float):
                delta = f"{b - a:+.4f}"
            print(f"{key:<20}{model:<11}{fmt(a):<26}{fmt(b):<26}{delta:<10}")
        print("-" * len(hdr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS), help="where the result JSONs live")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    hdr = f"{'metric':<20}{'model':<11}{'paper':<26}{'repro':<26}{'delta':<10}"

    offline = {
        m: (offline_metrics(jload(p)), offline_metrics(jload(results_dir / REPRO_OFFLINE[m].name)))
        for m, p in REF.items()
    }
    print_table(
        "OFFLINE (draft head, held-out accept)",
        offline,
        ("n", "mean_n_accept", "frac_accept_ge2", "frac_accept_6", "decode_speedup", "spec_l2"),
        hdr,
    )

    live = {
        m: (
            live_metrics(jload(results_dir / f"live_specdec_{m}.json"), "speedup_vs_official_generate")
            or live_metrics(REF[m], "speedup_vs_official_generate"),
            live_metrics(jload(results_dir / REPRO_LIVE[m].name), "speedup_vs_official_generate"),
        )
        for m in REF
    }
    print()
    print_table("LIVE (CUDA-sync wall clock, n=12)", live, ("n", "mean_n_accept", "speedup"), hdr)


if __name__ == "__main__":
    main()
