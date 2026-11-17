#!/usr/bin/env python3
"""Live AutoVLA specdec on NAVSIM navtest: CUDA-sync wall-clock vs official generate."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.accept import AcceptConfig
from trajspec.adapters.autovla import AutoVLARuntime
from trajspec.draft_head import DraftHead, pack_extra, pack_speed
from trajspec.navsim_data import JSON_DIR, SENSOR_DIR, json_to_autovla_features


def sync_ms(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def summarize(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 2),
        "p50": round(float(np.median(a)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
    }


def l2_navsim(pred, gt, horizon: int = 10):
    pred = np.asarray(pred)[:, :2]
    gt = np.asarray(gt)[:, :2]
    n = min(len(pred), len(gt), horizon)
    if n == 0:
        return {"ade": None}
    d = np.linalg.norm(pred[:n] - gt[:n], axis=-1)
    out = {"ade": round(float(d.mean()), 4)}
    for k in (1, 2, 3, 4, 5):
        m = min(2 * k, n)
        if m:
            out[f"l2_{k}s"] = round(float(d[:m].mean()), 4)
    return out


def cameras_exist(feat: dict) -> bool:
    root = feat.get("sensor_data_path") or ""
    for name in ("front_camera", "front_left_camera", "front_right_camera"):
        paths = feat["images"].get(name) or []
        if len(paths) < 4:
            return False
        for p in paths[:4]:
            full = os.path.join(root, p) if root and not os.path.isabs(p) else p
            if not os.path.isfile(full):
                return False
    return True


def iter_navtest(json_dir: Path, sensor_root: Path, stride: int, limit: int, start: int):
    files = sorted(json_dir.glob("*.json"))
    n = 0
    for path in files[start::stride]:
        rec = json.loads(path.read_text())
        feat = json_to_autovla_features(rec, sensor_root=sensor_root)
        if not cameras_exist(feat):
            continue
        yield feat
        n += 1
        if n >= limit:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--stride", type=int, default=800)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--json-dir", type=Path, default=JSON_DIR)
    p.add_argument("--sensor-root", type=Path, default=SENSOR_DIR)
    p.add_argument("--ckpt", default=str(ROOT / "results/specdec_autovla_navsim.pt"))
    p.add_argument("--out", default=str(ROOT / "results/live_specdec_autovla_navsim.json"))
    p.add_argument("--skip-official-generate", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    runner = AutoVLARuntime(device="cuda", temperature=None)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = blob.get("head") or blob.get("state_dict")
    head = DraftHead(
        int(blob["hidden_size"]),
        horizon=int(blob.get("horizon", args.horizon)),
        extra_size=int(blob.get("extra_size", 16)),
    ).to(device)
    head.load_state_dict(sd)
    head.eval()
    cfg = AcceptConfig(lon_m=0.5, lat_m=0.3)
    horizon = args.horizon

    rows = []
    seen = 0
    for s in iter_navtest(args.json_dir, args.sensor_root, args.stride, args.n + args.warmup, args.start):
        timed = seen >= args.warmup
        extra = pack_extra(s)
        speed = pack_speed(s)
        gt = np.asarray(s["gt"][:horizon], dtype=np.float32)

        rec = {"token": s["token"]}
        if not args.skip_official_generate:
            t_off, teacher_off = sync_ms(lambda: runner.predict(s))
            rec["t_official_generate_ms"] = round(t_off, 2)
            rec["official"] = l2_navsim(teacher_off, gt, horizon)

        t_ar, (ar_xy, ar_info) = sync_ms(lambda: runner.ar_action_generate(s, horizon=horizon))
        rec["t_ar_keep_think_ms"] = round(t_ar, 2)
        rec["ar"] = {**l2_navsim(ar_xy, gt, horizon), **{k: ar_info[k] for k in ar_info if k != "action_ids"}}

        t_spec, (spec_xy, spec_info) = sync_ms(
            lambda: runner.spec_generate(s, head, extra, speed, cfg, horizon=horizon)
        )
        rec["spec"] = {
            **l2_navsim(spec_xy, gt, horizon),
            **{
                k: spec_info[k]
                for k in spec_info
                if k
                not in (
                    "draft_xy",
                    "teacher_verify_xy",
                    "draft_ids",
                    "teacher_verify_ids",
                    "committed_ids",
                )
            },
        }
        rec["n_accept"] = spec_info["n_accept"]
        rec["t_spec_e2e_ms"] = round(t_spec, 2)
        rec["t_spec_parts_ms"] = spec_info["t_spec_e2e_ms"]
        rec["t_verify_ms"] = spec_info["t_verify_ms"]
        rec["t_tail_ms"] = spec_info["t_tail_ms"]
        rec["spec_decode_steps"] = spec_info["spec_decode_steps"]
        rec["ade_spec_vs_ar"] = round(float(np.linalg.norm(spec_xy[:horizon] - ar_xy[:horizon], axis=-1).mean()), 4)

        seen += 1
        tag = f"[{len(rows)}/{args.n}]" if timed else "[warmup]"
        print(
            f"{tag} spec={rec['t_spec_e2e_ms']:.0f}ms ar={rec['t_ar_keep_think_ms']:.0f}ms "
            f"off={rec.get('t_official_generate_ms', 0):.0f}ms think={rec['spec'].get('n_think_tokens', '?')} "
            f"accept={rec['n_accept']} ade(spec,ar)={rec['ade_spec_vs_ar']}",
            flush=True,
        )
        if timed:
            rows.append(rec)

    spec_t = np.array([r["t_spec_e2e_ms"] for r in rows])
    ar_t = np.array([r["t_ar_keep_think_ms"] for r in rows])
    nacc = np.array([r["n_accept"] for r in rows], dtype=np.float64)
    summary = {
        "model": "AutoVLA-3B",
        "dataset": "navsim-navtest",
        "mode": "live_teacher_force_verify",
        "n": len(rows),
        "horizon": horizon,
        "accept": {"lon_m": 0.5, "lat_m": 0.3, "mean_n_accept": round(float(nacc.mean()), 3)},
        "measured_ms": {
            "spec_e2e": summarize(spec_t),
            "ar_keep_think_e2e": summarize(ar_t),
            "verify": summarize([r["t_verify_ms"] for r in rows]),
            "tail": summarize([r["t_tail_ms"] for r in rows]),
            "draft": summarize([r["spec"]["t_draft_ms"] for r in rows]),
            "prefill_in_spec": summarize([r["spec"]["t_prefill_ms"] for r in rows]),
            "think_in_spec": summarize([r["spec"].get("t_think_ms", 0) for r in rows]),
        },
        "speedup_vs_keep_think_ar": round(float(ar_t.mean() / spec_t.mean()), 3),
        "mean_ade_spec_vs_ar": round(float(np.mean([r["ade_spec_vs_ar"] for r in rows])), 4),
        "note": (
            "NAVSIM 10 waypoints / 5s. Draft from prefill hidden. Think tokens decoded "
            "until the next token would be an action; then one teacher-force verifies draft action tokens."
        ),
        "rows": rows,
    }
    if not args.skip_official_generate:
        off_t = np.array([r["t_official_generate_ms"] for r in rows])
        summary["measured_ms"]["official_generate"] = summarize(off_t)
        summary["speedup_vs_official_generate"] = round(float(off_t.mean() / spec_t.mean()), 3)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
