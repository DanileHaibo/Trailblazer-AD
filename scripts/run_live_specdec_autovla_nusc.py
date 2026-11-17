#!/usr/bin/env python3
"""Live AutoVLA specdec on nuScenes val: CUDA-sync wall-clock vs official generate.

Same protocol as OpenDrive live: n=12, warmup=2, start=1500 on the OpenDrive
token order, accept lon=0.50m / lat=0.30m, horizon=6. Decode = E2E - prefill.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.accept import AcceptConfig
from trajspec.adapters.autovla import AutoVLARuntime
from trajspec.b2d_data import l2_horizon
from trajspec.draft_head import DraftHead, pack_extra, pack_speed
from trajspec.eval_loop import l2_uniad_at
from trajspec.nusc_data import VAL_PKL, load_infos, pack_sample_full, token_index

OD_CACHE = ROOT / "results" / "cache_opendrive_nusc_val.pt"


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


def od_tokens() -> list[str]:
    blob = torch.load(OD_CACHE, map_location="cpu", weights_only=False)
    return [r["token"] for r in blob["rows"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--start", type=int, default=1500, help="OpenDrive cache index, same as OD live")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--ckpt", default=str(ROOT / "results/specdec_autovla_full.pt"))
    p.add_argument("--out", default=str(ROOT / "results/live_specdec_autovla_nusc.json"))
    p.add_argument("--skip-official-generate", action="store_true")
    p.add_argument("--traj-only", action="store_true", help="Official trajectory-only: no CoT think tokens")
    args = p.parse_args()

    device = torch.device("cuda")
    runner = AutoVLARuntime(device="cuda", temperature=None, horizon=args.horizon)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    head = DraftHead(
        int(blob["hidden_size"]),
        horizon=args.horizon,
        extra_size=int(blob.get("extra_size", 16)),
        width=512,
    ).to(device)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    cfg = AcceptConfig(lon_m=0.5, lat_m=0.3)
    horizon = args.horizon

    tokens = od_tokens()
    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    idxs = list(range(args.start, min(args.start + args.n + args.warmup, len(tokens))))

    rows = []
    seen = 0
    for cache_i in idxs:
        token = tokens[cache_i]
        s = pack_sample_full(infos, tok2i[token], tok2i)
        if s is None:
            print(f"[skip] cannot pack {token}", flush=True)
            continue
        timed = seen >= args.warmup
        extra = pack_extra(s)
        speed = pack_speed(s)
        gt = np.asarray(s["gt"][:horizon], dtype=np.float32)

        rec = {"token": token, "cache_index": cache_i}
        skip_off = args.skip_official_generate or args.traj_only
        if not skip_off:
            t_off, teacher_off = sync_ms(lambda: runner.predict(s))
            rec["t_official_generate_ms"] = round(t_off, 2)
            rec["official"] = {**l2_horizon(teacher_off[:horizon], gt), **l2_uniad_at(teacher_off[:horizon], gt)}

        t_ar, (ar_xy, ar_info) = sync_ms(
            lambda: runner.ar_action_generate(s, horizon=horizon, skip_think=args.traj_only)
        )
        rec["t_ar_keep_think_ms"] = round(t_ar, 2)
        rec["ar"] = {
            **l2_horizon(ar_xy, gt),
            **l2_uniad_at(ar_xy, gt),
            **{k: ar_info[k] for k in ar_info if k not in ("action_ids", "hidden")},
        }

        spec_xy, spec_info = runner.spec_generate(
            s, head, extra, speed, cfg, horizon=horizon, skip_think=args.traj_only
        )
        rec["spec"] = {
            **l2_horizon(spec_xy, gt),
            **l2_uniad_at(spec_xy, gt),
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
        rec["t_spec_e2e_ms"] = spec_info["t_spec_e2e_ms"]
        rec["t_prefill_ms"] = spec_info["t_prefill_ms"]
        rec["t_think_ms"] = spec_info["t_think_ms"]
        rec["t_verify_ms"] = spec_info["t_verify_ms"]
        rec["t_tail_ms"] = spec_info["t_tail_ms"]
        rec["t_draft_ms"] = spec_info["t_draft_ms"]
        rec["spec_decode_steps"] = spec_info["spec_decode_steps"]
        rec["ade_spec_vs_ar"] = round(float(np.linalg.norm(spec_xy[:horizon] - ar_xy[:horizon], axis=-1).mean()), 4)

        seen += 1
        tag = f"[{len(rows) + 1}/{args.n}]" if timed else "[warmup]"
        print(
            f"{tag} spec={rec['t_spec_e2e_ms']:.0f}ms ar={rec['t_ar_keep_think_ms']:.0f}ms "
            f"off={rec.get('t_official_generate_ms', 0):.0f}ms pre={rec['t_prefill_ms']:.0f}ms "
            f"think={rec['t_think_ms']:.0f}ms accept={rec['n_accept']}",
            flush=True,
        )
        if timed:
            rows.append(rec)
        if len(rows) >= args.n:
            break

    spec_t = np.array([r["t_spec_e2e_ms"] for r in rows])
    ar_t = np.array([r["t_ar_keep_think_ms"] for r in rows])
    pre_t = np.array([r["t_prefill_ms"] for r in rows])
    nacc = np.array([r["n_accept"] for r in rows], dtype=np.float64)
    spec_dec = spec_t - pre_t
    ar_e2e = np.array([r["ar"]["t_e2e_ms"] for r in rows])
    ar_dec = ar_e2e - np.array([r["ar"]["t_prefill_ms"] for r in rows])

    latency = {
        "teacher": {
            "e2e_ms": round(float(ar_e2e.mean()), 2),
            "decode_ms": round(float(ar_dec.mean()), 2),
            "prefill_ms": round(float(np.mean([r["ar"]["t_prefill_ms"] for r in rows])), 2),
            "note": (
                "trajectory-only AR (force <answer> prefix, no CoT)"
                if args.traj_only
                else "keep-think AR internals; official generate() is reported separately if run"
            ),
        },
        "spec": {
            "e2e_ms": round(float(spec_t.mean()), 2),
            "decode_ms": round(float(spec_dec.mean()), 2),
            "prefill_ms": round(float(pre_t.mean()), 2),
        },
        "ar": {
            "e2e_ms": round(float(ar_e2e.mean()), 2),
            "decode_ms": round(float(ar_dec.mean()), 2),
            "prefill_ms": round(float(np.mean([r["ar"]["t_prefill_ms"] for r in rows])), 2),
            "prefix_or_think_ms": round(float(np.mean([r["ar"].get("t_prefix_ms") or r["ar"]["t_think_ms"] for r in rows])), 2),
            "action_ms": round(float(np.mean([r["ar"].get("t_action_ms", r["ar"]["t_decode_ms"]) for r in rows])), 2),
        },
    }
    skip_off = args.skip_official_generate or args.traj_only
    if not skip_off:
        off_t = np.array([r["t_official_generate_ms"] for r in rows])
        latency["official_generate"] = {
            "e2e_ms": round(float(off_t.mean()), 2),
            "decode_ms": round(float((off_t - pre_t).mean()), 2),
            "note": "official generate(); decode = e2e - spec prefill (same prompt)",
        }

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    summary = {
        "model": "AutoVLA-3B",
        "dataset": "nuScenes val (OpenDrive token order)",
        "device": gpu,
        "mode": "traj_only_live" if args.traj_only else "live_teacher_force_verify",
        "n": len(rows),
        "start_index": args.start,
        "horizon": horizon,
        "ckpt": args.ckpt,
        "accept": {"lon_m": 0.5, "lat_m": 0.3, "mean_n_accept": round(float(nacc.mean()), 3)},
        "latency": latency,
        "measured_ms": {
            "spec_e2e": summarize(spec_t),
            "spec_decode": summarize(spec_dec),
            "teacher_e2e": summarize(ar_e2e),
            "teacher_decode": summarize(ar_dec),
            "prefill": summarize(pre_t),
            "think_or_prefix_in_spec": summarize([r["t_think_ms"] for r in rows]),
            "verify": summarize([r["t_verify_ms"] for r in rows]),
            "tail": summarize([r["t_tail_ms"] for r in rows]),
            "draft": summarize([r["t_draft_ms"] for r in rows]),
        },
        "speedup_e2e_vs_teacher": round(float(ar_e2e.mean() / spec_t.mean()), 3),
        "speedup_decode_vs_teacher": round(float(ar_dec.mean() / spec_dec.mean()), 3),
        "mean_ade_spec_vs_ar": round(float(np.mean([r["ade_spec_vs_ar"] for r in rows])), 4),
        "note": (
            "CUDA-sync wall-clock. E2E includes prefill. Decode = E2E - prefill. "
            + (
                "Trajectory-only: no CoT; teacher-force <answer> prefix then action tokens. "
                if args.traj_only
                else "AutoVLA decode still contains CoT think tokens. "
            )
            + "Same n=12 / start=1500 protocol as OpenDrive live."
        ),
        "rows": rows,
    }
    if not skip_off:
        off_t = np.array([r["t_official_generate_ms"] for r in rows])
        summary["measured_ms"]["official_generate"] = summarize(off_t)
        summary["measured_ms"]["official_decode"] = summarize(off_t - pre_t)
        summary["speedup_e2e_vs_official"] = round(float(off_t.mean() / spec_t.mean()), 3)
        summary["speedup_decode_vs_official"] = round(float((off_t - pre_t).mean() / spec_dec.mean()), 3)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
