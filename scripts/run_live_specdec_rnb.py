#!/usr/bin/env python3
"""Live RnB waypoint VLA specdec vs official generate (CUDA-sync wall-clock)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.accept import AcceptConfig
from trajspec.adapters.rnb_waypoints import RnbWaypointsRuntime
from trajspec.metrics import l2_horizon
from trajspec.draft_head import DraftHead, pack_extra, pack_speed
from trajspec.eval_loop import l2_uniad_at
from trajspec.nusc_data import VAL_PKL, load_infos, pack_sample_rnb, token_index


def sync_ms(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    import time

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


def iter_valid(infos, stride, limit, start=0, shard=0, num_shards=1):
    tok2i = token_index(infos)
    n = 0
    n_valid = 0
    for i in range(len(infos)):
        feat = pack_sample_rnb(infos, i, tok2i)
        if feat is None:
            continue
        if num_shards > 1:
            take = (n_valid % num_shards) == shard
        else:
            take = (n_valid % max(stride, 1)) == 0
        n_valid += 1
        if not take:
            continue
        if num_shards <= 1 and n_valid // max(stride, 1) < start:
            continue
        yield feat
        n += 1
        if n >= limit:
            break


def _summarize_rows(rows):
    off_ms = [r["t_official_generate_ms"] for r in rows]
    spec_ms = [r["spec"]["t_spec_e2e_ms"] for r in rows]
    accs = [r["n_accept"] for r in rows]
    mean_off = float(np.mean(off_ms)) if rows else 0.0
    mean_spec = float(np.mean(spec_ms)) if rows else 0.0
    return {
        "n": len(rows),
        "mean_n_accept": round(float(np.mean(accs)), 3) if rows else 0.0,
        "official_generate_ms": summarize(off_ms) if rows else {},
        "spec_e2e_ms": summarize(spec_ms) if rows else {},
        "prefill_ms": summarize([r["spec"]["t_prefill_ms"] for r in rows]) if rows else {},
        "verify_ms": summarize([r["spec"]["t_verify_ms"] for r in rows]) if rows else {},
        "tail_ms": summarize([r["spec"]["t_tail_ms"] for r in rows]) if rows else {},
        "draft_ms": summarize([r["spec"]["t_draft_ms"] for r in rows]) if rows else {},
        "speedup_vs_official": round(mean_off / max(mean_spec, 1e-6), 3) if rows else 0.0,
        "official_ade": round(float(np.mean([r["official"]["ade"] for r in rows])), 4) if rows else 0.0,
        "spec_ade": round(float(np.mean([r["spec"]["ade"] for r in rows])), 4) if rows else 0.0,
        "ade_spec_vs_official": round(float(np.mean([r["ade_spec_vs_official"] for r in rows])), 4) if rows else 0.0,
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--stride", type=int, default=80)
    p.add_argument("--start", type=int, default=20)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--ckpt", default=str(ROOT / "weights/draft_waypoints_nusc.pt"))
    p.add_argument("--out", default=str(ROOT / "results/live_specdec_rnb.json"))
    p.add_argument("--save-every", type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda")
    runner = RnbWaypointsRuntime(device="cuda")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    head = DraftHead(int(blob["hidden_size"]), horizon=6, extra_size=int(blob.get("extra_size", 16))).to(device)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    cfg = AcceptConfig(lon_m=0.50, lat_m=0.30)
    infos = load_infos(VAL_PKL)

    rows = []
    seen = 0
    target = args.n
    for s in iter_valid(
        infos,
        args.stride,
        args.n + args.warmup,
        start=args.start,
        shard=args.shard,
        num_shards=args.num_shards,
    ):
        extra = pack_extra(s)
        speed = pack_speed(s)
        gt = np.asarray(s["gt"][:6], dtype=np.float32)
        rec = {"token": s["token"]}
        t_off, teacher = sync_ms(lambda: runner.predict(s))
        rec["t_official_generate_ms"] = round(t_off, 2)
        rec["official"] = {**l2_horizon(teacher[:6], gt), **l2_uniad_at(teacher[:6], gt)}
        spec_xy, spec_info = runner.spec_generate(s, head, extra, speed, cfg, horizon=6)
        rec["spec"] = {
            **l2_horizon(spec_xy, gt),
            **l2_uniad_at(spec_xy, gt),
            **{k: spec_info[k] for k in spec_info if k not in ("draft_text", "teacher_verify_text", "final_text")},
        }
        rec["n_accept"] = spec_info["n_accept"]
        rec["ade_spec_vs_official"] = float(
            np.mean(np.linalg.norm(spec_xy[:6] - teacher[:6], axis=-1))
        )
        seen += 1
        if seen > args.warmup:
            rows.append(rec)
            print(
                f"[shard{args.shard} {len(rows)}/{target}] tok={s['token'][:8]} acc={spec_info['n_accept']} "
                f"off={t_off:.0f}ms spec={spec_info['t_spec_e2e_ms']:.0f}ms "
                f"vsT={rec['ade_spec_vs_official']:.2f}",
                flush=True,
            )
            if args.save_every > 0 and len(rows) % args.save_every == 0:
                tmp = _summarize_rows(rows)
                tmp["shard"] = args.shard
                tmp["num_shards"] = args.num_shards
                Path(args.out).write_text(json.dumps(tmp, ensure_ascii=False, indent=2))
                print(f"[ckpt] {args.out} n={len(rows)}", flush=True)
        if len(rows) >= target:
            break

    out = _summarize_rows(rows)
    out["shard"] = args.shard
    out["num_shards"] = args.num_shards
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "rows"}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
