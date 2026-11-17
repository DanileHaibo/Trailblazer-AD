#!/usr/bin/env python3
"""Live AutoVLA traj-only SD baselines. Shared propose, one tail per accept rule."""
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
from trajspec.sd_baselines import GEOMETRIC, KERV_LITE, SPECVLA_ID, SPECVLA_POSE, TOKEN_EXACT

OD_CACHE = ROOT / "results" / "cache_opendrive_nusc_val.pt"
MODES = [
    ("Spec-VLA (exact)", TOKEN_EXACT),
    ("Spec-VLA (id-relax)", SPECVLA_ID),
    ("Spec-VLA (pose-relax)", SPECVLA_POSE),
    ("KERV-lite", KERV_LITE),
    ("TrajSpecDec (ours)", GEOMETRIC),
]


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
    return {"mean": round(float(a.mean()), 2), "p50": round(float(np.median(a)), 2), "p90": round(float(np.percentile(a, 90)), 2)}


def od_tokens() -> list[str]:
    blob = torch.load(OD_CACHE, map_location="cpu", weights_only=False)
    return [r["token"] for r in blob["rows"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--id-radius", type=int, default=16)
    p.add_argument("--ckpt", default=str(ROOT / "results/specdec_autovla_trajonly.pt"))
    p.add_argument("--out", default=str(ROOT / "results/live_sd_baselines_autovla.json"))
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
    tokens = od_tokens()
    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    idxs = list(range(args.start, min(args.start + args.n + args.warmup, len(tokens))))

    method_rows = {name: [] for name, _ in MODES}
    ar_rows = []
    seen = 0
    for cache_i in idxs:
        token = tokens[cache_i]
        s = pack_sample_full(infos, tok2i[token], tok2i)
        if s is None:
            print(f"[skip] {token}", flush=True)
            continue
        timed = seen >= args.warmup
        extra = pack_extra(s)
        speed = pack_speed(s)
        gt = np.asarray(s["gt"][: args.horizon], dtype=np.float32)

        t_ar, (ar_xy, ar_info) = sync_ms(
            lambda: runner.ar_action_generate(s, horizon=args.horizon, skip_think=True)
        )
        pack = runner.spec_propose(s, head, extra, speed, horizon=args.horizon, skip_think=True)
        rec_ar = {
            "token": token,
            "t_ar_ms": round(t_ar, 2),
            "t_e2e_ms": ar_info["t_e2e_ms"],
            "t_prefill_ms": ar_info["t_prefill_ms"],
            "t_decode_ms": ar_info["t_e2e_ms"] - ar_info["t_prefill_ms"],
            "ade_gt": round(float(l2_horizon(ar_xy, gt)["ade"]), 4),
        }
        if timed:
            ar_rows.append(rec_ar)

        for name, mode in MODES:
            spec_xy, info = runner.spec_commit(pack, cfg, accept_mode=mode, id_radius=args.id_radius)
            rec = {
                "token": token,
                "n_accept": info["n_accept"],
                "t_spec_e2e_ms": info["t_spec_e2e_ms"],
                "t_prefill_ms": info["t_prefill_ms"],
                "t_think_ms": info["t_think_ms"],
                "t_draft_ms": info["t_draft_ms"],
                "t_verify_ms": info["t_verify_ms"],
                "t_tail_ms": info["t_tail_ms"],
                "n_serial_tail": info["n_serial_tail"],
                "ade_vs_ar": round(float(np.linalg.norm(spec_xy[: args.horizon] - ar_xy[: args.horizon], axis=-1).mean()), 4),
                "ade_gt": round(float(l2_horizon(spec_xy, gt)["ade"]), 4),
                "l2": {**l2_horizon(spec_xy, gt), **l2_uniad_at(spec_xy, gt)},
            }
            rec["t_decode_ms"] = round(rec["t_spec_e2e_ms"] - rec["t_prefill_ms"], 2)
            if timed:
                method_rows[name].append(rec)
        seen += 1
        tag = f"[{len(ar_rows)}/{args.n}]" if timed else "[warmup]"
        accs = {name: method_rows[name][-1]["n_accept"] if timed and method_rows[name] else "-" for name, _ in MODES}
        print(f"{tag} ar={rec_ar['t_e2e_ms']:.0f}ms acc={accs}", flush=True)
        if len(ar_rows) >= args.n:
            break

    def pack_method(name, rows):
        e2e = np.array([r["t_spec_e2e_ms"] for r in rows])
        dec = np.array([r["t_decode_ms"] for r in rows])
        pre = np.array([r["t_prefill_ms"] for r in rows])
        return {
            "n": len(rows),
            "mean_n_accept": round(float(np.mean([r["n_accept"] for r in rows])), 3),
            "e2e_ms": round(float(e2e.mean()), 2),
            "decode_ms": round(float(dec.mean()), 2),
            "prefill_ms": round(float(pre.mean()), 2),
            "tail_ms": round(float(np.mean([r["t_tail_ms"] for r in rows])), 2),
            "ade_vs_ar": round(float(np.mean([r["ade_vs_ar"] for r in rows])), 4),
            "ade_gt": round(float(np.mean([r["ade_gt"] for r in rows])), 4),
            "measured_ms": {"e2e": summarize(e2e), "decode": summarize(dec)},
        }

    ar_e2e = np.array([r["t_e2e_ms"] for r in ar_rows])
    ar_dec = np.array([r["t_decode_ms"] for r in ar_rows])
    summary = {
        "model": "AutoVLA-3B",
        "mode": "traj_only_live_sd_baselines",
        "n": len(ar_rows),
        "start_index": args.start,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "id_radius": args.id_radius,
        "accept": {"lon_m": 0.5, "lat_m": 0.3},
        "AR greedy": {
            "n": len(ar_rows),
            "mean_n_accept": 0.0,
            "e2e_ms": round(float(ar_e2e.mean()), 2),
            "decode_ms": round(float(ar_dec.mean()), 2),
            "prefill_ms": round(float(np.mean([r["t_prefill_ms"] for r in ar_rows])), 2),
            "ade_vs_ar": 0.0,
            "ade_gt": round(float(np.mean([r["ade_gt"] for r in ar_rows])), 4),
        },
        "methods": {name: pack_method(name, method_rows[name]) for name, _ in MODES},
        "rows": {"ar": ar_rows, **{name: method_rows[name] for name, _ in MODES}},
    }
    ar = summary["AR greedy"]
    for name, m in summary["methods"].items():
        m["speedup_e2e"] = round(ar["e2e_ms"] / max(m["e2e_ms"], 1e-6), 3)
        m["speedup_decode"] = round(ar["decode_ms"] / max(m["decode_ms"], 1e-6), 3)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
