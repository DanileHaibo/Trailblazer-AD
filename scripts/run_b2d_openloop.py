#!/usr/bin/env python3
"""Bench2Drive open-loop eval for AutoVLA and OpenDriveVLA + waypoint specdec."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/xinhchen2/hhb/AutoVLA")

from trajspec.accept import AcceptConfig, longest_prefix_accept
from trajspec.adapters.autovla import AutoVLARuntime
from trajspec.adapters.opendrivevla import OpenDriveLMRuntime
from trajspec.b2d_data import load_infos, subsample_valid, l2_horizon
from trajspec.draft_head import DraftHead
from trajspec.spec_loop import train_step

B2D_ROOT = Path("/home/xinhchen2/hhb/Bench2DriveZoo/data/bench2drive")
VAL_PKL = Path("/home/xinhchen2/hhb/Bench2DriveZoo/data/infos/b2d_infos_val.pkl")
TRAIN_PKL = Path("/home/xinhchen2/hhb/Bench2DriveZoo/data/infos/b2d_infos_train.pkl")
QWEN = Path("/data2/hf_overlap_models/Qwen2.5-VL-3B-Instruct")
AUTOVLA_CKPT = Path("/data2/hf_overlap_models/AutoVLA/AutoVLA_PDMS_89.ckpt")
OPENDRIVE_CKPT = Path("/data2/hf_overlap_models/OpenDriveVLA-0.5B")


def mean_dict(rows, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = round(float(np.mean(vals)), 4) if vals else None
    return out


def spec_from_draft(draft, teacher, cfg: AcceptConfig, horizon: int = 6):
    d = torch.as_tensor(draft[:, :2], dtype=torch.float32)
    t = torch.as_tensor(teacher[:, :2], dtype=torch.float32)
    k = min(d.shape[0], t.shape[0], horizon)
    n = longest_prefix_accept(d[:k], t[:k], cfg)
    if n >= k:
        spec = d[:k]
        steps = 1
    else:
        parts = []
        if n > 0:
            parts.append(d[:n])
        parts.append(t[n : n + 1])
        if n + 1 < t.shape[0]:
            parts.append(t[n + 1 : k])
        spec = torch.cat(parts, dim=0)[:k]
        steps = 1 + max(k - n - 1, 0)
    return spec.numpy(), n, steps


def collect_and_train(runner, samples, steps=800, supervise="teacher", horizon=6):
    cache_h, cache_y = [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        h = runner.prefill_hidden(s)
        if supervise == "teacher":
            y = torch.as_tensor(runner.predict(s)[:horizon], dtype=torch.float32)
        else:
            y = torch.as_tensor(s["gt"][:horizon], dtype=torch.float32)
        if y.shape[0] < horizon:
            pad = y[-1:].repeat(horizon - y.shape[0], 1) if y.shape[0] else torch.zeros(horizon, 2)
            y = torch.cat([y, pad], dim=0)[:horizon]
        cache_h.append(h)
        cache_y.append(y[:horizon])
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  cache {i+1}/{len(samples)}  elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"[cache] {len(cache_h)} samples in {time.time()-t0:.1f}s", flush=True)
    hs = torch.stack(cache_h, dim=0)
    ys = torch.stack(cache_y, dim=0)
    head = DraftHead(int(hs.shape[-1]), horizon=horizon).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3)
    losses = []
    for i in range(steps):
        idx = torch.randint(0, hs.shape[0], (min(32, hs.shape[0]),))
        loss = train_step(head, hs[idx].cuda(), ys[idx].cuda(), opt)
        losses.append(loss)
    head.eval()
    return head, losses


def eval_split(name, runner, head, samples, accept_cfg, horizon=6):
    rows = []
    for i, s in enumerate(samples):
        t_gen0 = time.perf_counter()
        teacher = runner.predict(s)
        t_gen = (time.perf_counter() - t_gen0) * 1000
        t_h0 = time.perf_counter()
        hidden = runner.prefill_hidden(s)
        with torch.no_grad():
            draft = head(hidden.cuda())[0, :horizon].cpu().numpy()
        t_draft = (time.perf_counter() - t_h0) * 1000
        spec, n_acc, spec_steps = spec_from_draft(draft, teacher, accept_cfg, horizon=horizon)
        gt = s["gt"]
        rec = {
            "idx": s["idx"],
            "folder": s["folder"],
            "n_accept": int(n_acc),
            "spec_decode_steps": int(spec_steps),
            "decode_speedup": round(horizon / max(spec_steps, 1e-6), 3),
            "gen_ms": round(t_gen, 1),
            "prefill_draft_ms": round(t_draft, 1),
            "teacher": l2_horizon(teacher, gt),
            "spec": l2_horizon(spec, gt),
            "draft": l2_horizon(draft, gt),
        }
        rows.append(rec)
        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"  [{name}] {i+1}/{len(samples)} accept={n_acc} "
                f"teacherADE={rec['teacher']['ade']:.3f} specADE={rec['spec']['ade']:.3f} "
                f"gen={t_gen:.0f}ms",
                flush=True,
            )
    def flatten(tag):
        return mean_dict(
            [{k: r[tag][k] for k in ("l2_1s", "l2_2s", "l2_3s", "ade")} for r in rows],
            ["l2_1s", "l2_2s", "l2_3s", "ade"],
        )
    summary = {
        "n": len(rows),
        "mean_n_accept": round(float(np.mean([r["n_accept"] for r in rows])), 3),
        "mean_decode_speedup": round(float(np.mean([r["decode_speedup"] for r in rows])), 3),
        "mean_gen_ms": round(float(np.mean([r["gen_ms"] for r in rows])), 1),
        "mean_prefill_draft_ms": round(float(np.mean([r["prefill_draft_ms"] for r in rows])), 1),
        "teacher_openloop": flatten("teacher"),
        "spec_openloop": flatten("spec"),
        "draft_openloop": flatten("draft"),
    }
    return summary, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-n", type=int, default=80)
    p.add_argument("--val-n", type=int, default=120)
    p.add_argument("--train-stride", type=int, default=40)
    p.add_argument("--val-stride", type=int, default=80)
    p.add_argument("--train-steps", type=int, default=1200)
    p.add_argument("--lon-m", type=float, default=0.5)
    p.add_argument("--lat-m", type=float, default=0.3)
    p.add_argument("--smoke", type=int, default=0, help="if >0, only predict this many val samples then exit")
    p.add_argument("--models", default="autovla,opendrive", help="comma list: autovla,opendrive")
    p.add_argument("--out", default=str(ROOT / "results" / "b2d_openloop.json"))
    args = p.parse_args()

    print("[data] loading B2D infos", flush=True)
    val_infos = load_infos(VAL_PKL)
    train_infos = load_infos(TRAIN_PKL)
    val_samples = subsample_valid(val_infos, B2D_ROOT, args.val_stride, args.val_n, start=20)
    train_samples = subsample_valid(train_infos, B2D_ROOT, args.train_stride, args.train_n, start=10)
    print(f"[data] train={len(train_samples)} val={len(val_samples)}", flush=True)
    if not val_samples:
        raise SystemExit("no valid B2D val samples (check camera paths)")

    want = {x.strip() for x in args.models.split(",") if x.strip()}
    accept_cfg = AcceptConfig(lon_m=args.lon_m, lat_m=args.lat_m)
    reports = {}
    notes = [
        "Bench2Drive open-loop follows VAD: 2 Hz x 6 waypoints = 3 s, L2 mean-upto-t.",
        "AutoVLA is the official visual AR teacher (3 cameras x 4 frames).",
        "OpenDriveVLA official vision is UniAD track/map; B2D has no UniAD features, so we run the 0.5B LM on ego/history/command text in the official waypoint format.",
        "Both teachers were trained on nuScenes/nuPlan, not CARLA/B2D; absolute L2 may be large. Specdec is measured against the teacher.",
        "Draft head is trained on teacher waypoints (not GT). Accept: lon/lat metres.",
    ]

    if args.smoke:
        sample = val_samples[0]
        print("[smoke] gt", np.round(sample["gt"], 3).tolist(), flush=True)
        if "autovla" in want and AUTOVLA_CKPT.exists() and QWEN.exists():
            runner = AutoVLARuntime(device="cuda")
            t0 = time.time()
            pred = runner.predict(sample)
            print(f"[smoke] AutoVLA {time.time()-t0:.1f}s pred={np.round(pred[:6], 3).tolist()}", flush=True)
            print("[smoke] AutoVLA L2", l2_horizon(pred, sample["gt"]), flush=True)
            del runner
            torch.cuda.empty_cache()
        if "opendrive" in want:
            runner = OpenDriveLMRuntime(device="cuda")
            t0 = time.time()
            pred = runner.predict(sample)
            print(f"[smoke] OpenDrive {time.time()-t0:.1f}s pred={np.round(pred[:6], 3).tolist()}", flush=True)
            print("[smoke] OpenDrive L2", l2_horizon(pred, sample["gt"]), flush=True)
        return

    if "autovla" in want:
        if AUTOVLA_CKPT.exists() and QWEN.exists():
            runner = AutoVLARuntime(device="cuda")
            head, losses = collect_and_train(
                runner, train_samples, steps=args.train_steps, supervise="teacher"
            )
            torch.save(
                {"state_dict": head.state_dict(), "hidden_size": runner.hidden_size},
                ROOT / "results" / "draft_autovla_b2d.pt",
            )
            summary, rows = eval_split("autovla", runner, head, val_samples, accept_cfg)
            summary["final_train_loss"] = round(float(np.mean(losses[-50:])), 5)
            reports["AutoVLA"] = {"summary": summary, "samples": rows[:16]}
            del runner
            torch.cuda.empty_cache()
        else:
            notes.append(f"AutoVLA weights missing ckpt={AUTOVLA_CKPT.exists()} qwen={QWEN.exists()}")

    if "opendrive" in want:
        try:
            runner = OpenDriveLMRuntime(device="cuda")
            head, losses = collect_and_train(
                runner, train_samples, steps=args.train_steps, supervise="teacher"
            )
            torch.save(
                {"state_dict": head.state_dict(), "hidden_size": runner.hidden_size},
                ROOT / "results" / "draft_opendrive_b2d.pt",
            )
            summary, rows = eval_split("opendrivevla", runner, head, val_samples, accept_cfg)
            summary["final_train_loss"] = round(float(np.mean(losses[-50:])), 5)
            reports["OpenDriveVLA"] = {"summary": summary, "samples": rows[:16]}
        except Exception as e:
            notes.append(f"OpenDriveVLA failed: {type(e).__name__}: {e}")
            print("[opendrive] failed:", type(e).__name__, e, flush=True)

    result = {
        "benchmark": "Bench2Drive open-loop (VAD protocol: 2Hz x 6 steps = 3s, L2 mean-upto-t)",
        "data": {
            "val_pkl": str(VAL_PKL),
            "n_train": len(train_samples),
            "n_val": len(val_samples),
        },
        "accept": {"lon_m": args.lon_m, "lat_m": args.lat_m},
        "notes": notes,
        "reports": reports,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v.get("summary") if isinstance(v, dict) else v for k, v in reports.items()}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
