#!/usr/bin/env python3
"""Train Orion-style draft heads against two AR VLA tokenizations, then run
waypoint-level speculative decoding.

Default `--mode toy` uses teachers that emit:
  * OpenDriveVLA textual waypoints  [(x,y), ...] x6
  * AutoVLA physical action tokens  <action_i> x10
so the accept/serialize path is the official one even without gated weights.

`--mode real` loads AutoVLA / OpenDriveVLA checkpoints when present.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.accept import AcceptConfig
from trajspec.adapters.autovla import AutoVLATeacher
from trajspec.adapters.opendrivevla import OpenDriveVLATeacher
from trajspec.adapters.toy import AutoVLAStyleTeacher, OpenDriveStyleTeacher
from trajspec.draft_head import DraftHead
from trajspec.scenes import sample_scenes
from trajspec.spec_loop import speculative_trajectory, train_step


def train_draft(teacher, scenes, steps=2000, lr=2e-3, batch_size=32, device="cpu"):
    head = DraftHead(teacher.hidden_size, horizon=teacher.horizon).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    losses = []
    teacher.eval()
    cache = []
    with torch.no_grad():
        for scene in scenes:
            hidden, kv = teacher.prefill(scene)
            target = teacher.teacher_waypoints(scene, kv, hidden)
            cache.append((hidden.float().cpu(), target.float().cpu()))
    for i in range(steps):
        idx = torch.randint(0, len(cache), (batch_size,))
        hidden = torch.stack([cache[int(j)][0] for j in idx], dim=0).to(device)
        target = torch.stack([cache[int(j)][1] for j in idx], dim=0).to(device)
        loss = train_step(head, hidden, target, opt)
        losses.append(loss)
    head.eval()
    return head, losses


def eval_spec(teacher, head, scenes, accept_cfg, device="cpu"):
    rows = []
    t0 = time.perf_counter()
    for scene in scenes:
        with torch.no_grad():
            r = speculative_trajectory(teacher, head.to("cpu"), scene, accept_cfg)
        ade_draft = float(((r.draft - r.teacher_full[: r.draft.shape[0]]) ** 2).sum(-1).sqrt().mean())
        k = min(r.trajectory.shape[0], r.teacher_full.shape[0])
        ade_spec = float(((r.trajectory[:k] - r.teacher_full[:k]) ** 2).sum(-1).sqrt().mean())
        rows.append(
            {
                "n_accept": r.n_accept,
                "horizon": r.baseline_decode_steps,
                "spec_decode_steps": r.spec_decode_steps,
                "decode_speedup": round(r.decode_speedup, 3),
                "ade_draft_m": round(ade_draft, 4),
                "ade_spec_m": round(ade_spec, 4),
            }
        )
    elapsed = time.perf_counter() - t0
    n = max(len(rows), 1)
    max_h = max(x["horizon"] for x in rows)
    prefix_accept = [0] * max_h
    for x in rows:
        for i in range(x["n_accept"]):
            prefix_accept[i] += 1
    summary = {
        "n_scenes": len(rows),
        "mean_n_accept": round(sum(x["n_accept"] for x in rows) / n, 3),
        "accept_rate": round(
            sum(x["n_accept"] for x in rows) / max(sum(x["horizon"] for x in rows), 1), 3
        ),
        "per_index_accept": [round(c / n, 3) for c in prefix_accept],
        "mean_decode_speedup": round(sum(x["decode_speedup"] for x in rows) / n, 3),
        "mean_ade_draft_m": round(sum(x["ade_draft_m"] for x in rows) / n, 4),
        "mean_ade_spec_m": round(sum(x["ade_spec_m"] for x in rows) / n, 4),
        "wall_s": round(elapsed, 3),
    }
    return summary, rows


def run_one(name, teacher, train_scenes, val_scenes, accept_cfg, train_steps, out_dir):
    print(f"\n=== {name}  horizon={teacher.horizon}  hidden={teacher.hidden_size} ===", flush=True)
    head, losses = train_draft(teacher, train_scenes, steps=train_steps)
    ckpt = out_dir / f"draft_{name}.pt"
    torch.save({"state_dict": head.state_dict(), "horizon": teacher.horizon, "hidden_size": teacher.hidden_size}, ckpt)
    summary, rows = eval_spec(teacher, head, val_scenes, accept_cfg)
    summary["final_train_loss"] = round(sum(losses[-50:]) / min(50, len(losses)), 5)
    summary["draft_ckpt"] = str(ckpt)
    print(json.dumps(summary, indent=2), flush=True)
    return {"name": name, "summary": summary, "samples": rows[:8]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["toy", "real"], default="toy")
    p.add_argument("--train-scenes", type=int, default=256)
    p.add_argument("--val-scenes", type=int, default=64)
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--lon-m", type=float, default=0.35)
    p.add_argument("--lat-m", type=float, default=0.20)
    p.add_argument("--autovla-ckpt", default="")
    p.add_argument("--opendrive-ckpt", default="")
    p.add_argument("--out", default=str(ROOT / "results" / "two_vla_spec.json"))
    args = p.parse_args()

    out_dir = Path(args.out).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    accept_cfg = AcceptConfig(lon_m=args.lon_m, lat_m=args.lat_m)

    reports = []
    notes = []

    if args.mode == "real":
        od = OpenDriveVLATeacher(checkpoint=args.opendrive_ckpt or None)
        av = AutoVLATeacher(checkpoint=args.autovla_ckpt or None)
        if od.available():
            od.load()
            scenes_tr = sample_scenes(args.train_scenes, od.horizon, seed=1)
            scenes_va = sample_scenes(args.val_scenes, od.horizon, seed=2)
            reports.append(run_one("opendrivevla_real", od, scenes_tr, scenes_va, accept_cfg, args.train_steps, out_dir))
        else:
            notes.append("OpenDriveVLA checkpoint missing; fell back to toy serialization teacher")
        if av.available():
            av.load()
            scenes_tr = sample_scenes(args.train_scenes, av.horizon, seed=3)
            scenes_va = sample_scenes(args.val_scenes, av.horizon, seed=4)
            reports.append(run_one("autovla_real", av, scenes_tr, scenes_va, accept_cfg, args.train_steps, out_dir))
        else:
            notes.append("AutoVLA checkpoint missing; fell back to toy serialization teacher")

    # Always run the two AR tokenizations (official serialize/parse).
    od_toy = OpenDriveStyleTeacher()
    av_toy = AutoVLAStyleTeacher()
    reports.append(
        run_one(
            "opendrivevla_text_waypoints",
            od_toy,
            sample_scenes(args.train_scenes, od_toy.horizon, seed=11),
            sample_scenes(args.val_scenes, od_toy.horizon, seed=12),
            accept_cfg,
            args.train_steps,
            out_dir,
        )
    )
    reports.append(
        run_one(
            "autovla_action_tokens",
            av_toy,
            sample_scenes(args.train_scenes, av_toy.horizon, seed=21),
            sample_scenes(args.val_scenes, av_toy.horizon, seed=22),
            accept_cfg,
            args.train_steps,
            out_dir,
        )
    )

    result = {
        "mode": args.mode,
        "accept": {"lon_m": args.lon_m, "lat_m": args.lat_m},
        "teachers": [
            {
                "name": "OpenDriveVLA",
                "ar": "textual waypoints [(x,y), ...] x6 over 3s",
                "verify": "waypoint-level lon/lat thresholds, not BPE exact match",
            },
            {
                "name": "AutoVLA",
                "ar": "physical <action_i> tokens x10 over 5s",
                "verify": "decoded pose vs draft pose, same metre thresholds",
            },
        ],
        "notes": notes,
        "reports": reports,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
