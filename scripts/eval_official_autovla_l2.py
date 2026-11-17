#!/usr/bin/env python3
"""Official-protocol AutoVLA nuScenes evaluation (L2 only).

Mirrors AutoVLA/tools/eval/nusc_eval.py exactly for everything that does not
need segmentation data:

  SFTDataset(val config) -> per-sample builders.compute_features / compute_targets
  -> model.autovla.predict -> pred_xy
  -> PlanningMetric.compute_L2 (the official masked L2)

Collision rate is omitted because nusc_eval_seg is not available locally; that
is stated explicitly in the output. L2 numbers are directly comparable to the
paper's L2 columns.

Usage:
  python tools_eval_official_l2.py --config <yaml> --checkpoint <ckpt> \
      --num_samples 20 --out results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

AUTOVLA = Path("/home/xinhchen2/hhb/AutoVLA")
sys.path.insert(0, str(AUTOVLA))
sys.path.insert(0, str(AUTOVLA / "navsim"))

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor

from dataset_utils.sft_dataset import SFTDataset  # noqa: E402
from models.autovla import SFTAutoVLA  # noqa: E402
from tools.eval.planning_metrics import PlanningMetric  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="official_l2.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    processor = AutoProcessor.from_pretrained(cfg["model"]["pretrained_model_path"], use_fast=True)
    ds = SFTDataset(cfg["data"]["val"], cfg["model"], processor)

    model = SFTAutoVLA(cfg)
    # The released checkpoint stores keys as `autovla.vlm.model.*` (transformers 4.x
    # naming). Loading it verbatim leaves every parameter missing (the model stays
    # randomly initialised and emits plain text instead of <action_i> tokens).
    # Remap onto the transformers 5.x SFTAutoVLA layout.
    def _remap(k: str) -> str:
        k = k[len("autovla.") :] if k.startswith("autovla.") else k
        if k.startswith("vlm.visual."):
            return "vlm.model." + k[len("vlm.") :]
        if (
            k.startswith("vlm.model.layers.")
            or k.startswith("vlm.model.embed_tokens")
            or k.startswith("vlm.model.norm")
        ):
            return "vlm.model.language_model." + k[len("vlm.model.") :]
        return k

    raw = torch.load(args.checkpoint, map_location=args.device)["state_dict"]
    sd = {_remap(k): v for k, v in raw.items()}
    missing, unexpected = model.autovla.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        print(f"[warn] missing sample: {missing[:3]}", flush=True)
        print(f"[warn] unexpected sample: {unexpected[:3]}", flush=True)
    else:
        print("[load] checkpoint matched exactly (missing=0 unexpected=0)", flush=True)
    model.to(args.device)
    model.autovla.device = args.device
    model.eval()
    print("[load] model ready", flush=True)

    metric = PlanningMetric(n_future=6)
    n_total = len(ds.scenes)
    n = n_total if args.num_samples <= 0 else min(args.num_samples, n_total)
    idxs = list(range(args.start, min(args.start + n, n_total)))
    print(f"[data] {n_total} scenes, evaluating {len(idxs)}", flush=True)

    rows = []
    t0 = time.time()
    for k, idx in enumerate(tqdm(idxs, ncols=90)):
        scene_path, _ = ds.scenes[idx]
        scene_data = json.load(open(scene_path))

        feat, tgt = {}, {}
        for b in ds._agent.get_feature_builders():
            feat.update(b.compute_features(scene_data))
        for b in ds._agent.get_target_builders():
            tgt.update(b.compute_targets(scene_data))

        with torch.inference_mode():
            pred_traj, out_text = model.autovla.predict(feat)
        if pred_traj is None or len(pred_traj) == 0:
            continue

        gt = tgt["gt_pos_raw"]
        pred_xy = pred_traj[:, :2].to(gt.device)

        # official coordinate transform (nusc_eval.py lines 137-148)
        gt_u = gt.unsqueeze(0).clone()
        gt_u[:, :, [0, 1]] = gt_u[:, :, [1, 0]]
        gt_u[:, :, 0] = -gt_u[:, :, 0]
        pr_u = pred_xy.unsqueeze(0).clone()
        pr_u[:, :, [0, 1]] = pr_u[:, :, [1, 0]]
        pr_u[:, :, 0] = -pr_u[:, :, 0]

        mask = torch.ones(1, 6, 2, device=gt.device)
        L2 = metric.compute_L2(pr_u[0, :6].unsqueeze(0), gt_u[0, :6].unsqueeze(0), mask)
        rec = {
            "token": scene_data["token"],
            "l2": [round(float(x), 4) for x in L2[0].tolist()],
            "pred_xy": [[round(float(a), 4), round(float(b), 4)] for a, b in pred_xy[:6].tolist()],
            "gt_xy": [[round(float(a), 4), round(float(b), 4)] for a, b in gt[:6].tolist()],
        }
        rows.append(rec)
        if (k + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  [{k+1}/{len(idxs)}] {el/(k+1):.2f}s/it", flush=True)

    l2 = np.array([r["l2"] for r in rows])
    stp = np.array([l2[:, :2].mean(), l2[:, :4].mean(), l2[:, :6].mean()])
    uni = l2[:, [1, 3, 5]].mean(0)
    out = {
        "n": len(rows),
        "protocol": "official AutoVLA nusc_eval (L2 only, no segmentation/collision)",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "stp3_l2": [round(float(x), 4) for x in stp],
        "uniad_l2": [round(float(x), 4) for x in uni],
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n===== {len(rows)} samples =====")
    print(f"ST-P3 L2 1/2/3s: {stp[0]:.4f} / {stp[1]:.4f} / {stp[2]:.4f}")
    print(f"UniAD L2 1/2/3s: {uni[0]:.4f} / {uni[1]:.4f} / {uni[2]:.4f}")
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
