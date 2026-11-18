#!/usr/bin/env python3
"""Sweep DeeAD's exit depth to find where the intermediate readout stays usable.

DeeAD's premise is that deep layers over-refine and can be skipped. On a 24-layer
0.5B model that may or may not hold, so measure the accuracy/latency curve instead
of assuming one depth.

Reports, per depth: ADE, token count, wall clock, and the L2 gap to the full-depth
answer on the same frame.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

OD_ROOT = Path("/home/xinhchen2/hhb/OpenDriveVLA")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(OD_ROOT)
sys.path.insert(0, str(OD_ROOT))
sys.path.insert(0, str(OD_ROOT / "drivevla"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from mmengine import Config
from torch.utils.data import DataLoader, Subset

from data_utils.nuscenes_llava_datacollector import DataCollatorForLLaVANuScenesDataset
from data_utils.nuscenes_llava_dataset import LLaVANuScenesDataset
from llava.model.builder import load_pretrained_model
from llava.train.train import DataArguments
from llava.utils import disable_torch_init
from utils.tensor_utils import move_data_to_device
from utils.trajectory_utils import retrieve_traj

from trajspec.b2d_data import l2_horizon
from trajspec.nusc_data import VAL_PKL, load_infos, pack_kinematics, token_index
from trajspec.opendrive_live import prefill_uniad, reset_uniad_temporal
from trajspec.serialize import od_to_ego_xy, parse_opendrive_traj

sys.path.insert(0, str(ROOT / "scripts"))
from run_four_accelerators import greedy_at_depth  # noqa: E402

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"
HORIZON = 6


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--depths", default="8,12,16,18,20,22,24")
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--out", default=str(ROOT / "results/deead_depth_sweep.json"))
    args = p.parse_args()
    if not os.path.isabs(args.out):
        args.out = str(ROOT / args.out)
    depths = [int(x) for x in args.depths.split(",")]

    disable_torch_init()
    device = torch.device("cuda:0")
    tokenizer, model, _, _ = load_pretrained_model(
        CKPT, model_base=None, model_name="llava_qwen", device_map=device, multimodal=True,
        attn_implementation="eager",
        overwrite_config={"image_aspect_ratio": "pad", "vision_tower_test_mode": True},
    )
    model.float().to(device).eval()

    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    gt_map = {}
    for tok, i in tok2i.items():
        s = pack_kinematics(infos[i])
        if s is None:
            continue
        g = s.get("gt")
        gt_map[tok] = torch.as_tensor(g[:HORIZON], dtype=torch.float32) if g is not None else torch.zeros(HORIZON, 2)
    vision = model.get_vision_tower()

    uniad_cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
    data_args = DataArguments(data_path=None, lazy_preprocess=True, frames_upbound=32)
    dataset = LLaVANuScenesDataset(tokenizer, data_args, uniad_cfg.data.test,
                                   llava_test_mode=True, use_uniad_pth=False)
    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
    idxs = list(range(args.start, min(args.start + args.n + args.warmup, len(dataset))))
    loader = DataLoader(Subset(dataset, idxs), batch_size=1, collate_fn=collator, num_workers=0)

    rows = []
    for i, data in enumerate(loader):
        sid = data["id"]
        if isinstance(sid, (list, tuple)):
            sid = sid[0]
        token = str(sid).replace("_trajectory", "")
        data = move_data_to_device(data, device)
        gt = gt_map.get(token, torch.zeros(HORIZON, 2)).numpy().astype(np.float32)
        has_gt = bool(np.linalg.norm(gt) > 0.2)

        reset_uniad_temporal(model)
        t_vis, vt_res = sync_ms(lambda: vision(data["uniad_data"]))
        d_shared = {**data, "uniad_pth": vt_res, "uniad_data": None}

        with torch.inference_mode():
            t_ar, cont = sync_ms(lambda: model.generate(
                data["input_ids"], uniad_data=None, uniad_pth=vt_res,
                qa_instance_ind=data.get("qa_instance_ind"),
                do_sample=False, temperature=0, max_new_tokens=args.max_new, num_beams=1))
        ar_ego = od_to_ego_xy(torch.tensor(retrieve_traj(
            tokenizer.batch_decode(cont, skip_special_tokens=True)[0]), dtype=torch.float32)).numpy()

        reset_uniad_temporal(model)
        _, meta = prefill_uniad(model, d_shared)
        meta["prompt_ids"] = [int(t) for t in data["input_ids"][0].tolist()]

        rec = {"token": token, "has_gt": has_gt, "t_vis": round(t_vis, 2),
               "t_ar": round(t_ar, 2), "by_depth": {}}
        for dep in depths:
            t, (text, ids) = sync_ms(lambda d=dep: greedy_at_depth(model, tokenizer, meta, d, args.max_new))
            xy = od_to_ego_xy(parse_opendrive_traj(text, horizon=HORIZON)).numpy()
            l2 = l2_horizon(xy[:HORIZON], gt) if has_gt else {}
            rec["by_depth"][str(dep)] = {
                "t_ms": round(t, 2),
                "n_tokens": len(ids),
                "ade_gt": round(float(l2["ade"]), 4) if has_gt else None,
                "ade_vs_full": round(float(np.linalg.norm(xy[:HORIZON] - ar_ego[:HORIZON], axis=-1).mean()), 4),
                "text_head": text[:50],
            }
            print(f"[{i}] depth={dep:2d} t={t:7.0f}ms tok={len(ids):3d} "
                  f"ADE={rec['by_depth'][str(dep)]['ade_gt']} gap={rec['by_depth'][str(dep)]['ade_vs_full']:.3f}",
                  flush=True)
        if i >= args.warmup:
            rows.append(rec)

    agg = {}
    for dep in depths:
        sel = [r["by_depth"][str(dep)] for r in rows]
        gt_sel = [r["by_depth"][str(dep)] for r in rows if r["has_gt"]]
        agg[str(dep)] = {
            "t_ms": round(float(np.mean([s["t_ms"] for s in sel])), 1),
            "n_tokens": round(float(np.mean([s["n_tokens"] for s in sel])), 1),
            "ade_gt": round(float(np.mean([s["ade_gt"] for s in gt_sel])), 4) if gt_sel else None,
            "ade_vs_full": round(float(np.mean([s["ade_vs_full"] for s in sel])), 4),
        }
    summary = {
        "model": "OpenDriveVLA-0.5B+UniAD",
        "device": torch.cuda.get_device_name(0),
        "n": len(rows), "start_index": args.start, "n_layers": 24,
        "by_depth": agg,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
