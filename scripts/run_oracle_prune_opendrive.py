#!/usr/bin/env python3
"""Oracle ceiling for visual-token pruning on OpenDriveVLA-0.5B + UniAD.

FastDriveVLA (AAAI'26) and Prune2Drive (CVPR'26) both report large prefill
speedups from pruning multi-view visual tokens. The question this script answers
is different and more basic: **how much end-to-end latency is left on the table
at all**, for *any* pruning policy, including a perfect one?

We therefore skip the pruner entirely and slice the post-tower features
(scene / track / map) to keep a fraction of the tokens. This is an oracle: it has
no quality guarantee whatsoever, so it is an upper bound on what a real pruner can
deliver. If the oracle barely moves E2E, no pruning paper can help this teacher.

Fraction 1.0 is the unmodified model and serves as the control.
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

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"


def sync_ms(fn, reps=1):
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--fractions", default="1.0,0.5,0.25,0.125,0.0")
    p.add_argument("--out", default=str(ROOT / "results/oracle_prune_opendrive.json"))
    args = p.parse_args()
    if not os.path.isabs(args.out):
        args.out = str(ROOT / args.out)
    fractions = [float(x) for x in args.fractions.split(",")]

    disable_torch_init()
    device = torch.device("cuda:0")
    tokenizer, model, _, _ = load_pretrained_model(
        CKPT,
        model_base=None,
        model_name="llava_qwen",
        device_map=device,
        multimodal=True,
        attn_implementation="sdpa",
        overwrite_config={"image_aspect_ratio": "pad", "vision_tower_test_mode": True},
    )
    model.float().to(device).eval()

    uniad_cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
    data_args = DataArguments(data_path=None, lazy_preprocess=True, frames_upbound=32)
    dataset = LLaVANuScenesDataset(
        tokenizer, data_args, uniad_cfg.data.test, llava_test_mode=True, use_uniad_pth=False
    )
    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
    idxs = list(range(args.start, min(args.start + args.n + args.warmup, len(dataset))))
    loader = DataLoader(Subset(dataset, idxs), batch_size=1, collate_fn=collator, num_workers=0)

    # Patch the feature projection so that visual tokens can be sliced.
    orig_encode = model.encode_vision_tower_result
    state = {"frac": 1.0, "n_vis": 0}

    def patched(vision_tower_result):
        scene_f, track_f, map_f = orig_encode(vision_tower_result)
        f = state["frac"]
        keep = []
        total = 0
        for feat in (scene_f, track_f, map_f):
            if feat is None:
                keep.append(None)
                continue
            k = int(round(feat.shape[0] * f))
            total += k
            keep.append(feat[:k] if k > 0 else feat[:0])
        state["n_vis"] = total
        return keep[0], keep[1], keep[2]

    model.encode_vision_tower_result = patched

    rows = []
    for i, data in enumerate(loader):
        data = move_data_to_device(data, device)

        def _gen():
            with torch.inference_mode():
                return model.generate(
                    data["input_ids"],
                    uniad_data=data.get("uniad_data"),
                    uniad_pth=data.get("uniad_pth"),
                    qa_instance_ind=data.get("qa_instance_ind"),
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=96,
                    num_beams=1,
                )

        sweep = []
        base_text = None
        for f in fractions:
            state["frac"] = f

            def _gen_f():
                with torch.inference_mode():
                    return model.generate(
                        data["input_ids"],
                        uniad_data=data.get("uniad_data"),
                        uniad_pth=data.get("uniad_pth"),
                        qa_instance_ind=data.get("qa_instance_ind"),
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=96,
                        num_beams=1,
                    )

            t, cont = sync_ms(_gen_f)
            text = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            if base_text is None:
                base_text = text
            sweep.append(
                {
                    "frac": f,
                    "n_vis_tokens": state["n_vis"],
                    "t_generate_ms": round(t, 2),
                    "text_matches_full": bool(text == base_text),
                    "head": text[:60],
                }
            )
            print(
                f"  frac={f:5.3f} vis={state['n_vis']:3d} gen={t:8.1f} ms  same={text == base_text}",
                flush=True,
            )
        # restore for the official timing / answer
        state["frac"] = 1.0
        rec = {"sweep": sweep}
        if i >= args.warmup:
            rows.append(rec)
            print(f"[{len(rows)}/{args.n}] done", flush=True)
        if len(rows) >= args.n:
            break

    # Aggregate by fraction
    agg = {}
    for r in rows:
        for s in r["sweep"]:
            agg.setdefault(s["frac"], []).append(s)
    table = []
    for f in fractions:
        v = agg.get(f, [])
        if not v:
            continue
        table.append(
            {
                "frac": f,
                "n_vis_tokens": int(np.mean([x["n_vis_tokens"] for x in v])),
                "t_generate_ms": round(float(np.mean([x["t_generate_ms"] for x in v])), 2),
                "answer_unchanged_rate": round(float(np.mean([x["text_matches_full"] for x in v])), 3),
            }
        )
    base = next((t for t in table if t["frac"] == 1.0), None)
    if base:
        for t in table:
            t["speedup_vs_full"] = round(base["t_generate_ms"] / max(t["t_generate_ms"], 1e-6), 3)

    summary = {
        "model": "OpenDriveVLA-0.5B+UniAD",
        "mode": "oracle_visual_token_slice",
        "device": torch.cuda.get_device_name(0),
        "n": len(rows),
        "start_index": args.start,
        "table": table,
        "note": (
            "No pruner, no quality guarantee: features are simply truncated, so this is an "
            "UPPER BOUND on any visual-token pruning method. answer_unchanged_rate compares the "
            "decoded string to the unpruned one on the same frame."
        ),
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
