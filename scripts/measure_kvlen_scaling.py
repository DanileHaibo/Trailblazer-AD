#!/usr/bin/env python3
"""Does OpenDriveVLA decode latency scale with prompt length?

The stage profile says AR decode is 73% of E2E and that the prompt is 620 tokens,
of which ~160 are UniAD scene/track/map tokens. Those visual tokens live in the KV
cache, so they are paid *again* at every decode step. That means visual-token
pruning should save roughly (n_vis/prompt_len) * decode_share of E2E, not just the
1.8% that LM prefill accounts for.

This script measures one decode step against prompt length by truncating the
prompt embeddings, to answer "how much is a pruned visual token worth?" without
committing to any particular pruning policy.
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


def sync_ms(fn, reps=3):
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--out", default=str(ROOT / "results/kvlen_scaling_opendrive.json"))
    args = p.parse_args()
    if not os.path.isabs(args.out):
        args.out = str(ROOT / args.out)

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

    rows = []
    for i, data in enumerate(loader):
        data = move_data_to_device(data, device)
        with torch.inference_mode():
            _, pos_ids, attn, _, embeds, _ = model.prepare_inputs_labels_for_multimodal_uniad_vlm(
                data["input_ids"],
                None,
                None,
                None,
                None,
                None,
                uniad_data=data.get("uniad_data"),
                uniad_pth=data.get("uniad_pth"),
                qa_instance_ind=data.get("qa_instance_ind"),
            )
        full_len = int(embeds.shape[1])
        attn_dtype = attn.dtype if attn is not None else torch.long
        embed = model.get_model().embed_tokens

        # Sweep kept prompt length. We truncate the *tail of the prompt*, i.e. keep
        # the visual block and drop trailing text, which is the worst case for the
        # question "does decode cost track total KV length".
        lengths = sorted({full_len, 540, 500, 460, 420, 380, 300, 200}, reverse=True)
        sweep = []
        for L in lengths:
            if L > full_len:
                continue
            e = embeds[:, :L].contiguous()
            a = attn[:, :L].contiguous() if attn is not None else None
            ps = pos_ids[:, :L].contiguous() if pos_ids is not None else None

            def _pre():
                return model(inputs_embeds=e, attention_mask=a, position_ids=ps, use_cache=True)

            t_pre = sync_ms(_pre)
            with torch.inference_mode():
                pre = model(inputs_embeds=e, attention_mask=a, position_ids=ps, use_cache=True)
            past = pre.past_key_values
            tok = torch.tensor([[151643]], device=device, dtype=torch.long)

            def _step():
                em = embed(tok)
                aa = torch.ones(1, L + 1, device=device, dtype=attn_dtype)
                pp = torch.arange(L, L + 1, device=device).unsqueeze(0)
                return model(inputs_embeds=em, attention_mask=aa, position_ids=pp, past_key_values=past, use_cache=True)

            t_step = sync_ms(_step)
            sweep.append({"prompt_len": L, "t_prefill_ms": round(t_pre, 2), "t_step_ms": round(t_step, 2)})
            print(f"  L={L:4d}  prefill={t_pre:7.2f}  step={t_step:6.2f}", flush=True)

        rec = {"prompt_len": full_len, "sweep": sweep}
        if i >= args.warmup:
            rows.append(rec)
            print(f"[{len(rows)}/{args.n}] captured full_len={full_len}", flush=True)
        if len(rows) >= args.n:
            break

    # Aggregate
    agg = {}
    for r in rows:
        for s in r["sweep"]:
            agg.setdefault(s["prompt_len"], []).append((s["t_prefill_ms"], s["t_step_ms"]))
    table = []
    for L in sorted(agg, reverse=True):
        v = np.asarray(agg[L], dtype=np.float64)
        table.append(
            {
                "prompt_len": L,
                "t_prefill_ms": round(float(v[:, 0].mean()), 2),
                "t_step_ms": round(float(v[:, 1].mean()), 3),
            }
        )

    # Linear fit of step time on prompt length -> marginal cost per prompt token
    ls = np.array([t["prompt_len"] for t in table], dtype=np.float64)
    ss = np.array([t["t_step_ms"] for t in table], dtype=np.float64)
    slope, intercept = np.polyfit(ls, ss, 1)
    summary = {
        "model": "OpenDriveVLA-0.5B+UniAD",
        "device": torch.cuda.get_device_name(0),
        "n": len(rows),
        "start_index": args.start,
        "sweep": table,
        "fit_step_ms": {"slope_ms_per_prompt_token": round(float(slope), 6), "intercept_ms": round(float(intercept), 4)},
        "note": (
            "Truncating the prompt tail. Used to price one KV-cache token at decode time, "
            "which is what visual-token pruning actually saves."
        ),
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
