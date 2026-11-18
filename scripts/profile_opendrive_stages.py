#!/usr/bin/env python3
"""Stage-level latency profile of OpenDriveVLA-0.5B + UniAD on one GPU.

Splits the official generate() latency into the stages that the acceleration
papers in this space actually target:

    UniAD vision -> feature encode -> prompt assembly -> LM prefill -> AR decode

The output says which of those stages a candidate method can even touch.
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


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--out", default=str(ROOT / "results/profile_opendrive_stages.json"))
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

    vision_tower = model.get_vision_tower()
    rows = []
    for i, data in enumerate(loader):
        data = move_data_to_device(data, device)

        def _generate():
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

        t_gen, cont = sync_ms(_generate)
        n_new = int(cont.shape[1])

        def _vision():
            with torch.inference_mode():
                return vision_tower(data["uniad_data"])

        t_vision, vt_res = sync_ms(_vision)

        def _encode():
            with torch.inference_mode():
                return model.encode_vision_tower_result(vt_res)

        t_encode, (scene_f, track_f, map_f) = sync_ms(_encode)

        def _prep():
            with torch.inference_mode():
                return model.prepare_inputs_labels_for_multimodal_uniad_vlm(
                    data["input_ids"],
                    None,
                    None,
                    None,
                    None,
                    None,
                    uniad_data=None,
                    uniad_pth=vt_res,
                    qa_instance_ind=data.get("qa_instance_ind"),
                )

        t_prep, (_, pos_ids, attn, _, embeds, _) = sync_ms(_prep)

        def _lm_prefill():
            with torch.inference_mode():
                return model(
                    inputs_embeds=embeds,
                    attention_mask=attn,
                    position_ids=pos_ids,
                    use_cache=True,
                )

        t_lm, pre = sync_ms(_lm_prefill)

        past = pre.past_key_values
        cache_len = int(embeds.shape[1])
        attn_dtype = attn.dtype if attn is not None else torch.long
        tok = torch.tensor([[int(np.argmax(pre.logits[0, -1].float().cpu().numpy()))]], device=device)

        def _step():
            with torch.inference_mode():
                em = model.get_model().embed_tokens(tok)
                a = torch.ones(1, cache_len + 1, device=device, dtype=attn_dtype)
                ps = torch.arange(cache_len, cache_len + 1, device=device).unsqueeze(0)
                return model(
                    inputs_embeds=em, attention_mask=a, position_ids=ps, past_key_values=past, use_cache=True
                )

        t_step, _ = sync_ms(_step)

        n_vis = 0
        for f in (scene_f, track_f, map_f):
            if f is not None:
                n_vis += int(f.shape[0])

        rec = {
            "prompt_len": int(embeds.shape[1]),
            "n_vis_tokens": n_vis,
            "n_text_tokens": int(embeds.shape[1]) - n_vis,
            "n_new_tokens": n_new,
            "t_official_generate_ms": round(t_gen, 2),
            "t_vision_uniad_ms": round(t_vision, 2),
            "t_vision_encode_ms": round(t_encode, 2),
            "t_prepare_inputs_ms": round(t_prep, 2),
            "t_lm_prefill_ms": round(t_lm, 2),
            "t_one_decode_step_ms": round(t_step, 2),
        }
        rec["t_prefill_accounted_ms"] = round(t_vision + t_encode + t_prep + t_lm, 2)
        rec["t_decode_total_ms"] = round(max(t_gen - rec["t_prefill_accounted_ms"], 0.0), 2)
        if i >= args.warmup:
            rows.append(rec)
        print(
            f"[{len(rows)}/{args.n}] gen={t_gen:.0f} vis={t_vision:.0f} enc={t_encode:.0f} "
            f"prep={t_prep:.0f} lm={t_lm:.0f} step={t_step:.1f} ntok={n_new} plen={rec['prompt_len']}",
            flush=True,
        )
        if len(rows) >= args.n:
            break

    def mean(k):
        return round(float(np.mean([r[k] for r in rows])), 2)

    keys = [k for k in rows[0] if k.startswith(("t_", "n_", "prompt"))]
    summary = {
        "model": "OpenDriveVLA-0.5B+UniAD",
        "device": torch.cuda.get_device_name(0),
        "n": len(rows),
        "start_index": args.start,
        "mean": {k: mean(k) for k in keys},
        "stage_share": {},
        "rows": rows,
    }
    m = summary["mean"]
    tot = m["t_official_generate_ms"]
    for name, key in (
        ("vision_uniad", "t_vision_uniad_ms"),
        ("vision_encode", "t_vision_encode_ms"),
        ("prepare_inputs", "t_prepare_inputs_ms"),
        ("lm_prefill", "t_lm_prefill_ms"),
        ("decode", "t_decode_total_ms"),
    ):
        summary["stage_share"][name] = round(m[key] / tot, 4)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
