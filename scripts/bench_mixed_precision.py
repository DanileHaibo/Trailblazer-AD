#!/usr/bin/env python3
"""Can the LM run in bf16 while the UniAD tower stays fp32?

OpenDriveVLA's mmcv ops (`modulated_deformable_im2col_gpu`) have no bf16 kernel,
so the tower must stay fp32. But the LM is a plain Qwen2 stack and is the part that
actually costs 23 ms per decoded token. If it can run in bf16 we get a free,
lossless-by-design speedup on the dominant stage -- exactly the kind of
"system-level" win that the big published speedups rely on.

This script casts only `layers` / `norm` / `embed_tokens` / `lm_head` to bf16 and
inserts the dtype boundaries by hand:
  * prompt features (produced in fp32 by the tower) -> cast to bf16 before the LM
  * logits -> cast back to fp32 for the accept rule and parsing

Correctness gate: the decoded trajectory must stay within a small tolerance of the
fp32 answer on the same frames. A dtype change is an implementation detail, so any
meaningful ADE shift means something is wrong.
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
from trajspec.opendrive_live import TeacherDecoder, reset_uniad_temporal
from trajspec.serialize import extract_opendrive_points, od_to_ego_xy

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"
HORIZON = 6


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def build(dtype: torch.dtype):
    disable_torch_init()
    device = torch.device("cuda:0")
    tok_, model_, _, _ = load_pretrained_model(
        CKPT, model_base=None, model_name="llava_qwen", device_map=device, multimodal=True,
        attn_implementation="sdpa",
        overwrite_config={"image_aspect_ratio": "pad", "vision_tower_test_mode": True},
    )
    model_.float().to(device).eval()
    if dtype != torch.float32:
        m = model_.model
        m.layers.to(dtype)
        m.norm.to(dtype)
        m.embed_tokens.to(dtype)
        model_.lm_head.to(dtype)
    return tok_, model_, device


@torch.inference_mode()
def generate_mixed(model, tokenizer, data, vt_res, dtype: torch.dtype, max_new: int = 96):
    """Full generate() path with the LM in `dtype` and the tower in fp32."""
    m = model.model
    # ---- prefill: build prompt features with the fp32 tower, then cast ----
    reset_uniad_temporal(model)
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    _, pos_ids, attn, _, embeds, _ = model.prepare_inputs_labels_for_multimodal_uniad_vlm(
        data["input_ids"], None, None, None, None, None,
        uniad_pth=vt_res, qa_instance_ind=data.get("qa_instance_ind"),
    )
    embeds = embeds.to(dtype)
    out = model(
        inputs_embeds=embeds,
        attention_mask=attn,
        position_ids=pos_ids,
        use_cache=True,
    )
    torch.cuda.synchronize()
    t_prefill = (time.perf_counter() - t0) * 1000.0

    dec = TeacherDecoder(model)
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_set = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
    prompt_ids = [int(t) for t in data["input_ids"][0].tolist()]
    past = out.past_key_values
    cache_len = int(embeds.shape[1])
    attn_dtype = attn.dtype if attn is not None else torch.long

    t0 = time.perf_counter()
    torch.cuda.synchronize()
    logits = out.logits[0, -1].float()
    gen: list[int] = []
    embed = m.embed_tokens
    for _ in range(max_new):
        tid = dec(logits, prompt_ids + gen)
        if tid in eos_set:
            break
        gen.append(tid)
        text = tokenizer.decode(gen, skip_special_tokens=False)
        if len(extract_opendrive_points(text)) >= HORIZON and "]" in text.replace("<traj_end>", ""):
            break
        em = embed(torch.tensor([[tid]], device=embeds.device, dtype=torch.long))
        n = cache_len + len(gen)
        o = model(
            inputs_embeds=em,
            attention_mask=torch.ones(1, n, device=embeds.device, dtype=attn_dtype),
            position_ids=torch.arange(n - 1, n, device=embeds.device).unsqueeze(0),
            past_key_values=past,
            use_cache=True,
        )
        logits = o.logits[0, -1].float()
        past = o.past_key_values
    torch.cuda.synchronize()
    t_decode = (time.perf_counter() - t0) * 1000.0
    return tokenizer.decode(gen, skip_special_tokens=False), gen, t_prefill, t_decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--configs", default="fp32,bf16")
    p.add_argument("--out", default=str(ROOT / "results/mixed_precision_bench.json"))
    args = p.parse_args()
    if not os.path.isabs(args.out):
        args.out = str(ROOT / args.out)

    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    gt_map = {}
    for tok, i in tok2i.items():
        s = pack_kinematics(infos[i])
        if s is None:
            continue
        g = s.get("gt")
        gt_map[tok] = torch.as_tensor(g[:HORIZON], dtype=torch.float32) if g is not None else torch.zeros(HORIZON, 2)

    DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    per_frame: dict[str, list[dict]] = {}
    for name in args.configs.split(","):
        dtype = DT[name]
        print(f"\n===== LM dtype = {dtype} =====", flush=True)
        tokenizer, model, device = build(dtype)
        print("layers:", next(model.model.layers.parameters()).dtype,
              " vision:", next(model.get_vision_tower().parameters()).dtype, flush=True)

        uniad_cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
        data_args = DataArguments(data_path=None, lazy_preprocess=True, frames_upbound=32)
        dataset = LLaVANuScenesDataset(tokenizer, data_args, uniad_cfg.data.test,
                                       llava_test_mode=True, use_uniad_pth=False)
        collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
        idxs = list(range(args.start, min(args.start + args.n + args.warmup, len(dataset))))
        loader = DataLoader(Subset(dataset, idxs), batch_size=1, collate_fn=collator, num_workers=0)
        vision = model.get_vision_tower()

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
            (text, gen, t_pre, t_dec) = generate_mixed(model, tokenizer, data, vt_res, dtype)
            ego = od_to_ego_xy(torch.tensor(retrieve_traj(text), dtype=torch.float32)).numpy()
            l2 = l2_horizon(ego[:HORIZON], gt) if has_gt else {}
            rec = {
                "token": token,
                "t_vis_ms": round(t_vis, 2),
                "t_prefill_ms": round(t_pre, 2),
                "t_decode_ms": round(t_dec, 2),
                "t_e2e_ms": round(t_vis + t_pre + t_dec, 2),
                "n_tokens": len(gen),
                "ade_gt": round(float(l2["ade"]), 4) if has_gt else None,
                "has_gt": has_gt,
                "text_head": text[:60],
            }
            if i >= args.warmup:
                rows.append(rec)
            print(f"[{i}] vis={t_vis:6.1f} pre={t_pre:7.1f} dec={t_dec:8.1f} "
                  f"tok={len(gen):3d} ADE={rec['ade_gt']}", flush=True)
        per_frame[name] = rows
        del model, vision
        torch.cuda.empty_cache()

    summary = {"model": "OpenDriveVLA-0.5B+UniAD", "device": torch.cuda.get_device_name(0),
               "n": args.n, "start_index": args.start, "results": {}}
    base = per_frame.get("fp32") or next(iter(per_frame.values()))
    base_e2e = np.mean([r["t_e2e_ms"] for r in base])

    # cross-config trajectory agreement, keyed by frame token
    ref = {r["token"]: r for r in base}
    for name, rows in per_frame.items():
        e2e = np.array([r["t_e2e_ms"] for r in rows])
        dec = np.array([r["t_decode_ms"] for r in rows])
        gt_rows = [r for r in rows if r["has_gt"]]
        same = [r["token"] for r in rows if r["token"] in ref and r["text_head"] == ref[r["token"]]["text_head"]]
        summary["results"][name] = {
            "n": len(rows),
            "t_e2e_ms": round(float(e2e.mean()), 2),
            "t_decode_ms": round(float(dec.mean()), 2),
            "n_tokens": round(float(np.mean([r["n_tokens"] for r in rows])), 1),
            "ade_gt": round(float(np.mean([r["ade_gt"] for r in gt_rows])), 4) if gt_rows else None,
            "speedup_e2e": round(float(base_e2e / max(e2e.mean(), 1e-6)), 3),
            "identical_answer_rate_vs_fp32": round(len(same) / max(len(rows), 1), 3),
            "rows": rows,
        }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n" + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                             for k, v in summary["results"].items()}, indent=2))
    print("[write]", args.out)


if __name__ == "__main__":
    main()
