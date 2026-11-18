#!/usr/bin/env python3
"""Is our decode slow because of the model, or because we run it in fp32?

Our harness calls `model.float()` (fp32) while OpenDriveVLA's own evaluation runs
bf16. A 0.5B model decoding one token should not cost 23 ms -- that number is
implausible for the parameter count, so the arithmetic is worth checking before
blaming the algorithm.

Sweeps dtype x attention implementation on the same frames and reports decode-step
latency plus the ADE drift (a dtype change must not move the trajectory if it is a
pure implementation detail).
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
from trajspec.opendrive_live import _cache_seq_len, _continue, prefill_uniad, reset_uniad_temporal
from trajspec.serialize import od_to_ego_xy

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"
HORIZON = 6


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def build(dtype_name: str, attn: str):
    disable_torch_init()
    device = torch.device("cuda:0")
    tok_, model_, _, _ = load_pretrained_model(
        CKPT, model_base=None, model_name="llava_qwen", device_map=device, multimodal=True,
        attn_implementation=attn,
        overwrite_config={"image_aspect_ratio": "pad", "vision_tower_test_mode": True},
    )
    if dtype_name == "fp32":
        model_.float()
    elif dtype_name == "bf16":
        model_.to(torch.bfloat16)
    elif dtype_name == "fp16":
        model_.to(torch.float16)
    model_.to(device).eval()
    return tok_, model_, device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--configs", default="fp32:sdpa,bf16:sdpa,bf16:eager,fp16:sdpa")
    p.add_argument("--out", default=str(ROOT / "results/dtype_bench.json"))
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

    results = {}
    for conf in args.configs.split(","):
        dtype_name, attn = conf.split(":")
        print(f"\n===== {dtype_name} / {attn} =====", flush=True)
        tokenizer, model, device = build(dtype_name, attn)
        dtype = next(model.parameters()).dtype
        print(f"param dtype = {dtype}", flush=True)

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

            def _gen():
                with torch.inference_mode():
                    return model.generate(
                        data["input_ids"], uniad_data=None, uniad_pth=vt_res,
                        qa_instance_ind=data.get("qa_instance_ind"),
                        do_sample=False, temperature=0, max_new_tokens=96, num_beams=1,
                    )

            t_gen, cont = sync_ms(_gen)
            text = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            ego = od_to_ego_xy(torch.tensor(retrieve_traj(text), dtype=torch.float32)).numpy()
            n_tok = int(cont.shape[1])
            l2 = l2_horizon(ego[:HORIZON], gt) if has_gt else {}

            # single cached decode step, for a clean per-token cost
            reset_uniad_temporal(model)
            d_shared = {**data, "uniad_pth": vt_res, "uniad_data": None}
            pre, meta = prefill_uniad(model, d_shared)
            cache_len = _cache_seq_len(pre.past_key_values) or int(meta["prompt_len"])
            attn_dtype = meta["attention_mask"].dtype if meta["attention_mask"] is not None else torch.long
            first = int(pre.logits[0, -1].float().argmax().item())

            def _step():
                return _continue(
                    model, torch.tensor([[first]], device=device), pre.past_key_values,
                    cache_len, attn_dtype,
                )

            t_step, _ = sync_ms(_step)

            rec = {
                "t_vis_ms": round(t_vis, 2),
                "t_generate_ms": round(t_gen, 2),
                "t_step_ms": round(t_step, 3),
                "n_tokens": n_tok,
                "ade_gt": round(float(l2["ade"]), 4) if has_gt else None,
                "has_gt": has_gt,
                "text_head": text[:60],
            }
            if i >= args.warmup:
                rows.append(rec)
            print(f"[{i}] gen={t_gen:8.1f}ms step={t_step:6.2f}ms tok={n_tok:3d} ADE={rec['ade_gt']}", flush=True)

        step_ms = np.array([r["t_step_ms"] for r in rows])
        gen_ms = np.array([r["t_generate_ms"] for r in rows])
        gt_rows = [r for r in rows if r["has_gt"]]
        results[conf] = {
            "param_dtype": str(dtype),
            "n": len(rows),
            "t_generate_ms": round(float(gen_ms.mean()), 2),
            "t_step_ms": round(float(step_ms.mean()), 3),
            "n_tokens": round(float(np.mean([r["n_tokens"] for r in rows])), 1),
            "ade_gt": round(float(np.mean([r["ade_gt"] for r in gt_rows])), 4) if gt_rows else None,
            "rows": rows,
        }
        del model, vision
        torch.cuda.empty_cache()

    base = results.get("fp32:sdpa") or next(iter(results.values()))
    for k, v in results.items():
        v["speedup_vs_fp32"] = round(base["t_generate_ms"] / max(v["t_generate_ms"], 1e-6), 3)
        v["step_speedup_vs_fp32"] = round(base["t_step_ms"] / max(v["t_step_ms"], 1e-6), 3)

    summary = {
        "model": "OpenDriveVLA-0.5B+UniAD",
        "device": torch.cuda.get_device_name(0),
        "n": args.n, "start_index": args.start,
        "results": results,
        "note": (
            "Pure dtype/attention sweep. A change here must not move the trajectory: "
            "compare ade_gt across configs to confirm numerical equivalence."
        ),
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n" + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in results.items()}, indent=2))
    print("[write]", args.out)


if __name__ == "__main__":
    main()
