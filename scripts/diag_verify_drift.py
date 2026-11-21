#!/usr/bin/env python3
"""How far does SD's teacher-force verify drift from official generate()?

The unified table shows a suspicious pattern: frames with n_accept == 0 (full
reject, so the output should be the teacher trajectory verbatim) still differ from
AR greedy by 0.15 m ADE on average. That can only come from the verify path
computing different logits than `generate()` does.

Candidates:
  * attention mask dtype (long vs float) in the incremental `_continue` call
  * position ids restarting from cache_len instead of the prompt's own ids
  * `past_key_values` reuse vs recompute

This script measures, per frame, the L2 gap between:
  A. official generate() output
  B. first-token greedy chain through `_continue` on the cached prefill (the SD
     verify path), decoded with the same horizon
and dumps the token-level first difference, so the drift is attributable rather
than guessed.
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

from trajspec.metrics import l2_horizon
from trajspec.draft_head import DraftHead
from trajspec.nusc_data import VAL_PKL, load_infos, pack_kinematics, token_index
from trajspec.opendrive_live import _cache_seq_len, _continue, prefill_uniad
from trajspec.serialize import TRAJ_END, TRAJ_START, format_opendrive_answer, od_to_ego_xy

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"
HEAD = ROOT / "results/specdec_opendrive_heldout.pt"
HORIZON = 6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--out", default=str(ROOT / "results/verify_drift.json"))
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
    blob = torch.load(HEAD, map_location="cpu", weights_only=False)
    head = DraftHead(int(blob["hidden_size"]), horizon=HORIZON, width=512).to(device)
    head.load_state_dict(blob["state_dict"])
    head.eval()

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
        sample_id = data["id"]
        if isinstance(sample_id, (list, tuple)):
            sample_id = sample_id[0]
        token = str(sample_id).replace("_trajectory", "")
        data = move_data_to_device(data, device)

        # A. official generate
        with torch.inference_mode():
            cont = model.generate(
                data["input_ids"],
                uniad_data=data.get("uniad_data"),
                uniad_pth=data.get("uniad_pth"),
                qa_instance_ind=data.get("qa_instance_ind"),
                do_sample=False,
                temperature=0,
                max_new_tokens=args.max_new,
                num_beams=1,
            )
        ar_ids = cont[0].tolist()
        ar_text = tokenizer.decode(ar_ids, skip_special_tokens=False)
        ar_od = torch.tensor(retrieve_traj(tokenizer.batch_decode(cont, skip_special_tokens=True)[0]))

        # B. incremental chain from the cached prefill (SD verify path)
        pre, meta = prefill_uniad(model, data)
        cache_len = _cache_seq_len(pre.past_key_values) or int(meta["prompt_len"])
        attn_dtype = meta["attention_mask"].dtype if meta["attention_mask"] is not None else torch.long
        logits = pre.logits[0, -1].float()
        past = pre.past_key_values
        chain = []
        filled = cache_len
        for _ in range(min(len(ar_ids), args.max_new)):
            tid = int(logits.view(-1).argmax().item())
            chain.append(tid)
            step = torch.tensor([[tid]], device=device, dtype=torch.long)
            out = _continue(model, step, past, filled, attn_dtype)
            filled += 1
            logits = out.logits[0, -1].float()
            past = out.past_key_values
        chain_text = tokenizer.decode(chain, skip_special_tokens=False)
        chain_od = torch.tensor(retrieve_traj(chain_text))

        k = min(len(ar_ids), len(chain))
        first_diff = next((j for j in range(k) if ar_ids[j] != chain[j]), -1)
        ar_ego = od_to_ego_xy(ar_od.float()).numpy()
        ch_ego = od_to_ego_xy(chain_od.float()).numpy()
        l2 = float(np.linalg.norm(ar_ego[:HORIZON] - ch_ego[:HORIZON], axis=-1).mean())
        rec = {
            "token": token,
            "n_ar_tokens": len(ar_ids),
            "n_chain_tokens": len(chain),
            "prefix_len": int(meta["prompt_len"]),
            "cache_len": int(cache_len),
            "attn_dtype": str(attn_dtype),
            "first_token_diff": int(first_diff),
            "token_match_upto": int(first_diff if first_diff >= 0 else k),
            "token_match_rate": round((first_diff if first_diff >= 0 else k) / max(k, 1), 4),
            "ade_ar_vs_chain": round(l2, 4),
            "ar_text": ar_text[:70],
            "chain_text": chain_text[:70],
        }
        if i >= args.warmup:
            rows.append(rec)
            print(
                f"[{len(rows)}/{args.n}] plen={rec['prefix_len']} cache={rec['cache_len']} "
                f"match_upto={rec['token_match_upto']}/{k} firstdiff={first_diff} "
                f"ADE(AR,chain)={l2:.4f}",
                flush=True,
            )
        if len(rows) >= args.n:
            break

    summary = {
        "model": "OpenDriveVLA-0.5B+UniAD",
        "n": len(rows),
        "start_index": args.start,
        "mean_token_match_rate": round(float(np.mean([r["token_match_rate"] for r in rows])), 4),
        "mean_ade_ar_vs_chain": round(float(np.mean([r["ade_ar_vs_chain"] for r in rows])), 4),
        "n_identical": int(sum(1 for r in rows if r["first_token_diff"] == -1)),
        "n_differ": int(sum(1 for r in rows if r["first_token_diff"] >= 0)),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
