#!/usr/bin/env python3
"""Cache official OpenDriveVLA+UniAD last-token hiddens on nuScenes val.

Teacher trajectories are read from the already-finished plan_conv.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

OD_ROOT = Path("/home/xinhchen2/hhb/OpenDriveVLA")
os.chdir(OD_ROOT)
sys.path.insert(0, str(OD_ROOT))
sys.path.insert(0, str(OD_ROOT / "drivevla"))
sys.path.insert(0, "/home/xinhchen2/hhb/TrajSpecDec")

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

from trajspec.draft_head import pack_extra, pack_speed
from trajspec.nusc_data import VAL_PKL, load_infos, pack_sample, token_index

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"
PRED = Path("/home/xinhchen2/hhb/TrajSpecDec/results/opendrive_nusc_plan_conv.json")


def od_xy_to_ego(xy: torch.Tensor) -> torch.Tensor:
    """OpenDrive/UniAD lidar (x right, y forward) -> ego (+X forward, +Y left)."""
    xy = xy.reshape(-1, 2).float()
    return torch.stack([xy[:, 1], -xy[:, 0]], dim=-1)


def load_preds(path: Path) -> dict:
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            token = r["id"].replace("_trajectory", "")
            ans = r["answer"][0] if isinstance(r["answer"], list) else r["answer"]
            out[token] = od_xy_to_ego(torch.tensor(retrieve_traj(ans), dtype=torch.float32))
    return out


def extras_by_token():
    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    extra, speed, gt = {}, {}, {}
    for tok, i in tok2i.items():
        s = pack_sample(infos, i, tok2i)
        if s is None:
            continue
        extra[tok] = pack_extra(s)
        speed[tok] = pack_speed(s)
        gt[tok] = torch.as_tensor(s["gt"][:6], dtype=torch.float32)
    return extra, speed, gt


def last_hidden(model, data):
    ids = data["input_ids"]
    (
        _,
        position_ids,
        attention_mask,
        _,
        inputs_embeds,
        _,
    ) = model.prepare_inputs_labels_for_multimodal_uniad_vlm(
        ids,
        None,
        None,
        None,
        None,
        None,
        uniad_data=data.get("uniad_data"),
        uniad_pth=data.get("uniad_pth"),
        qa_instance_ind=data.get("qa_instance_ind"),
    )
    out = model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states[-1] if hasattr(out, "hidden_states") else out[0]
    return hs[:, -1].float().cpu()[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/home/xinhchen2/hhb/TrajSpecDec/results/cache_opendrive_nusc_val.pt")
    p.add_argument("--max-samples", type=int, default=0)
    args = p.parse_args()

    preds = load_preds(PRED)
    extra_map, speed_map, gt_map = extras_by_token()
    print(f"[od-cache] preds={len(preds)} extras={len(extra_map)}", flush=True)

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
    n = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))
    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)

    out = Path(args.out)
    rows = []
    if out.exists():
        rows = torch.load(out, map_location="cpu", weights_only=False)["rows"]
        print(f"[resume] {len(rows)}", flush=True)
    done = {r["token"] for r in rows}

    start = len(rows)
    if start > n:
        start = n
    print(f"[resume] subset start={start}/{n} already={len(done)}", flush=True)
    run_ds = Subset(dataset, list(range(start, n))) if start else dataset
    loader = DataLoader(run_ds, batch_size=1, collate_fn=collator, num_workers=0, shuffle=False)

    hidden_size = int(getattr(model.config, "hidden_size", 896))
    for i, data in enumerate(loader):
        if start + i >= n:
            break
        sample_id = data["id"]
        if isinstance(sample_id, (list, tuple)):
            sample_id = sample_id[0]
        token = str(sample_id).replace("_trajectory", "")
        if token in done:
            continue
        if token not in preds:
            print(f"[skip] no pred {token}", flush=True)
            continue
        data = move_data_to_device(data, device)
        with torch.inference_mode():
            hidden = last_hidden(model, data)
        extra = extra_map.get(token, torch.zeros(16))
        speed = speed_map.get(token, torch.zeros(1))
        gt = gt_map.get(token, torch.zeros(6, 2))
        rows.append(
            {
                "token": token,
                "hidden": hidden.cpu().float(),
                "teacher": preds[token][:6].float(),
                "gt": gt[:6].float(),
                "extra": extra.float(),
                "speed": speed.float(),
            }
        )
        done.add(token)
        if len(rows) % 20 == 0 or len(rows) == 1:
            print(f"[od-cache] {len(rows)}/{n} {token[:8]}", flush=True)
        if len(rows) % 20 == 0:
            torch.save({"rows": rows, "hidden_size": hidden.shape[-1]}, out)
    torch.save({"rows": rows, "hidden_size": rows[0]["hidden"].shape[-1] if rows else hidden_size}, out)
    print(f"[done] {out} n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
