#!/usr/bin/env python3
"""Single-GPU official OpenDriveVLA inference on nuScenes (UniAD online, no DeepSpeed)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

OD_ROOT = Path("/home/xinhchen2/hhb/OpenDriveVLA")
os.chdir(OD_ROOT)
sys.path.insert(0, str(OD_ROOT))
sys.path.insert(0, str(OD_ROOT / "drivevla"))

import torch
from mmengine import Config
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data_utils.nuscenes_llava_datacollector import DataCollatorForLLaVANuScenesDataset
from data_utils.nuscenes_llava_dataset import LLaVANuScenesDataset
from llava.model.builder import load_pretrained_model
from llava.train.train import DataArguments
from llava.utils import disable_torch_init
from utils.tensor_utils import move_data_to_device

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=CKPT)
    p.add_argument("--output", default="/home/xinhchen2/hhb/TrajSpecDec/results/opendrive_nusc_plan_conv.json")
    p.add_argument("--max-samples", type=int, default=0, help="0 = full val")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--resume", action="store_true", help="append to an existing jsonl, skipping done ids")
    return p.parse_args()


def main():
    args = parse_args()
    disable_torch_init()
    device = torch.device("cuda:0")

    print("[od] loading model", args.model_path, flush=True)
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path,
        model_base=None,
        model_name="llava_qwen",
        device_map=device,
        multimodal=True,
        attn_implementation=args.attn_implementation,
        overwrite_config={"image_aspect_ratio": "pad", "vision_tower_test_mode": True},
    )
    # UniAD BatchNorm cannot mix fp16 weights with autocast activations.
    model.float().to(device)
    model.eval()
    print("[od] model ready (fp32)", flush=True)

    uniad_cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
    data_args = DataArguments(data_path=None, lazy_preprocess=True, frames_upbound=32)
    print("[od] building nuScenes dataset (cache + infos)...", flush=True)
    dataset = LLaVANuScenesDataset(
        tokenizer,
        data_args,
        uniad_cfg.data.test,
        llava_test_mode=True,
        use_uniad_pth=False,
    )
    n = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))
    if n < len(dataset):
        dataset = Subset(dataset, list(range(n)))
    print(f"[od] dataset size={n} (full={len(dataset) if args.max_samples<=0 else 'subset'})", flush=True)

    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collator,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"[resume] {len(done_ids)} already written, appending", flush=True)
    elif out_path.exists():
        out_path.unlink()

    t0 = time.time()
    for i, data in enumerate(tqdm(loader, ncols=100)):
        _sid = data["id"]
        if isinstance(_sid, (list, tuple)):
            _sid = _sid[0]
        if args.resume and _sid in done_ids:
            continue
        data = move_data_to_device(data, device)
        with torch.inference_mode():
            cont = model.generate(
                data["input_ids"],
                uniad_data=data.get("uniad_data"),
                uniad_pth=data.get("uniad_pth"),
                qa_instance_ind=data.get("qa_instance_ind"),
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
                num_beams=1,
            )
        answer = tokenizer.batch_decode(cont, skip_special_tokens=True)
        sample_id = data["id"]
        if isinstance(sample_id, (list, tuple)):
            sample_id = sample_id[0]
        result = {"id": sample_id, "question": data["question"], "answer": answer}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        if i == 0:
            print("[od] first answer:", answer[0][:400] if answer else None, flush=True)
        elapsed = time.time() - t0
        print(f"[od] {i+1}/{n} id={sample_id} t={elapsed:.1f}s last={elapsed/(i+1):.2f}s/it", flush=True)

    print(f"[od] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
