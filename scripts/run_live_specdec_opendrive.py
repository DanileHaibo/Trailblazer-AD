#!/usr/bin/env python3
"""Live OpenDriveVLA+UniAD specdec: one teacher-force of draft [(x,y),...] vs official generate."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OD_ROOT = Path(os.environ.get("TRAJSPEC_OPENDRIVE_REPO", "/home/xinhchen2/hhb/OpenDriveVLA"))
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

from trajspec.accept import AcceptConfig
from trajspec.b2d_data import l2_horizon
from trajspec.draft_head import DraftHead, pack_extra, pack_speed
from trajspec.eval_loop import l2_uniad_at
from trajspec.nusc_data import VAL_PKL, load_infos, pack_kinematics, token_index
from trajspec.opendrive_live import spec_generate_opendrive
from trajspec.serialize import od_to_ego_xy

CKPT = os.environ.get("TRAJSPEC_OPENDRIVE_CKPT", "/data2/hf_overlap_models/OpenDriveVLA-0.5B")
HEAD = Path(os.environ.get("TRAJSPEC_OPENDRIVE_HEAD", str(ROOT / "weights/draft_opendrive_nusc.pt")))


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def summarize(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 2),
        "p50": round(float(np.median(a)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
    }


def extras_map():
    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    extra, speed, gt = {}, {}, {}
    for tok, i in tok2i.items():
        s = pack_kinematics(infos[i])
        if s is None:
            continue
        extra[tok] = pack_extra(s)
        speed[tok] = pack_speed(s)
        g = s.get("gt")
        gt[tok] = torch.as_tensor(g[:6], dtype=torch.float32) if g is not None else torch.zeros(6, 2)
    return extra, speed, gt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--start", type=int, default=1500, help="Skip mixed-coord cache prefix")
    p.add_argument("--out", default=str(ROOT / "results/live_specdec_opendrive.json"))
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--head", default=str(HEAD))
    args = p.parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    disable_torch_init()
    device = torch.device("cuda:0")
    tokenizer, model, _, _ = load_pretrained_model(
        args.ckpt,
        model_base=None,
        model_name="llava_qwen",
        device_map=device,
        multimodal=True,
        attn_implementation="sdpa",
        overwrite_config={"image_aspect_ratio": "pad", "vision_tower_test_mode": True},
    )
    model.float().to(device).eval()

    blob = torch.load(args.head, map_location="cpu", weights_only=False)
    head = DraftHead(int(blob["hidden_size"]), horizon=6, width=512).to(device)
    sd = blob.get("state_dict") or blob.get("head")
    head.load_state_dict(sd)
    head.eval()
    cfg = AcceptConfig(lon_m=0.5, lat_m=0.3)
    extra_map, speed_map, gt_map = extras_map()

    uniad_cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
    data_args = DataArguments(data_path=None, lazy_preprocess=True, frames_upbound=32)
    dataset = LLaVANuScenesDataset(
        tokenizer, data_args, uniad_cfg.data.test, llava_test_mode=True, use_uniad_pth=False
    )
    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
    n_take = args.n + args.warmup
    idxs = list(range(args.start, min(args.start + n_take, len(dataset))))
    loader = DataLoader(Subset(dataset, idxs), batch_size=1, collate_fn=collator, num_workers=0)

    rows = []
    for i, data in enumerate(loader):
        sample_id = data["id"]
        if isinstance(sample_id, (list, tuple)):
            sample_id = sample_id[0]
        token = str(sample_id).replace("_trajectory", "")
        data = move_data_to_device(data, device)
        extra = extra_map.get(token, torch.zeros(16)).float().to(device)
        if extra.numel() < 16:
            extra = torch.nn.functional.pad(extra.view(-1), (0, 16 - extra.numel()))
        speed = speed_map.get(token, torch.zeros(1)).float().to(device)
        gt = gt_map.get(token, torch.zeros(6, 2)).numpy().astype(np.float32)
        timed = i >= args.warmup

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

        t_gen, cont = sync_ms(_gen)
        ans = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        teacher_od = torch.tensor(retrieve_traj(ans), dtype=torch.float32)
        teacher_ego = od_to_ego_xy(teacher_od).numpy()

        spec_xy, spec_info = spec_generate_opendrive(
            model, tokenizer, data, head, extra, speed, cfg, horizon=6
        )
        rec = {
            "token": token,
            "t_official_generate_ms": round(t_gen, 2),
            "n_accept": spec_info["n_accept"],
            "t_spec_e2e_ms": spec_info["t_spec_e2e_ms"],
            "t_prefill_ms": spec_info["t_prefill_ms"],
            "t_verify_ms": spec_info["t_verify_ms"],
            "t_tail_ms": spec_info["t_tail_ms"],
            "t_draft_ms": spec_info["t_draft_ms"],
            "n_draft_tokens": spec_info["n_draft_tokens"],
            "n_serial_tail": spec_info["n_serial_tail"],
            "teacher": {**l2_horizon(teacher_ego, gt), **l2_uniad_at(teacher_ego, gt)},
            "spec": {**l2_horizon(spec_xy, gt), **l2_uniad_at(spec_xy, gt)},
            "ade_spec_vs_teacher": round(float(np.linalg.norm(spec_xy - teacher_ego[:6], axis=-1).mean()), 4),
            "draft_text": spec_info["draft_text"],
            "teacher_verify_text": spec_info["teacher_verify_text"][:180],
            "final_text": spec_info["final_text"][:180],
        }
        if timed:
            rows.append(rec)
            print(
                f"[{len(rows)}/{args.n}] gen={t_gen:.0f}ms spec={rec['t_spec_e2e_ms']:.0f}ms "
                f"pre={rec['t_prefill_ms']:.0f}ms ver={rec['t_verify_ms']:.0f}ms "
                f"tail={rec['t_tail_ms']:.0f}ms accept={rec['n_accept']} "
                f"ade(spec,teacher)={rec['ade_spec_vs_teacher']}",
                flush=True,
            )
        else:
            print(f"[warmup] gen={t_gen:.0f}ms spec={rec['t_spec_e2e_ms']:.0f}ms accept={rec['n_accept']}", flush=True)
        if len(rows) >= args.n:
            break

    spec_t = np.array([r["t_spec_e2e_ms"] for r in rows])
    gen_t = np.array([r["t_official_generate_ms"] for r in rows])
    nacc = np.array([r["n_accept"] for r in rows], dtype=np.float64)
    summary = {
        "model": "OpenDriveVLA-0.5B+UniAD",
        "mode": "live_teacher_force_verify",
        "n": len(rows),
        "start_index": args.start,
        "accept": {"lon_m": 0.5, "lat_m": 0.3, "mean_n_accept": round(float(nacc.mean()), 3)},
        "measured_ms": {
            "official_generate": summarize(gen_t),
            "spec_e2e": summarize(spec_t),
            "prefill": summarize([r["t_prefill_ms"] for r in rows]),
            "verify": summarize([r["t_verify_ms"] for r in rows]),
            "tail": summarize([r["t_tail_ms"] for r in rows]),
            "draft": summarize([r["t_draft_ms"] for r in rows]),
        },
        "speedup_vs_official_generate": round(float(gen_t.mean() / spec_t.mean()), 3),
        "mean_ade_spec_vs_teacher": round(float(np.mean([r["ade_spec_vs_teacher"] for r in rows])), 4),
        "note": "Prefill runs UniAD+VLM once with KV cache. Verify is one teacher-force of the draft waypoint string. Tail is serial greedy only after geometric reject.",
        "rows": rows,
    }
    Path(out_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    print("[write]", out_path, flush=True)


if __name__ == "__main__":
    main()
