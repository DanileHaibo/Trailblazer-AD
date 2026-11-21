#!/usr/bin/env python3
"""Live OpenDriveVLA+UniAD SD baselines. Shared propose, one tail per accept rule."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

OD_ROOT = Path("/home/xinhchen2/hhb/OpenDriveVLA")
ROOT = Path("/home/xinhchen2/hhb/TrajSpecDec")
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
from trajspec.metrics import l2_horizon
from trajspec.draft_head import DraftHead, pack_extra, pack_speed
from trajspec.eval_loop import l2_uniad_at
from trajspec.nusc_data import VAL_PKL, load_infos, pack_kinematics, token_index
from trajspec.opendrive_live import spec_commit_opendrive, spec_propose_opendrive
from trajspec.sd_baselines import GEOMETRIC, KERV_LITE, SCAFFOLD, SPECVLA_POSE, TOKEN_EXACT
from trajspec.serialize import od_to_ego_xy

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"
HEAD = ROOT / "results/specdec_opendrive_heldout.pt"
MODES = [
    ("Spec-VLA (exact)", TOKEN_EXACT),
    ("Scaffold-SD", SCAFFOLD),
    ("Spec-VLA (pose-relax)", SPECVLA_POSE),
    ("KERV-lite", KERV_LITE),
    ("TrajSpecDec (ours)", GEOMETRIC),
]


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def summarize(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": round(float(a.mean()), 2), "p50": round(float(np.median(a)), 2), "p90": round(float(np.percentile(a, 90)), 2)}


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
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--out", default=str(ROOT / "results/live_sd_baselines_opendrive.json"))
    args = p.parse_args()
    # We chdir into the OpenDriveVLA repo, so a relative --out would resolve there.
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
    head = DraftHead(int(blob["hidden_size"]), horizon=6, width=512).to(device)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    cfg = AcceptConfig(lon_m=0.5, lat_m=0.3)
    extra_map, speed_map, gt_map = extras_map()

    uniad_cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
    data_args = DataArguments(data_path=None, lazy_preprocess=True, frames_upbound=32)
    dataset = LLaVANuScenesDataset(
        tokenizer, data_args, uniad_cfg.data.test, llava_test_mode=True, use_uniad_pth=False
    )
    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
    idxs = list(range(args.start, min(args.start + args.n + args.warmup, len(dataset))))
    loader = DataLoader(Subset(dataset, idxs), batch_size=1, collate_fn=collator, num_workers=0)

    method_rows = {name: [] for name, _ in MODES}
    ar_rows = []
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
        rec_ar = {
            "token": token,
            "t_official_generate_ms": round(t_gen, 2),
            "ade_gt": round(float(l2_horizon(teacher_ego[:6], gt)["ade"]), 4),
        }
        pack = spec_propose_opendrive(model, tokenizer, data, head, extra, speed, horizon=6)
        rec_ar["t_prefill_ms"] = round(pack["t_prefill_ms"], 2)
        rec_ar["t_decode_ms"] = round(t_gen - pack["t_prefill_ms"], 2)
        if timed:
            ar_rows.append(rec_ar)

        for name, mode in MODES:
            spec_xy, info = spec_commit_opendrive(pack, cfg, accept_mode=mode)
            rec = {
                "token": token,
                "n_accept": info["n_accept"],
                "t_spec_e2e_ms": info["t_spec_e2e_ms"],
                "t_prefill_ms": info["t_prefill_ms"],
                "t_draft_ms": info["t_draft_ms"],
                "t_verify_ms": info["t_verify_ms"],
                "t_tail_ms": info["t_tail_ms"],
                "n_serial_tail": info["n_serial_tail"],
                "n_draft_tokens": info["n_draft_tokens"],
                "ade_vs_teacher": round(float(np.linalg.norm(spec_xy[:6] - teacher_ego[:6], axis=-1).mean()), 4),
                "ade_gt": round(float(l2_horizon(spec_xy[:6], gt)["ade"]), 4),
                "l2": {**l2_horizon(spec_xy[:6], gt), **l2_uniad_at(spec_xy[:6], gt)},
            }
            rec["t_decode_ms"] = round(rec["t_spec_e2e_ms"] - rec["t_prefill_ms"], 2)
            if timed:
                method_rows[name].append(rec)
        if timed:
            accs = {name: method_rows[name][-1]["n_accept"] for name, _ in MODES}
            print(
                f"[{len(ar_rows)}/{args.n}] gen={t_gen:.0f}ms spec_ours={method_rows['TrajSpecDec (ours)'][-1]['t_spec_e2e_ms']:.0f}ms acc={accs}",
                flush=True,
            )
        else:
            print(f"[warmup] gen={t_gen:.0f}ms", flush=True)
        if len(ar_rows) >= args.n:
            break

    def pack_method(rows):
        e2e = np.array([r["t_spec_e2e_ms"] for r in rows])
        dec = np.array([r["t_decode_ms"] for r in rows])
        return {
            "n": len(rows),
            "mean_n_accept": round(float(np.mean([r["n_accept"] for r in rows])), 3),
            "e2e_ms": round(float(e2e.mean()), 2),
            "decode_ms": round(float(dec.mean()), 2),
            "prefill_ms": round(float(np.mean([r["t_prefill_ms"] for r in rows])), 2),
            "tail_ms": round(float(np.mean([r["t_tail_ms"] for r in rows])), 2),
            "ade_vs_teacher": round(float(np.mean([r["ade_vs_teacher"] for r in rows])), 4),
            "ade_gt": round(float(np.mean([r["ade_gt"] for r in rows])), 4),
            "measured_ms": {"e2e": summarize(e2e), "decode": summarize(dec)},
        }

    ar_e2e = np.array([r["t_official_generate_ms"] for r in ar_rows])
    ar_dec = np.array([r["t_decode_ms"] for r in ar_rows])
    summary = {
        "model": "OpenDriveVLA-0.5B+UniAD",
        "mode": "live_sd_baselines",
        "n": len(ar_rows),
        "start_index": args.start,
        "device": torch.cuda.get_device_name(0),
        "accept": {"lon_m": 0.5, "lat_m": 0.3},
        "AR greedy": {
            "n": len(ar_rows),
            "mean_n_accept": 0.0,
            "e2e_ms": round(float(ar_e2e.mean()), 2),
            "decode_ms": round(float(ar_dec.mean()), 2),
            "prefill_ms": round(float(np.mean([r["t_prefill_ms"] for r in ar_rows])), 2),
            "ade_vs_teacher": 0.0,
            "ade_gt": round(float(np.mean([r["ade_gt"] for r in ar_rows])), 4),
        },
        "methods": {name: pack_method(method_rows[name]) for name, _ in MODES},
        "rows": {"ar": ar_rows, **{name: method_rows[name] for name, _ in MODES}},
    }
    ar = summary["AR greedy"]
    for name, m in summary["methods"].items():
        m["speedup_e2e"] = round(ar["e2e_ms"] / max(m["e2e_ms"], 1e-6), 3)
        m["speedup_decode"] = round(ar["decode_ms"] / max(m["decode_ms"], 1e-6), 3)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
