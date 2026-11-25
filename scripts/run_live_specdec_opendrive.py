#!/usr/bin/env python3
"""Live OpenDriveVLA+UniAD specdec: one teacher-force of draft [(x,y),...] vs official generate."""
from __future__ import annotations

import argparse
import gc
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
from trajspec.metrics import l2_horizon, l2_uniad_at
from trajspec.draft_head import DraftHead, pack_extra, pack_speed
from trajspec.nusc_data import VAL_PKL, load_infos, pack_kinematics, token_index
from trajspec.opendrive_live import reset_uniad_temporal, spec_generate_opendrive
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
    extra, speed, gt, gt_mask = {}, {}, {}, {}
    for tok, i in tok2i.items():
        s = pack_kinematics(infos[i])
        if s is None:
            continue
        extra[tok] = pack_extra(s)
        speed[tok] = pack_speed(s)
        g = s.get("gt")
        gt[tok] = torch.as_tensor(g[:6], dtype=torch.float32) if g is not None else torch.zeros(6, 2)
        m = s.get("gt_mask")
        gt_mask[tok] = (
            np.asarray(m[:6], dtype=np.float32)
            if m is not None
            else np.ones(6, dtype=np.float32)
        )
    return extra, speed, gt, gt_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--start", type=int, default=1500, help="Skip mixed-coord cache prefix")
    p.add_argument("--shard", type=int, default=0, help="Shard id for multi-GPU (stride split)")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--out", default=str(ROOT / "results/live_specdec_opendrive.json"))
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--head", default=str(HEAD))
    p.add_argument("--save-every", type=int, default=50, help="Checkpoint partial results every N timed frames")
    p.add_argument("--resume", action="store_true", help="Resume from existing --out rows (skip done tokens)")
    args = p.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard < args.num_shards):
        raise SystemExit(f"bad shard={args.shard} num_shards={args.num_shards}")
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    done_tokens: set[str] = set()
    if args.resume and out_path.exists():
        prev = json.loads(out_path.read_text())
        rows = list(prev.get("rows") or [])
        done_tokens = {str(r["token"]) for r in rows}
        print(f"[resume] loaded {len(rows)} rows from {out_path}", flush=True)

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
    extra_map, speed_map, gt_map, gt_mask_map = extras_map()

    uniad_cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
    data_args = DataArguments(data_path=None, lazy_preprocess=True, frames_upbound=32)
    dataset = LLaVANuScenesDataset(
        tokenizer, data_args, uniad_cfg.data.test, llava_test_mode=True, use_uniad_pth=False
    )
    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
    # Contiguous prefix from --start, then stride-split across GPUs so dual-GPU
    # covers the full val without overlapping tokens.
    pool = list(range(args.start, len(dataset)))
    if args.num_shards > 1:
        pool = [i for i in pool if i % args.num_shards == args.shard]
    n_take = args.n + args.warmup
    idxs = pool[:n_take]
    print(
        f"[shard] id={args.shard}/{args.num_shards} start={args.start} "
        f"pool={len(pool)} take={len(idxs)} target_timed={args.n}",
        flush=True,
    )
    loader = DataLoader(Subset(dataset, idxs), batch_size=1, collate_fn=collator, num_workers=0)

    # When resuming, skip warmup — already past the prefix.
    skip_warmup = bool(done_tokens)
    for i, data in enumerate(loader):
        if len(rows) >= args.n:
            break
        sample_id = data["id"]
        if isinstance(sample_id, (list, tuple)):
            sample_id = sample_id[0]
        token = str(sample_id).replace("_trajectory", "")
        if token in done_tokens:
            continue
        data = move_data_to_device(data, device)
        extra = extra_map.get(token, torch.zeros(16)).float().to(device)
        if extra.numel() < 16:
            extra = torch.nn.functional.pad(extra.view(-1), (0, 16 - extra.numel()))
        speed = speed_map.get(token, torch.zeros(1)).float().to(device)
        gt = gt_map.get(token, torch.zeros(6, 2)).numpy().astype(np.float32)
        gmask = gt_mask_map.get(token, np.ones(6, dtype=np.float32))
        timed = skip_warmup or i >= args.warmup

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

        try:
            # UniAD prev_frame_info persists across calls; reset so generate↔spec share
            # the same cold temporal state (otherwise accept/L2 are contaminated).
            reset_uniad_temporal(model)
            t_gen, cont = sync_ms(_gen)
            ans = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            try:
                teacher_od = torch.tensor(retrieve_traj(ans), dtype=torch.float32)
            except Exception:
                from trajspec.serialize import parse_opendrive_traj

                teacher_od = parse_opendrive_traj(ans, horizon=6)
            teacher_ego = od_to_ego_xy(teacher_od).numpy()

            reset_uniad_temporal(model)
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
                "teacher": {**l2_horizon(teacher_ego, gt, gmask), **l2_uniad_at(teacher_ego, gt, gmask)},
                "spec": {**l2_horizon(spec_xy, gt, gmask), **l2_uniad_at(spec_xy, gt, gmask)},
                "ade_spec_vs_teacher": round(float(np.linalg.norm(spec_xy - teacher_ego[:6], axis=-1).mean()), 4),
                "draft_text": spec_info["draft_text"],
                "teacher_verify_text": spec_info["teacher_verify_text"][:180],
                "final_text": spec_info["final_text"][:180],
            }
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[skip] token={token} err={msg}", flush=True)
            # CUDA/driver faults must NOT be "skipped" — continuing after an
            # unknown CUDA error is what wedged the L20 previously. Checkpoint
            # and exit cleanly so the GPU stays recoverable.
            if isinstance(e, torch.cuda.CudaError) or "CUDA" in msg or "cuda" in msg:
                Path(out_path).write_text(json.dumps(_build_summary(rows, args), ensure_ascii=False, indent=2))
                print(f"[abort-cuda] saved n={len(rows)} -> {out_path}", flush=True)
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise SystemExit(2) from e
            continue

        if timed:
            rows.append(rec)
            done_tokens.add(token)
            print(
                f"[shard{args.shard} {len(rows)}/{args.n}] gen={t_gen:.0f}ms spec={rec['t_spec_e2e_ms']:.0f}ms "
                f"pre={rec['t_prefill_ms']:.0f}ms ver={rec['t_verify_ms']:.0f}ms "
                f"tail={rec['t_tail_ms']:.0f}ms accept={rec['n_accept']} "
                f"ade(spec,teacher)={rec['ade_spec_vs_teacher']}",
                flush=True,
            )
            if args.save_every > 0 and len(rows) % args.save_every == 0:
                Path(out_path).write_text(json.dumps(_build_summary(rows, args), ensure_ascii=False, indent=2))
                print(f"[ckpt] {out_path} n={len(rows)}", flush=True)
                # Keep allocator tidy on long runs (UniAD activations accumulate).
                gc.collect()
                torch.cuda.empty_cache()
        else:
            print(f"[warmup] gen={t_gen:.0f}ms spec={rec['t_spec_e2e_ms']:.0f}ms accept={rec['n_accept']}", flush=True)

    summary = _build_summary(rows, args)
    Path(out_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    print("[write]", out_path, flush=True)


def _build_summary(rows, args):
    spec_t = np.array([r["t_spec_e2e_ms"] for r in rows], dtype=np.float64)
    gen_t = np.array([r["t_official_generate_ms"] for r in rows], dtype=np.float64)
    nacc = np.array([r["n_accept"] for r in rows], dtype=np.float64)
    return {
        "model": "OpenDriveVLA-0.5B+UniAD",
        "mode": "live_teacher_force_verify",
        "n": len(rows),
        "start_index": args.start,
        "shard": getattr(args, "shard", 0),
        "num_shards": getattr(args, "num_shards", 1),
        "accept": {"lon_m": 0.5, "lat_m": 0.3, "mean_n_accept": round(float(nacc.mean()), 3) if len(rows) else 0.0},
        "measured_ms": {
            "official_generate": summarize(gen_t),
            "spec_e2e": summarize(spec_t),
            "prefill": summarize([r["t_prefill_ms"] for r in rows]),
            "verify": summarize([r["t_verify_ms"] for r in rows]),
            "tail": summarize([r["t_tail_ms"] for r in rows]),
            "draft": summarize([r["t_draft_ms"] for r in rows]),
        },
        "speedup_vs_official_generate": round(float(gen_t.mean() / spec_t.mean()), 3) if len(rows) and spec_t.mean() > 0 else 0.0,
        "mean_ade_spec_vs_teacher": round(float(np.mean([r["ade_spec_vs_teacher"] for r in rows])), 4) if rows else 0.0,
        "teacher_l2": {
            k: round(float(np.mean([r["teacher"][k] for r in rows if k in r["teacher"]])), 4)
            for k in ("l2_1s", "l2_2s", "l2_3s", "ade", "l2_at_1s", "l2_at_2s", "l2_at_3s")
        },
        "spec_l2": {
            k: round(float(np.mean([r["spec"][k] for r in rows if k in r["spec"]])), 4)
            for k in ("l2_1s", "l2_2s", "l2_3s", "ade", "l2_at_1s", "l2_at_2s", "l2_at_3s")
        },
        "note": "Prefill runs UniAD+VLM once with KV cache. Verify is one teacher-force of the draft waypoint string. Tail is serial greedy only after geometric reject. UniAD temporal state is reset before generate and before spec.",
        "rows": rows,
    }


if __name__ == "__main__":
    main()
