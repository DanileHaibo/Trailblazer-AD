#!/usr/bin/env python3
"""PD-VLA-style parallel (Jacobi) decoding for OpenDriveVLA-0.5B + UniAD.

PD-VLA (IROS 2025) turns AR decoding into a fixed-point iteration: initialize the
whole action chunk, then refresh every position at once and repeat until the chunk
stops changing. Its two ingredients are (i) bidirectional attention *inside* the
chunk and (ii) the fact that the trajectory chunk is short enough for the whole
iteration to be cheap.

Applied to OpenDriveVLA the action chunk is the textual waypoint string
`<traj_start>[(x,y),...]<traj_end>`, so we give the chunk a 4-D mask that is causal
over the prompt and fully connected over the chunk. Without that in-chunk
bidirectionality the iteration collapses to the trivial fixed point
`(0.00,0.00)` -- see `--mask causal` for the ablation that shows it.

Reported per sample: iterations to converge, latency, ADE vs the official greedy
output on the same frame.
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
from trajspec.opendrive_live import prefill_uniad
from trajspec.serialize import TRAJ_END, TRAJ_START, od_to_ego_xy, parse_opendrive_traj

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"
HORIZON = 6
SLOT = 66  # token length of a 6-waypoint answer, measured from official generate


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


def template_ids(tokenizer, slot: int) -> list[int]:
    """Canonical zero-trajectory answer, padded/truncated to exactly `slot` tokens.

    This is PD-VLA's "default action" initialization: it carries no teacher
    information beyond the token layout.
    """
    inner = ",".join("(0.00,0.00)" for _ in range(HORIZON))
    ids = tokenizer(f"{TRAJ_START}[{inner}]{TRAJ_END}", add_special_tokens=False)["input_ids"]
    if isinstance(ids[0], list):
        ids = ids[0]
    pad = tokenizer("0", add_special_tokens=False)["input_ids"]
    if isinstance(pad[0], list):
        pad = pad[0]
    if len(ids) >= slot:
        return list(ids[:slot])
    return list(ids) + [int(pad[0])] * (slot - len(ids))


def build_mask(prefix_len: int, chunk_len: int, dtype, device, chunk_bidir: bool):
    """[1, 1, L, L] additive mask; prompt causal, chunk either causal or all-visible."""
    L = prefix_len + chunk_len
    keep = torch.zeros(L, L, dtype=torch.bool, device=device)
    keep[:, :prefix_len] = True  # everyone sees the prompt
    idx = torch.arange(prefix_len, L, device=device)
    if chunk_bidir:
        keep[prefix_len:, prefix_len:] = True
    else:
        keep[prefix_len:, prefix_len:] = idx[:, None] >= idx[None, :]
    masked = torch.finfo(dtype).min
    m = torch.zeros(L, L, dtype=dtype, device=device)
    m[~keep] = masked
    return m.view(1, 1, L, L)


def extras_map():
    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    gt = {}
    for tok, i in tok2i.items():
        s = pack_kinematics(infos[i])
        if s is None:
            continue
        g = s.get("gt")
        gt[tok] = torch.as_tensor(g[:HORIZON], dtype=torch.float32) if g is not None else torch.zeros(HORIZON, 2)
    return gt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--max-iter", type=int, default=12)
    p.add_argument("--slot", type=int, default=SLOT)
    p.add_argument("--mask", choices=["bidir", "causal"], default="bidir")
    p.add_argument("--out", default=str(ROOT / "results/live_pdvla_opendrive.json"))
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
    gt_map = extras_map()
    embed = model.get_model().embed_tokens

    uniad_cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
    data_args = DataArguments(data_path=None, lazy_preprocess=True, frames_upbound=32)
    dataset = LLaVANuScenesDataset(
        tokenizer, data_args, uniad_cfg.data.test, llava_test_mode=True, use_uniad_pth=False
    )
    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
    idxs = list(range(args.start, min(args.start + args.n + args.warmup, len(dataset))))
    loader = DataLoader(Subset(dataset, idxs), batch_size=1, collate_fn=collator, num_workers=0)

    init = template_ids(tokenizer, args.slot)
    rows = []
    for i, data in enumerate(loader):
        sample_id = data["id"]
        if isinstance(sample_id, (list, tuple)):
            sample_id = sample_id[0]
        token = str(sample_id).replace("_trajectory", "")
        data = move_data_to_device(data, device)
        gt = gt_map.get(token, torch.zeros(HORIZON, 2)).numpy().astype(np.float32)

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
        teacher_ego = od_to_ego_xy(torch.tensor(retrieve_traj(ans), dtype=torch.float32)).numpy()

        def _pre():
            return prefill_uniad(model, data)

        t_pre, (pre, meta) = sync_ms(_pre)
        prompt_embeds = meta["inputs_embeds"].float()
        prefix_len = int(prompt_embeds.shape[1])
        mask = build_mask(prefix_len, args.slot, prompt_embeds.dtype, device, args.mask == "bidir")
        chunk_pos = torch.arange(prefix_len, prefix_len + args.slot, device=device).unsqueeze(0)
        base_pos = meta["position_ids"]
        if base_pos is None:
            base_pos = torch.arange(prefix_len, device=device).unsqueeze(0)
        pos_ids = torch.cat([base_pos.to(device).view(1, -1), chunk_pos], dim=1)

        guess = torch.tensor([init], device=device, dtype=torch.long)
        iters = 0
        changed = args.slot
        t_iters = 0.0
        converged = False
        for _ in range(args.max_iter):
            def _jstep():
                emb = torch.cat([prompt_embeds, embed(guess).float()], dim=1)
                return model(
                    inputs_embeds=emb,
                    attention_mask=mask,
                    position_ids=pos_ids,
                    use_cache=False,
                )

            t_it, out = sync_ms(_jstep)
            t_iters += t_it
            iters += 1
            logits = out.logits[0, prefix_len - 1 : prefix_len - 1 + args.slot].float()
            new = logits.argmax(dim=-1, keepdim=True).view(1, -1)
            changed = int((new != guess).sum().item())
            guess = new
            if changed == 0:
                converged = True
                break

        committed = [int(x) for x in guess[0].tolist()]
        text = tokenizer.decode(committed, skip_special_tokens=False)
        parse_ok = True
        try:
            spec_ego = od_to_ego_xy(parse_opendrive_traj(text, horizon=HORIZON)).numpy()
        except ValueError:
            # Parallel decoding has no syntax guarantee: a fixed point can hold a
            # malformed number like "7..". Score that frame as a hard parse failure.
            spec_ego = np.zeros((HORIZON, 2), dtype=np.float32)
            parse_ok = False

        rec = {
            "token": token,
            "t_official_generate_ms": round(t_gen, 2),
            "t_prefill_ms": round(t_pre, 2),
            "t_iters_ms": round(t_iters, 2),
            "t_pd_e2e_ms": round(t_pre + t_iters, 2),
            "n_iters": iters,
            "converged": converged,
            "n_changed_last": changed,
            "parse_ok": parse_ok,
            "ade_vs_teacher": round(
                float(np.linalg.norm(spec_ego[:HORIZON] - teacher_ego[:HORIZON], axis=-1).mean()), 4
            ),
            "ade_gt": round(float(l2_horizon(spec_ego[:HORIZON], gt)["ade"]), 4),
            "teacher_ade_gt": round(float(l2_horizon(teacher_ego[:HORIZON], gt)["ade"]), 4),
            "n_new_tokens_official": int(cont.shape[1]),
            "text_head": text[:80],
        }
        if i >= args.warmup:
            rows.append(rec)
        print(
            f"[{len(rows)}/{args.n}] gen={t_gen:.0f} pre={t_pre:.0f} iters={iters} "
            f"({t_iters:.0f}ms) conv={converged} ade_T={rec['ade_vs_teacher']:.3f}",
            flush=True,
        )
        if len(rows) >= args.n:
            break

    pd_e2e = np.array([r["t_pd_e2e_ms"] for r in rows])
    ar_e2e = np.array([r["t_official_generate_ms"] for r in rows])
    summary = {
        "model": "OpenDriveVLA-0.5B+UniAD",
        "mode": f"pdvla_jacobi_{args.mask}",
        "device": torch.cuda.get_device_name(0),
        "n": len(rows),
        "start_index": args.start,
        "slot": args.slot,
        "max_iter": args.max_iter,
        "converged_rate": round(float(np.mean([r["converged"] for r in rows])), 3),
        "parse_ok_rate": round(float(np.mean([r["parse_ok"] for r in rows])), 3),
        "mean_n_iters": round(float(np.mean([r["n_iters"] for r in rows])), 2),
        "ar_e2e_ms": round(float(ar_e2e.mean()), 2),
        "pd_e2e_ms": round(float(pd_e2e.mean()), 2),
        "prefill_ms": round(float(np.mean([r["t_prefill_ms"] for r in rows])), 2),
        "speedup_e2e": round(float(ar_e2e.mean() / max(pd_e2e.mean(), 1e-6)), 3),
        "ade_vs_teacher": round(float(np.mean([r["ade_vs_teacher"] for r in rows])), 4),
        "ade_gt": round(float(np.mean([r["ade_gt"] for r in rows])), 4),
        "teacher_ade_gt": round(float(np.mean([r["teacher_ade_gt"] for r in rows])), 4),
        "measured_ms": {"pd_e2e": summarize(pd_e2e), "ar_e2e": summarize(ar_e2e)},
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
