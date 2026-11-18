#!/usr/bin/env python3
"""One table, one frame set: 4 acceleration families vs TrajSpecDec.

Design note (this is the part that makes the table trustworthy)
---------------------------------------------------------------
OpenDriveVLA's UniAD tower carries persistent temporal state
(`prev_frame_info`: prev_bev / prev_pos / scene_token, plus `track_base.max_obj_id`).
When frames are visited by index -- stride sampling, a non-contiguous list, or just
calling `generate()` before a re-prefill -- the tower sees the wrong BEV context.
Measured: the same frame prefilled twice around a generate() call differs by 0.65 in
logit space and the decoded token chain flips after 5-20 tokens. Even with the state
reset, repeated tower calls on one frame yield different track counts (30/31/30),
so AR-vs-spec numbers collected naively are not comparable.

Therefore: **the vision tower runs exactly once per frame**, its result is passed to
every method as `uniad_pth` (which bypasses the tower inside
`prepare_inputs_labels_for_multimodal_uniad_vlm`). All rows then consume byte-identical
visual features, and the only thing that varies is the decode strategy -- which is
precisely the claim being tested. The shared tower cost is included in every row.

Families
--------
  1. Spec-VLA (exact)      speculative decoding, strict token match
  2. Spec-VLA (pose-relax) speculative decoding, relaxed acceptance in metres
  3. PD-VLA                parallel / non-autoregressive decoding (Jacobi)
  4. Pruning (oracle)      visual-token pruning upper bound (FastDriveVLA / Prune2Drive)
  5. TrajSpecDec (ours)    waypoint-level SD, geometric accept + seam repair
  -  AR greedy             reference row
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

from trajspec.accept import AcceptConfig
from trajspec.b2d_data import l2_horizon
from trajspec.draft_head import DraftHead
from trajspec.nusc_data import VAL_PKL, load_infos, pack_kinematics, token_index
from trajspec.opendrive_live import (
    TeacherDecoder,
    _cache_seq_len,
    _continue,
    prefill_uniad,
    reset_uniad_temporal,
    spec_commit_opendrive,
    spec_propose_opendrive,
)
from trajspec.sd_baselines import GEOMETRIC, SPECVLA_POSE, TOKEN_EXACT
from trajspec.serialize import TRAJ_END, TRAJ_START, od_to_ego_xy, parse_opendrive_traj

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"
HEAD = ROOT / "results/specdec_opendrive_heldout.pt"
HORIZON = 6
SLOT = 66


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def extras_map():
    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    from trajspec.draft_head import pack_extra, pack_speed

    extra, speed, gt = {}, {}, {}
    for tok, i in tok2i.items():
        s = pack_kinematics(infos[i])
        if s is None:
            continue
        extra[tok] = pack_extra(s)
        speed[tok] = pack_speed(s)
        g = s.get("gt")
        gt[tok] = torch.as_tensor(g[:HORIZON], dtype=torch.float32) if g is not None else torch.zeros(HORIZON, 2)
    return extra, speed, gt


def template_ids(tokenizer, slot: int) -> list[int]:
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


def build_chunk_mask(prefix_len: int, chunk_len: int, dtype, device):
    L = prefix_len + chunk_len
    keep = torch.zeros(L, L, dtype=torch.bool, device=device)
    keep[:, :prefix_len] = True
    keep[prefix_len:, prefix_len:] = True
    m = torch.zeros(L, L, dtype=dtype, device=device)
    m[~keep] = torch.finfo(dtype).min
    return m.view(1, 1, L, L)


@torch.no_grad()
def pdvla_parallel(model, tokenizer, data, init_ids, max_iter=24):
    """PD-VLA: Jacobi refresh of the whole waypoint chunk to a fixed point."""
    t_pre, (pre, meta) = sync_ms(lambda: prefill_uniad(model, data))
    prompt_embeds = meta["inputs_embeds"].float()
    prefix_len = int(prompt_embeds.shape[1])
    mask = build_chunk_mask(prefix_len, len(init_ids), prompt_embeds.dtype, prompt_embeds.device)
    base_pos = meta["position_ids"]
    if base_pos is None:
        base_pos = torch.arange(prefix_len, device=prompt_embeds.device).unsqueeze(0)
    pos_ids = torch.cat(
        [base_pos.to(prompt_embeds.device).view(1, -1),
         torch.arange(prefix_len, prefix_len + len(init_ids), device=prompt_embeds.device).unsqueeze(0)],
        dim=1,
    )
    embed = model.get_model().embed_tokens
    guess = torch.tensor([init_ids], device=prompt_embeds.device, dtype=torch.long)
    t_iters, iters, converged = 0.0, 0, False
    for _ in range(max_iter):
        def _step():
            emb = torch.cat([prompt_embeds, embed(guess).float()], dim=1)
            return model(inputs_embeds=emb, attention_mask=mask, position_ids=pos_ids, use_cache=False)

        t_it, out = sync_ms(_step)
        t_iters += t_it
        iters += 1
        logits = out.logits[0, prefix_len - 1 : prefix_len - 1 + len(init_ids)].float()
        new = logits.argmax(dim=-1, keepdim=True).view(1, -1)
        changed = int((new != guess).sum().item())
        guess = new
        if changed == 0:
            converged = True
            break
    text = tokenizer.decode([int(x) for x in guess[0].tolist()], skip_special_tokens=False)
    parse_ok = True
    try:
        od = parse_opendrive_traj(text, horizon=HORIZON)
    except ValueError:
        od = torch.zeros(HORIZON, 2)
        parse_ok = False
    return {"xy": od_to_ego_xy(od).numpy(), "t_e2e_ms": t_pre + t_iters,
            "iters": iters, "converged": converged, "parse_ok": parse_ok}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--keep-frac", type=float, default=0.25)
    p.add_argument("--max-iter", type=int, default=24)
    p.add_argument("--out", default=str(ROOT / "results/unified_accel_table.json"))
    args = p.parse_args()
    if not os.path.isabs(args.out):
        args.out = str(ROOT / args.out)

    disable_torch_init()
    device = torch.device("cuda:0")
    tokenizer, model, _, _ = load_pretrained_model(
        CKPT, model_base=None, model_name="llava_qwen", device_map=device, multimodal=True,
        attn_implementation="sdpa",
        overwrite_config={"image_aspect_ratio": "pad", "vision_tower_test_mode": True},
    )
    model.float().to(device).eval()

    blob = torch.load(HEAD, map_location="cpu", weights_only=False)
    head = DraftHead(int(blob["hidden_size"]), horizon=HORIZON, width=512).to(device)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    cfg = AcceptConfig(lon_m=0.5, lat_m=0.3)
    extra_map, speed_map, gt_map = extras_map()
    vision = model.get_vision_tower()

    uniad_cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
    data_args = DataArguments(data_path=None, lazy_preprocess=True, frames_upbound=32)
    dataset = LLaVANuScenesDataset(tokenizer, data_args, uniad_cfg.data.test,
                                   llava_test_mode=True, use_uniad_pth=False)
    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
    idxs = list(range(args.start, min(args.start + args.n + args.warmup, len(dataset))))
    loader = DataLoader(Subset(dataset, idxs), batch_size=1, collate_fn=collator, num_workers=0)

    init_ids = template_ids(tokenizer, SLOT)
    PRUNE = f"Pruning (oracle {int(args.keep_frac*100)}%)"
    METHODS = ["AR greedy", "Spec-VLA (exact)", "Spec-VLA (pose-relax)",
               "PD-VLA (parallel)", PRUNE, "TrajSpecDec (ours)"]
    rows = {m: [] for m in METHODS}

    def record(name, xy, t_ms, gt, has_gt, extra=None):
        l2 = l2_horizon(xy[:HORIZON], gt) if has_gt else {}
        rec = {
            "t_e2e_ms": round(float(t_ms), 2),
            "has_gt": bool(has_gt),
            "ade_gt": round(float(l2["ade"]), 4) if has_gt else None,
            "l2_1s": round(float(l2["l2_1s"]), 4) if has_gt else None,
            "l2_2s": round(float(l2["l2_2s"]), 4) if has_gt else None,
            "l2_3s": round(float(l2["l2_3s"]), 4) if has_gt else None,
        }
        if extra:
            rec.update(extra)
        rows[name].append(rec)

    def gen_with(uniad_pth, data):
        with torch.inference_mode():
            return model.generate(
                data["input_ids"], uniad_data=None, uniad_pth=uniad_pth,
                qa_instance_ind=data.get("qa_instance_ind"),
                do_sample=False, temperature=0, max_new_tokens=96, num_beams=1,
            )

    done = 0
    for i, data in enumerate(loader):
        sid = data["id"]
        if isinstance(sid, (list, tuple)):
            sid = sid[0]
        token = str(sid).replace("_trajectory", "")
        data = move_data_to_device(data, device)
        gt = gt_map.get(token, torch.zeros(HORIZON, 2)).numpy().astype(np.float32)
        has_gt = bool(np.linalg.norm(gt) > 0.2)
        extra = extra_map.get(token, torch.zeros(16)).float().to(device)
        if extra.numel() < 16:
            extra = torch.nn.functional.pad(extra.view(-1), (0, 16 - extra.numel()))
        speed = speed_map.get(token, torch.zeros(1)).float().to(device)

        # ---- ONE vision pass, shared by every method ----
        reset_uniad_temporal(model)
        t_vis, vt_res = sync_ms(lambda: vision(data["uniad_data"]))
        d_shared = dict(data)
        d_shared["uniad_pth"] = vt_res
        d_shared["uniad_data"] = None

        # ---- 0. AR greedy ----
        t_ar, cont = sync_ms(lambda: gen_with(vt_res, data))
        ar_text = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        ar_ego = od_to_ego_xy(torch.tensor(retrieve_traj(ar_text), dtype=torch.float32)).numpy()

        # ---- 1/2/5. SD: one draft + one verify, three accept rules ----
        reset_uniad_temporal(model)
        pack = spec_propose_opendrive(model, tokenizer, d_shared, head, extra, speed, horizon=HORIZON)
        sd_out = {}
        for name, mode in (("Spec-VLA (exact)", TOKEN_EXACT),
                           ("Spec-VLA (pose-relax)", SPECVLA_POSE),
                           ("TrajSpecDec (ours)", GEOMETRIC)):
            xy, info = spec_commit_opendrive(pack, cfg, accept_mode=mode, max_new=96)
            sd_out[name] = (np.asarray(xy, dtype=np.float32), info)

        # ---- 3. PD-VLA parallel ----
        reset_uniad_temporal(model)
        pd = pdvla_parallel(model, tokenizer, d_shared, init_ids, max_iter=args.max_iter)

        # ---- 4. pruning (oracle): slice projected features, tower cost unchanged ----
        # `uniad_pth` still routes through encode_vision_tower_result, so the slice
        # has to be applied *inside* that call rather than to the returned dict.
        kf = args.keep_frac
        orig_encode = model.encode_vision_tower_result

        def _sliced_encode(vt_result):
            s_f, t_f, m_f = orig_encode(vt_result)
            def _cut(feat):
                if feat is None:
                    return None
                k = max(1, int(round(feat.shape[0] * kf))) if kf > 0 else 0
                return feat[:k]
            return _cut(s_f), _cut(t_f), _cut(m_f)

        model.encode_vision_tower_result = _sliced_encode
        try:
            t_pr, cont_pr = sync_ms(lambda: gen_with(vt_res, data))
        finally:
            model.encode_vision_tower_result = orig_encode
        pr_text = tokenizer.batch_decode(cont_pr, skip_special_tokens=True)[0]
        pr_ego = od_to_ego_xy(torch.tensor(retrieve_traj(pr_text), dtype=torch.float32)).numpy()

        if i >= args.warmup:
            done += 1
            record("AR greedy", ar_ego, t_vis + t_ar, gt, has_gt)
            for name in ("Spec-VLA (exact)", "Spec-VLA (pose-relax)", "TrajSpecDec (ours)"):
                xy, info = sd_out[name]
                record(name, xy, t_vis + info["t_spec_e2e_ms"], gt, has_gt,
                       {"n_accept": info["n_accept"]})
            record("PD-VLA (parallel)", pd["xy"], t_vis + pd["t_e2e_ms"], gt, has_gt,
                   {"iters": pd["iters"], "converged": pd["converged"], "parse_ok": pd["parse_ok"]})
            record(PRUNE, pr_ego, t_vis + t_pr, gt, has_gt)
            print(
                f"[{done}/{args.n}] vis={t_vis:.0f} AR={t_ar:.0f} "
                f"ours={sd_out['TrajSpecDec (ours)'][1]['t_spec_e2e_ms']:.0f} "
                f"PD={pd['t_e2e_ms']:.0f} prune={t_pr:.0f} | "
                f"ADE AR={rows['AR greedy'][-1]['ade_gt']} "
                f"ours={rows['TrajSpecDec (ours)'][-1]['ade_gt']} "
                f"PD={rows['PD-VLA (parallel)'][-1]['ade_gt']}",
                flush=True,
            )
        if done >= args.n:
            break

    def agg(name):
        rs = rows[name]
        gt_rs = [r for r in rs if r["has_gt"]]
        t = np.array([r["t_e2e_ms"] for r in rs], dtype=np.float64)
        out = {"n": len(rs), "n_with_gt": len(gt_rs),
               "e2e_ms": round(float(t.mean()), 2),
               "p50_ms": round(float(np.median(t)), 2),
               "p90_ms": round(float(np.percentile(t, 90)), 2)}
        if gt_rs:
            for k in ("ade_gt", "l2_1s", "l2_2s", "l2_3s"):
                out[k] = round(float(np.mean([r[k] for r in gt_rs])), 4)
            out["ade_median"] = round(float(np.median([r["ade_gt"] for r in gt_rs])), 4)
        accs = [r["n_accept"] for r in rs if r.get("n_accept") is not None]
        if accs:
            out["mean_n_accept"] = round(float(np.mean(accs)), 3)
        if name == "PD-VLA (parallel)":
            out["converged_rate"] = round(float(np.mean([r["converged"] for r in rs])), 3)
            out["parse_ok_rate"] = round(float(np.mean([r["parse_ok"] for r in rs])), 3)
        return out

    summary = {
        "model": "OpenDriveVLA-0.5B+UniAD",
        "device": torch.cuda.get_device_name(0),
        "n": done, "start_index": args.start, "horizon": HORIZON,
        "accept": {"lon_m": cfg.lon_m, "lat_m": cfg.lat_m},
        "prune_keep_frac": args.keep_frac,
        "protocol": (
            "Vision tower runs ONCE per frame and its result is shared by all rows via "
            "uniad_pth, so every method consumes identical visual features; only the decode "
            "strategy differs. E2E includes the shared tower cost. UniAD temporal state is "
            "reset before each call. ADE / ST-P3 L2 in metres vs VAD GT, lower is better."
        ),
        "methods": {m: agg(m) for m in METHODS},
        "rows": rows,
    }
    base = summary["methods"]["AR greedy"]["e2e_ms"]
    for m in METHODS:
        summary["methods"][m]["speedup_e2e"] = round(base / max(summary["methods"][m]["e2e_ms"], 1e-6), 3)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
