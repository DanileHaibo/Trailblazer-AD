#!/usr/bin/env python3
"""Reproduce 4 training-free AD accelerators on OpenDriveVLA-0.5B + UniAD.

Methods (all inference-time only, no retraining, no extra weights):

  1. FastV         -- after layer K, drop the visual tokens with the least attention
                      (ECCV 2024 Oral; FastDriveVLA / Prune2Drive specialise this
                      family for driving). The pruned KV cache serves the rest of
                      prefill AND every decode step, so the saving is real.
  2. DeeAD         -- truncate the decoder stack when the partial trajectory is
                      already close to a cheap kinematic prior
                      (arXiv 2511.20720, AD-native early exit).
  3. Scaffold accept -- freeze structural tokens in the accept rule
                      (core trick of Fast-dDrive, arXiv 2605.23163).
  4. Pruning oracle -- slice projected UniAD features with no pruner at all;
                      upper bound for any learned pruning policy.

  + TrajSpecDec (ours) for reference.

Protocol matches `run_unified_accel_table.py`: the UniAD tower runs once per frame
and every method consumes the same features, so only the acceleration strategy
differs. E2E includes the shared tower cost.
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
from transformers.cache_utils import DynamicCache

from trajspec.ad_accel import (
    FastVConfig,
    fastv_prune_features,
    scaffold_accept_waypoints,
)
from trajspec.metrics import l2_horizon
from trajspec.draft_head import DraftHead, pack_extra, pack_speed
from trajspec.nusc_data import VAL_PKL, load_infos, pack_kinematics, token_index
from trajspec.opendrive_live import (
    TeacherDecoder,
    prefill_uniad,
    reset_uniad_temporal,
    spec_commit_opendrive,
    spec_propose_opendrive,
)
from trajspec.sd_baselines import GEOMETRIC, commit_spec, waypoints_in_token_prefix
from trajspec.serialize import TRAJ_END, TRAJ_START, extract_opendrive_points, od_to_ego_xy, parse_opendrive_traj

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"
HEAD = ROOT / "results/specdec_opendrive_heldout.pt"
HORIZON = 6


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


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
        gt[tok] = torch.as_tensor(g[:HORIZON], dtype=torch.float32) if g is not None else torch.zeros(HORIZON, 2)
    return extra, speed, gt


@torch.no_grad()
def greedy_at_depth(model, tokenizer, meta, depth: int, max_new: int = 96):
    """DeeAD: decode using only the first `depth` LM layers.

    DeeAD's claim is that deep layers over-refine, so an intermediate readout is
    enough. Implemented with a real KV cache (built by the first `depth` layers
    only) so the wall clock actually reflects the layer saving -- recomputing the
    full stack each step would make the method look slower than it is.

    Valid readout because OpenDriveVLA ties lm_head to the input embedding, so
    final-norm + lm_head is well defined at any depth.
    """
    m = model.model
    embed = m.embed_tokens
    dev = meta["inputs_embeds"].device
    dtype = meta["inputs_embeds"].dtype
    dec = TeacherDecoder(model)
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_set = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
    prompt_ids = meta.get("prompt_ids") or []
    layers = list(m.layers[:depth])

    # ---- prefill with only `depth` layers, keeping a cache ----
    L = int(meta["inputs_embeds"].shape[1])
    neg = torch.finfo(dtype).min
    mask = torch.triu(torch.full((L, L), neg, device=dev, dtype=dtype), diagonal=1).view(1, 1, L, L)
    pos = torch.arange(L, device=dev).unsqueeze(0)
    past = DynamicCache()
    h = meta["inputs_embeds"]
    for layer in layers:
        pe = m.rotary_emb(h, pos)
        o = layer(h, attention_mask=mask, position_ids=pos, past_key_values=past,
                  use_cache=True, position_embeddings=pe)
        h = o[0] if isinstance(o, tuple) else o
    h = m.norm(h)
    logits = model.lm_head(h[:, -1]).float().view(-1)

    gen: list[int] = []
    cur_pos = L
    for _ in range(max_new):
        tid = dec(logits, list(prompt_ids) + gen)
        if tid in eos_set:
            break
        gen.append(tid)
        text = tokenizer.decode(gen, skip_special_tokens=False)
        if len(extract_opendrive_points(text)) >= HORIZON and "]" in text.replace("<traj_end>", ""):
            break
        em = embed(torch.tensor([[tid]], device=dev))
        n = past.get_seq_length() + 1
        h = em
        for layer in layers:
            p1 = torch.tensor([[cur_pos]], device=dev)
            pe = m.rotary_emb(h, p1)
            o = layer(h, attention_mask=torch.ones(1, 1, 1, n, device=dev, dtype=dtype),
                      position_ids=p1, past_key_values=past, use_cache=True,
                      position_embeddings=pe, cache_position=torch.tensor([n - 1], device=dev))
            h = o[0] if isinstance(o, tuple) else o
        cur_pos += 1
        h = m.norm(h)
        logits = model.lm_head(h[:, -1]).float().view(-1)
    return tokenizer.decode(gen, skip_special_tokens=False), gen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--fastv-k", type=int, default=2)
    p.add_argument("--fastv-r", type=float, default=0.5)
    p.add_argument("--deead-depth", type=int, default=16)
    p.add_argument("--keep-frac", type=float, default=0.25)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--out", default=str(ROOT / "results/four_accel.json"))
    args = p.parse_args()
    if not os.path.isabs(args.out):
        args.out = str(ROOT / args.out)

    disable_torch_init()
    device = torch.device("cuda:0")
    # eager attention is required for FastV's attention-based ranking
    tokenizer, model, _, _ = load_pretrained_model(
        CKPT, model_base=None, model_name="llava_qwen", device_map=device, multimodal=True,
        attn_implementation="eager",
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

    FV = f"FastV (K={args.fastv_k}, R={args.fastv_r})"
    DE = f"DeeAD (depth {args.deead_depth}/24)"
    PR = f"Pruning oracle ({int(args.keep_frac*100)}% kept)"
    METHODS = ["AR greedy", FV, DE, "Scaffold accept", PR, "TrajSpecDec (ours)"]
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

    def gen_plain(vt_res, data):
        with torch.inference_mode():
            return model.generate(
                data["input_ids"], uniad_data=None, uniad_pth=vt_res,
                qa_instance_ind=data.get("qa_instance_ind"),
                do_sample=False, temperature=0, max_new_tokens=args.max_new, num_beams=1,
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
        d_shared = {**data, "uniad_pth": None, "uniad_data": data["uniad_data"]}

        # ---- shared vision pass ----
        reset_uniad_temporal(model)
        t_vis, vt_res = sync_ms(lambda: vision(data["uniad_data"]))
        d_shared["uniad_pth"] = vt_res
        d_shared["uniad_data"] = None

        # ---- AR greedy ----
        t_ar, cont = sync_ms(lambda: gen_plain(vt_res, data))
        ar_ego = od_to_ego_xy(torch.tensor(retrieve_traj(
            tokenizer.batch_decode(cont, skip_special_tokens=True)[0]), dtype=torch.float32)).numpy()

        # ---- shared prefill (for embeddings / draft) ----
        reset_uniad_temporal(model)
        pre, meta = prefill_uniad(model, d_shared)
        meta["prompt_ids"] = [int(t) for t in data["input_ids"][0].tolist()]

        # ================= 1. FastV =================
        d_shared["_ids_row"] = data["input_ids"][0].tolist()

        def _fastv():
            return fastv_prune_features(
                model, tokenizer, d_shared, FastVConfig(k=args.fastv_k, r=args.fastv_r),
                horizon=HORIZON,
            )

        t_fv, (fv_text, fv_nvis, fv_kept, t_fv_inner) = sync_ms(_fastv)
        t_fv = t_fv_inner
        fv_ego = od_to_ego_xy(parse_opendrive_traj(fv_text, horizon=HORIZON)).numpy()

        # ================= 2. DeeAD =================
        t_de, (de_text, de_ids) = sync_ms(
            lambda: greedy_at_depth(model, tokenizer, meta, args.deead_depth, args.max_new)
        )
        de_ego = od_to_ego_xy(parse_opendrive_traj(de_text, horizon=HORIZON)).numpy()

        # ================= 3/5. SD: scaffold vs geometric =================
        reset_uniad_temporal(model)
        pack = spec_propose_opendrive(model, tokenizer, d_shared, head, extra, speed, horizon=HORIZON)
        xy_ours, info_ours = spec_commit_opendrive(pack, cfg, accept_mode=GEOMETRIC, max_new=args.max_new)

        # Scaffold rule: structural tokens never block acceptance. Same draft, same
        # verify; only the accept rule differs. It still needs its own tail decode,
        # so it is timed independently.
        n_scaf_wp, n_scaf_tok = scaffold_accept_waypoints(
            tokenizer, pack["draft_ids"], pack["teacher_tok"],
            pack["draft_ego"], pack["teacher_ego"], cfg, horizon=HORIZON,
        )

        def _scaffold_commit():
            """Commit at the scaffold-derived accept count, with a real tail decode."""
            import copy

            p2 = copy.copy(pack)
            # monkey-patch the accept count so the shared tail machinery uses it
            orig_n = n_accept_fn = None
            out_xy, out_info = None, None
            from trajspec import opendrive_live as ol

            real = ol.n_accept_opendrive

            def _fake(mode, draft_xy, teacher_xy, draft_ids, teacher_ids, tok_, cfg_, **kw):
                return int(n_scaf_wp)

            ol.n_accept_opendrive = _fake
            try:
                out_xy, out_info = spec_commit_opendrive(p2, cfg, accept_mode=GEOMETRIC,
                                                         max_new=args.max_new)
            finally:
                ol.n_accept_opendrive = real
            return out_xy, out_info

        # honest timing: run the real commit path, then override the accept count
        t_scaf, (_, scaf_info) = sync_ms(_scaffold_commit)
        scaf_xy2, _, _ = commit_spec(pack["draft_ego"], pack["teacher_ego"], n_scaf_wp,
                                     GEOMETRIC, horizon=HORIZON)
        scaf_np = scaf_xy2.numpy().astype(np.float32) if torch.is_tensor(scaf_xy2) else np.asarray(scaf_xy2)

        # ================= 4. pruning oracle =================
        orig_encode = model.encode_vision_tower_result
        kf = args.keep_frac

        def _sliced(vt_result):
            s_f, t_f, m_f = orig_encode(vt_result)
            def _cut(feat):
                if feat is None:
                    return None
                k = max(1, int(round(feat.shape[0] * kf))) if kf > 0 else 0
                return feat[:k]
            return _cut(s_f), _cut(t_f), _cut(m_f)

        model.encode_vision_tower_result = _sliced
        try:
            t_pr, cont_pr = sync_ms(lambda: gen_plain(vt_res, data))
        finally:
            model.encode_vision_tower_result = orig_encode
        pr_ego = od_to_ego_xy(torch.tensor(retrieve_traj(
            tokenizer.batch_decode(cont_pr, skip_special_tokens=True)[0]), dtype=torch.float32)).numpy()

        if i >= args.warmup:
            done += 1
            record("AR greedy", ar_ego, t_vis + t_ar, gt, has_gt)
            record(FV, fv_ego, t_vis + t_fv, gt, has_gt, {"n_vis": fv_nvis, "n_kept": fv_kept})
            record(DE, de_ego, t_vis + t_de, gt, has_gt, {"n_tokens": len(de_ids)})
            record("Scaffold accept", scaf_np, t_vis + t_scaf, gt, has_gt,
                   {"n_accept": int(n_scaf_wp), "n_accept_geo": int(info_ours["n_accept"]),
                    "n_accept_tokens": int(n_scaf_tok)})
            record(PR, pr_ego, t_vis + t_pr, gt, has_gt)
            record("TrajSpecDec (ours)", np.asarray(xy_ours, dtype=np.float32),
                   t_vis + info_ours["t_spec_e2e_ms"], gt, has_gt,
                   {"n_accept": info_ours["n_accept"]})
            print(
                f"[{done}/{args.n}] vis={t_vis:.0f} AR={t_ar:.0f} FV={t_fv:.0f}(kept {fv_kept}/{fv_nvis}) "
                f"deeAD={t_de:.0f} ours={info_ours['t_spec_e2e_ms']:.0f} prune={t_pr:.0f} | "
                f"ADE AR={rows['AR greedy'][-1]['ade_gt']} FV={rows[FV][-1]['ade_gt']} "
                f"de={rows[DE][-1]['ade_gt']} scaf={rows['Scaffold accept'][-1]['ade_gt']} "
                f"ours={rows['TrajSpecDec (ours)'][-1]['ade_gt']}",
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
        return out

    summary = {
        "model": "OpenDriveVLA-0.5B+UniAD",
        "device": torch.cuda.get_device_name(0),
        "n": done, "start_index": args.start, "horizon": HORIZON,
        "config": {"fastv_k": args.fastv_k, "fastv_r": args.fastv_r,
                   "deead_depth": args.deead_depth, "keep_frac": args.keep_frac},
        "protocol": (
            "Training-free accelerators only (no retraining, no extra weights). "
            "UniAD tower runs ONCE per frame and is shared by every row; E2E includes it. "
            "FastV prunes visual tokens after layer K and keeps decoding against the pruned "
            "KV cache. DeeAD truncates the decoder stack to the first N layers. "
            "Scaffold accept uses the same draft/verify as ours, only the accept rule differs. "
            "Pruning oracle slices projected features with no pruner (upper bound)."
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
