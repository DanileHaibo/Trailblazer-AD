#!/usr/bin/env python3
"""The speedup that actually exists here: system-level, not algorithmic.

Microbenchmark of the bare Qwen2-0.5B decoder on this L20 (620-token prompt,
`scripts/bench_decode_micro.py`):

    eager fp32        1 token 16.68 ms    prefill 26.41 ms
    bf16 decoder      1 token 14.76 ms    prefill 16.15 ms
    compiled bf16     1 token  1.68 ms    prefill  8.94 ms   <-- 9.9x on decode

A 1-token forward costing as much as a 620-token prefill is launch overhead, not
arithmetic: 24 layers x ~6 kernels per step at ~4 us each, plus Python. That is why
the published 4-12x driving-VLA papers are all algorithm/stack co-designs, not pure
algorithm work.

This script quantifies both gains separately and together, on identical frames:

  A. eager fp32 (our previous baseline)
  B. bf16 decoder stack + torch.compile("default")
  C. TrajSpecDec on top of B (fewer decode steps AND cheaper ones)

Implementation notes
--------------------
* UniAD's mmcv deformable-conv has no bf16 kernel, so the tower stays fp32. Only the
  decoder stack (layers / norm / embed_tokens / lm_head) is cast.
* The two dtype configurations run in **separate passes** over the same frames. Mixing
  them in one loop forced repeated `.to()` calls, which invalidated compiled graphs
  and produced dtype-mismatch errors -- a real footgun worth avoiding.
* "compile" here means a fixed prompt length per frame is *not* required: we compile
  with mode="default" and let guards re-trace on shape changes. mode="reduce-overhead"
  is faster in isolation but its CUDA-graph buffer reuse breaks an autoregressive loop.
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
from trajspec.draft_head import DraftHead, pack_extra, pack_speed
from trajspec.nusc_data import VAL_PKL, load_infos, pack_kinematics, token_index
from trajspec.opendrive_live import (
    TeacherDecoder,
    prefill_uniad,
    reset_uniad_temporal,
    spec_commit_opendrive,
    spec_propose_opendrive,
)
from trajspec.serialize import extract_opendrive_points, od_to_ego_xy

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"
HEAD = ROOT / "results/specdec_opendrive_heldout.pt"
HORIZON = 6


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


class _LMHead(torch.nn.Module):
    """Decoder stack + lm_head as one callable, so the head is inside the graph."""

    def __init__(self, inner, head):
        super().__init__()
        self.inner = inner
        self.head = head

    def forward(self, inputs_embeds, attention_mask, position_ids,
                past_key_values=None, use_cache=True):
        o = self.inner(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=use_cache, output_hidden_states=False,
        )
        logits = self.head(o.last_hidden_state)
        return logits, o.past_key_values


def build(dtype: torch.dtype, compile_lm: bool):
    disable_torch_init()
    device = torch.device("cuda:0")
    tokenizer, model, _, _ = load_pretrained_model(
        CKPT, model_base=None, model_name="llava_qwen", device_map=device, multimodal=True,
        attn_implementation="sdpa",
        overwrite_config={"image_aspect_ratio": "pad", "vision_tower_test_mode": True},
    )
    model.float().to(device).eval()
    if dtype != torch.float32:
        m = model.model
        # Only the decoder stack moves. The mm_projector_* layers must stay fp32
        # because they consume fp32 UniAD features; their output is cast to `dtype`
        # before entering the LM.
        m.layers.to(dtype)
        m.norm.to(dtype)
        m.embed_tokens.to(dtype)
        model.lm_head.to(dtype)
    wrapped = _LMHead(model.model, model.lm_head)
    lm = wrapped if not compile_lm else torch.compile(wrapped, mode="default", fullgraph=False)
    return tokenizer, model, device, lm


@torch.inference_mode()
def greedy(lm, embed, tokenizer, embeds, attn, pos, prompt_ids, max_new=96):
    logits_all, past = lm(inputs_embeds=embeds, attention_mask=attn, position_ids=pos)
    cache_len = int(embeds.shape[1])
    dec = TeacherDecoder(lm)
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_set = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
    logits = logits_all[0, -1].float()
    gen: list[int] = []
    for _ in range(max_new):
        tid = dec(logits, prompt_ids + gen)
        if tid in eos_set:
            break
        gen.append(tid)
        text = tokenizer.decode(gen, skip_special_tokens=False)
        if len(extract_opendrive_points(text)) >= HORIZON and "]" in text.replace("<traj_end>", ""):
            break
        n = cache_len + len(gen)
        em = embed(torch.tensor([[tid]], device=embeds.device))
        lg, past = lm(inputs_embeds=em,
                      attention_mask=torch.ones(1, n, device=embeds.device, dtype=torch.long),
                      position_ids=torch.arange(n - 1, n, device=embeds.device).unsqueeze(0),
                      past_key_values=past)
        logits = lg[0, -1].float()
    return tokenizer.decode(gen, skip_special_tokens=False), gen


def run_pass(tag: str, dtype: torch.dtype, compile_lm: bool, args, maps):
    gt_map, extra_map, speed_map = maps
    print(f"\n===== pass {tag}: dtype={dtype} compile={compile_lm} =====", flush=True)
    tokenizer, model, device, lm = build(dtype, compile_lm)
    m = model.model
    cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
    data_args = DataArguments(data_path=None, lazy_preprocess=True, frames_upbound=32)
    dataset = LLaVANuScenesDataset(tokenizer, data_args, cfg.data.test,
                                   llava_test_mode=True, use_uniad_pth=False)
    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
    idxs = list(range(args.start, min(args.start + args.n + args.warmup, len(dataset))))
    loader = DataLoader(Subset(dataset, idxs), batch_size=1, collate_fn=collator, num_workers=0)
    vision = model.get_vision_tower()

    blob = torch.load(HEAD, map_location="cpu", weights_only=False)
    head = DraftHead(int(blob["hidden_size"]), horizon=HORIZON, width=512).to(device)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    accept = AcceptConfig(lon_m=0.5, lat_m=0.3)

    rows = []
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

        reset_uniad_temporal(model)
        t_vis, vt_res = sync_ms(lambda: vision(data["uniad_data"]))

        # prompt built by the fp32 tower, cast for the LM
        reset_uniad_temporal(model)
        _, pos_ids, attn, _, embeds, _ = model.prepare_inputs_labels_for_multimodal_uniad_vlm(
            data["input_ids"], None, None, None, None, None,
            uniad_pth=vt_res, qa_instance_ind=data.get("qa_instance_ind"))
        embeds = embeds.to(dtype)
        # attention mask / position ids stay integer; nothing else to cast
        prompt_ids = [int(t) for t in data["input_ids"][0].tolist()]

        t_ar, (ar_text, ar_gen) = sync_ms(
            lambda: greedy(lm, m.embed_tokens, tokenizer, embeds, attn, pos_ids,
                           prompt_ids, args.max_new))

        # TrajSpecDec on the same (possibly accelerated) stack
        reset_uniad_temporal(model)
        d_shared = {**data, "uniad_pth": vt_res, "uniad_data": None}
        pack = spec_propose_opendrive(model, tokenizer, d_shared, head, extra, speed, horizon=HORIZON)
        xy_sd, info = spec_commit_opendrive(pack, accept, accept_mode="geometric", max_new=args.max_new)

        ar_ego = od_to_ego_xy(torch.tensor(retrieve_traj(ar_text), dtype=torch.float32)).numpy()
        rec = {
            "token": token, "has_gt": has_gt,
            "t_vis_ms": round(t_vis, 2),
            "t_ar_ms": round(t_ar, 2),
            "t_ar_e2e_ms": round(t_vis + t_ar, 2),
            "n_tokens": len(ar_gen),
            "ade_ar": round(float(l2_horizon(ar_ego[:HORIZON], gt)["ade"]), 4) if has_gt else None,
            "t_spec_ms": round(info["t_spec_e2e_ms"], 2),
            "t_spec_e2e_ms": round(t_vis + info["t_spec_e2e_ms"], 2),
            "n_accept": int(info["n_accept"]),
            "ade_spec": round(float(l2_horizon(np.asarray(xy_sd)[:HORIZON], gt)["ade"]), 4) if has_gt else None,
            "ar_text_head": ar_text[:60],
        }
        if i >= args.warmup:
            rows.append(rec)
        print(f"  [{i}] vis={t_vis:6.1f} AR={t_ar:8.1f} tok={len(ar_gen):3d} "
              f"spec={info['t_spec_e2e_ms']:7.1f} acc={info['n_accept']}", flush=True)
        if len(rows) >= args.n:
            break

    del model, vision, lm
    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--out", default=str(ROOT / "results/system_speedup.json"))
    args = p.parse_args()
    if not os.path.isabs(args.out):
        args.out = str(ROOT / args.out)

    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    gt_map, extra_map, speed_map = {}, {}, {}
    for tok, i in tok2i.items():
        s = pack_kinematics(infos[i])
        if s is None:
            continue
        extra_map[tok] = pack_extra(s)
        speed_map[tok] = pack_speed(s)
        g = s.get("gt")
        gt_map[tok] = torch.as_tensor(g[:HORIZON], dtype=torch.float32) if g is not None else torch.zeros(HORIZON, 2)
    maps = (gt_map, extra_map, speed_map)

    runs = {
        "A_eager_fp32": run_pass("A", torch.float32, False, args, maps),
        "B_compiled_bf16": run_pass("B", torch.bfloat16, True, args, maps),
    }

    summary = {"model": "OpenDriveVLA-0.5B+UniAD", "n": args.n, "start_index": args.start,
               "device": torch.cuda.get_device_name(0), "runs": {}, "headline": {}}
    base = {r["token"]: r for r in runs["A_eager_fp32"]}
    for tag, rows in runs.items():
        gt_rows = [r for r in rows if r["has_gt"]]
        same = [r["token"] for r in rows
                if r["token"] in base and r["ar_text_head"] == base[r["token"]]["ar_text_head"]]
        summary["runs"][tag] = {
            "n": len(rows),
            "t_ar_e2e_ms": round(float(np.mean([r["t_ar_e2e_ms"] for r in rows])), 2),
            "t_spec_e2e_ms": round(float(np.mean([r["t_spec_e2e_ms"] for r in rows])), 2),
            "t_ar_ms": round(float(np.mean([r["t_ar_ms"] for r in rows])), 2),
            "n_tokens": round(float(np.mean([r["n_tokens"] for r in rows])), 1),
            "mean_n_accept": round(float(np.mean([r["n_accept"] for r in rows])), 3),
            "ade_ar": round(float(np.mean([r["ade_ar"] for r in gt_rows])), 4) if gt_rows else None,
            "ade_spec": round(float(np.mean([r["ade_spec"] for r in gt_rows])), 4) if gt_rows else None,
            "identical_answer_rate_vs_A": round(len(same) / max(len(rows), 1), 3),
            "rows": rows,
        }
    a = summary["runs"]["A_eager_fp32"]
    b = summary["runs"]["B_compiled_bf16"]
    summary["headline"] = {
        "ar_speedup_A_to_B": round(a["t_ar_e2e_ms"] / max(b["t_ar_e2e_ms"], 1e-6), 3),
        "spec_speedup_A_to_B": round(a["t_spec_e2e_ms"] / max(b["t_spec_e2e_ms"], 1e-6), 3),
        "spec_vs_ar_within_A": round(a["t_ar_e2e_ms"] / max(a["t_spec_e2e_ms"], 1e-6), 3),
        "spec_vs_ar_within_B": round(b["t_ar_e2e_ms"] / max(b["t_spec_e2e_ms"], 1e-6), 3),
        "combined_A_ar_to_B_spec": round(a["t_ar_e2e_ms"] / max(b["t_spec_e2e_ms"], 1e-6), 3),
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                      for k, v in summary["runs"].items()}, indent=2))
    print("[write]", args.out)


if __name__ == "__main__":
    main()
