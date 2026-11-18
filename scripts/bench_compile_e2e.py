#!/usr/bin/env python3
"""E2E cost of the system-level speedup (bf16 + torch.compile) on real frames.

Context
-------
`scripts/bench_decode_micro.py` shows the bare decoder stack goes from
16.68 ms/token (eager fp32) to 1.68 ms/token (compiled bf16) -- 9.9x -- because a
1-token forward costs as much as a 620-token prefill, i.e. launch overhead dominates.

This script measures whether that holds *end to end* on the real pipeline.

Design decisions, and why
------------------------
* **No prompt padding.** The obvious trick (pad to a fixed L_PAD so shapes never
  change) is wrong for a StaticCache here: prompts vary 605-640 tokens, so the
  decode step must write at `n_prompt + k`, which differs per frame. Padding fixes
  the prefill shape but leaves the cache offset frame-dependent, and mixing the two
  silently corrupted the trajectory (repeated `0.0.0.00` tokens) during
  development. We therefore let the prompt keep its natural length.
* **Compile with `dynamic=True`.** Sequence length grows by 1 each step, so dynamic
  shapes avoid one re-trace per step. `dynamic=False` re-traces ~64 times per frame
  and is unusably slow (>10 min/frame in our earlier attempt).
* **Clone logits.** `reduce-overhead` reuses output buffers across replays; keeping a
  reference across loop iterations triggers
  "accessing tensor output of CUDAGraphs that has been overwritten".
* **Tower stays fp32** (its mmcv deformable-conv has no bf16 kernel) and is charged
  to every row, since it runs once per frame regardless of decode settings.

Reproduction note: run the fp32 and bf16 passes in this single process, sequentially.
Interleaving `.to(dtype)` calls inside one loop invalidates compiled graphs.
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
from trajspec.opendrive_live import TeacherDecoder, reset_uniad_temporal
from trajspec.serialize import extract_opendrive_points, od_to_ego_xy

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"
HORIZON = 6
MAX_NEW = 96


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


class LMHeadWrap(torch.nn.Module):
    """Decoder stack + lm_head, so the head is inside the compiled graph."""

    def __init__(self, inner, head):
        super().__init__()
        self.inner = inner
        self.head = head

    def forward(self, inputs_embeds, attention_mask, position_ids,
                past_key_values, cache_position):
        o = self.inner(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=True, cache_position=cache_position, output_hidden_states=False,
        )
        # return the cache so the caller can keep decoding; logits are NOT cloned here
        # because reduce-overhead reuses buffers -- callers clone what they keep.
        return self.head(o.last_hidden_state), o.past_key_values

    @property
    def embed_tokens(self):
        return self.inner.embed_tokens


@torch.inference_mode()
def greedy_with_cache(lm, embed, tokenizer, embeds, attn, pos, prompt_ids,
                      dtype, max_new: int = MAX_NEW):
    """Greedy decode reusing past_key_values (the real path).

    The attention mask MUST grow with the cache. Omitting it makes the model attend
    through the padded/unwritten tail and the output degenerates to a repeated token
    (observed while developing this script: 151667 x64), which is easy to mistake for
    a precision problem.
    """
    n_prompt = int(embeds.shape[1])
    dev = embeds.device
    # normalise to a 1-D additive-free bool/long mask of length n_prompt
    if attn is None:
        mask = torch.ones(1, n_prompt, dtype=torch.long, device=dev)
    else:
        mask = attn.view(1, -1).to(torch.long)

    o = lm(
        inputs_embeds=embeds.to(dtype), attention_mask=mask, position_ids=pos,
        past_key_values=None, cache_position=torch.arange(n_prompt, device=dev),
    )
    lg0, past = o
    logits = lg0[0, -1].float().clone()
    dec = TeacherDecoder(lm)
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_set = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
    gen: list[int] = []
    for k in range(max_new):
        tid = dec(logits, prompt_ids + gen)
        if tid in eos_set:
            break
        gen.append(tid)
        text = tokenizer.decode(gen, skip_special_tokens=False)
        if len(extract_opendrive_points(text)) >= HORIZON and "]" in text.replace("<traj_end>", ""):
            break
        n = n_prompt + k
        step_mask = torch.ones(1, n + 1, dtype=torch.long, device=dev)
        lg2, past = lm(
            inputs_embeds=embed(torch.tensor([[tid]], device=dev)).to(dtype),
            attention_mask=step_mask,
            position_ids=torch.tensor([[n]], device=dev),
            past_key_values=past, cache_position=torch.tensor([n], device=dev),
        )
        logits = lg2[0, 0].float().clone()
    return tokenizer.decode(gen, skip_special_tokens=False), gen


def build(dtype: torch.dtype, compile_mode: str):
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
        m.layers.to(dtype)
        m.norm.to(dtype)
        m.embed_tokens.to(dtype)
        model.lm_head.to(dtype)
    wrapped = LMHeadWrap(model.model, model.lm_head)
    lm = wrapped if compile_mode == "none" else torch.compile(
        wrapped, mode=compile_mode, fullgraph=False, dynamic=True)
    return tokenizer, model, device, lm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--start", type=int, default=1500)
    p.add_argument("--out", default=str(ROOT / "results/compile_e2e.json"))
    args = p.parse_args()
    if not os.path.isabs(args.out):
        args.out = str(ROOT / args.out)

    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    gt_map = {}
    for tok, i in tok2i.items():
        s = pack_kinematics(infos[i])
        if s is None:
            continue
        g = s.get("gt")
        gt_map[tok] = torch.as_tensor(g[:HORIZON], dtype=torch.float32) if g is not None else torch.zeros(HORIZON, 2)

    # ---- collect prompts + fp32 reference on one model instance ----
    tokenizer, model, device, _lm_unused = build(torch.float32, "none")
    uniad_cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
    data_args = DataArguments(data_path=None, lazy_preprocess=True, frames_upbound=32)
    dataset = LLaVANuScenesDataset(tokenizer, data_args, uniad_cfg.data.test,
                                   llava_test_mode=True, use_uniad_pth=False)
    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
    idxs = list(range(args.start, min(args.start + args.n + args.warmup, len(dataset))))
    loader = DataLoader(Subset(dataset, idxs), batch_size=1, collate_fn=collator, num_workers=0)
    vision = model.get_vision_tower()

    frames = []
    for i, data in enumerate(loader):
        sid = data["id"]
        if isinstance(sid, (list, tuple)):
            sid = sid[0]
        token = str(sid).replace("_trajectory", "")
        data = move_data_to_device(data, device)
        gt = gt_map.get(token, torch.zeros(HORIZON, 2)).numpy().astype(np.float32)
        reset_uniad_temporal(model)
        t_vis, vt_res = sync_ms(lambda: vision(data["uniad_data"]))
        reset_uniad_temporal(model)
        with torch.inference_mode():
            _, pos_ids, attn, _, embeds, _ = model.prepare_inputs_labels_for_multimodal_uniad_vlm(
                data["input_ids"], None, None, None, None, None,
                uniad_pth=vt_res, qa_instance_ind=data.get("qa_instance_ind"))
        reset_uniad_temporal(model)
        t_ref, cont = sync_ms(lambda: model.generate(
            data["input_ids"], uniad_data=None, uniad_pth=vt_res,
            qa_instance_ind=data.get("qa_instance_ind"),
            do_sample=False, temperature=0, max_new_tokens=MAX_NEW, num_beams=1))
        ref_text = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        ref_ego = od_to_ego_xy(torch.tensor(retrieve_traj(ref_text), dtype=torch.float32)).numpy()
        frames.append({
            "token": token, "gt": gt, "has_gt": bool(np.linalg.norm(gt) > 0.2),
            "t_vis": t_vis, "t_ref": t_ref,
            "embeds": embeds.detach().clone(),
            "pos": pos_ids.detach().clone() if pos_ids is not None else None,
            "attn": attn.detach().clone() if attn is not None else None,
            "n_prompt": int(embeds.shape[1]),
            "prompt_ids": [int(t) for t in data["input_ids"][0].tolist()],
            "ref_text": ref_text, "ref_ego": ref_ego,
        })
        print(f"[collect {len(frames)}] n_prompt={frames[-1]['n_prompt']} ref={t_ref:.0f}ms", flush=True)
        if len(frames) >= args.n + args.warmup:
            break
    del model, vision
    torch.cuda.empty_cache()

    results = {}
    for tag, dtype, mode in (("fp32_eager", torch.float32, "none"),
                             ("bf16_eager", torch.bfloat16, "none"),
                             ("bf16_compiled", torch.bfloat16, "default")):
        print(f"\n===== {tag} =====", flush=True)
        tk, model, dev, lm = build(dtype, mode)
        embed = model.model.embed_tokens
        rows = []
        for i, fr in enumerate(frames):
            emb = fr["embeds"].to(dev)
            att = fr["attn"].to(dev) if fr["attn"] is not None else torch.ones(
                1, fr["n_prompt"], dtype=torch.long, device=dev)
            pos = fr["pos"].to(dev) if fr["pos"] is not None else torch.arange(
                fr["n_prompt"], device=dev).unsqueeze(0)
            t, (text, gen) = sync_ms(lambda: greedy_with_cache(
                lm, embed, tk, emb, att, pos, fr["prompt_ids"], dtype))
            try:
                ego = od_to_ego_xy(torch.tensor(retrieve_traj(text), dtype=torch.float32)).numpy()
            except Exception:  # noqa: BLE001
                ego = np.zeros((HORIZON, 2), dtype=np.float32)
            ade_ref = float(np.linalg.norm(ego[:HORIZON] - fr["ref_ego"][:HORIZON], axis=-1).mean())
            l2 = l2_horizon(ego[:HORIZON], fr["gt"]) if fr["has_gt"] else {}
            rec = {"token": fr["token"], "t_ms": round(t, 2), "n_tokens": len(gen),
                   "ade_vs_fp32": round(ade_ref, 4),
                   "ade_gt": round(float(l2["ade"]), 4) if fr["has_gt"] else None,
                   "identical_tokens": text == fr["ref_text"]}
            if i >= args.warmup:
                rows.append(rec)
            print(f"  [{i}] {t:8.1f}ms tok={len(gen):3d} ADE_vs_fp32={ade_ref:.4f} "
                  f"same={text == fr['ref_text']}", flush=True)
            if len(rows) >= args.n:
                break
        del model, lm, embed
        torch.cuda.empty_cache()
        warm = frames[args.warmup:]
        results[tag] = {
            "n": len(rows),
            "t_decode_ms": round(float(np.mean([r["t_ms"] for r in rows])), 2),
            "t_e2e_ms": round(float(np.mean([r["t_ms"] + fr["t_vis"] for r, fr in zip(rows, warm)])), 2),
            "n_tokens": round(float(np.mean([r["n_tokens"] for r in rows])), 1),
            "ade_vs_fp32": round(float(np.mean([r["ade_vs_fp32"] for r in rows])), 4),
            "identical_rate": round(float(np.mean([r["identical_tokens"] for r in rows])), 3),
            "rows": rows,
        }
        ref_e2e = float(np.mean([fr["t_ref"] + fr["t_vis"] for fr in warm]))
        results[tag]["speedup_e2e_vs_official_generate"] = round(
            ref_e2e / max(results[tag]["t_e2e_ms"], 1e-6), 3)

    summary = {"model": "OpenDriveVLA-0.5B+UniAD", "device": torch.cuda.get_device_name(0),
               "n": args.n, "start_index": args.start, "max_new": MAX_NEW,
               "note": ("Natural prompt length (no padding). Tower fp32, charged to every row. "
                        "bf16 is not bit-exact for character-level numeric output, so we report "
                        "ADE against the fp32 answer rather than claiming identical output."),
               "results": results}
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n" + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                             for k, v in results.items()}, indent=2))
    print("[write]", args.out)


if __name__ == "__main__":
    main()
