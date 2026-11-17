#!/usr/bin/env python3
"""Smoke-test Qwen3-VL KV continue vs official generate (1 moving val frame)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.adapters.rnb_waypoints import RnbWaypointsRuntime, parse_planned_traj, od_to_ego
from trajspec.nusc_data import VAL_PKL, load_infos, pack_sample_rnb, token_index


def main():
    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    sample = None
    for i in range(len(infos)):
        s = pack_sample_rnb(infos, i, tok2i)
        if s is None:
            continue
        gt = np.asarray(s["gt"][:6], dtype=np.float32)
        if float(np.linalg.norm(gt[-1])) > 5.0:
            sample = s
            break
    assert sample is not None
    runner = RnbWaypointsRuntime(device="cuda")
    tokzr = runner.processor.tokenizer
    inp = runner._inputs(sample)
    prompt_len = int(inp["input_ids"].shape[1])

    gen = runner.model.generate(**inp, max_new_tokens=96, do_sample=False, use_cache=True)
    gen_text = tokzr.decode(gen[0, prompt_len:], skip_special_tokens=True)
    print("GENERATE", repr(gen_text[:200]), flush=True)

    pre = runner.model(**inp, use_cache=True, output_hidden_states=True)
    past = pre.past_key_values
    logits = pre.logits[0, -1].float()
    print("prefill cache_len", runner._cache_seq_len(past), "prompt", prompt_len, "rope", getattr(runner.model.model, "rope_deltas", None), flush=True)
    ids = []
    eos = tokzr.eos_token_id
    for _ in range(96):
        tid = int(logits.view(-1).argmax().item())
        if eos is not None and tid == eos:
            break
        ids.append(tid)
        step = torch.tensor([[tid]], device=runner.device)
        out = runner._forward_tokens(step, past)
        logits = out.logits[0, -1].float()
        past = out.past_key_values
        text = tokzr.decode(ids, skip_special_tokens=True)
        if "]" in text and len(parse_planned_traj(text, 6)) >= 6:
            n_pts = len(__import__("re").findall(r"\(", text))
            if n_pts >= 6:
                break
    greedy_text = tokzr.decode(ids, skip_special_tokens=True)
    print("GREEDY  ", repr(greedy_text[:200]), flush=True)
    print("match", greedy_text.strip() == gen_text.strip(), "n_ids", len(ids), "gen_new", int(gen.shape[1] - prompt_len), flush=True)
    gxy = od_to_ego(parse_planned_traj(gen_text, 6))
    kxy = od_to_ego(parse_planned_traj(greedy_text, 6))
    print("gen", np.round(gxy, 2).tolist(), flush=True)
    print("greedy", np.round(kxy, 2).tolist(), flush=True)
    print("ade greedy vs gen", float(np.mean(np.linalg.norm(gxy - kxy, axis=-1))), flush=True)


if __name__ == "__main__":
    main()
