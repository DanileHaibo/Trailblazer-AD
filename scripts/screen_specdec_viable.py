#!/usr/bin/env python3
"""Screen a driving VLA for speculative-decoding viability BEFORE investing in it.

The expensive failure mode is discovering, after caching and training a draft
head, that one parallel verify cannot reproduce the model's own serial decode.
That happens with multi-dimensional RoPE (Qwen2-VL / 2.5-VL / 3-VL, InternVL):
the visual patch positions are encoded in (t,h,w) and the delta between the
prefill positions and the continuation positions is not expressible as a plain
1-D offset, so `logits[i]` from a batched verify does not equal the greedy token
the model would have produced serially.

This script checks the essential property directly:

    teacher ids from ONE batched forward over [prompt + K candidate tokens]
      ==
    teacher ids from K sequential single-token forwards

If they differ, the model cannot host speculative decoding, regardless of how
good its draft would be.

Usage:
  python scripts/screen_specdec_viable.py --which od|waypoints
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/home/xinhchen2/hhb/TrajSpecDec")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def check(one_shot, serial_fn, tag, K=24):
    """Compare batched logits against serial greedy ids."""
    batched = one_shot()
    serial = serial_fn()
    n = min(len(batched), len(serial))
    eq = sum(int(a == b) for a, b in zip(batched[:n], serial[:n]))
    print(f"[{tag}] first {n} teacher ids")
    print(f"  batched: {batched[:n]}")
    print(f"  serial : {serial[:n]}")
    print(f"  match   : {eq}/{n}  ({100.0*eq/max(n,1):.0f}%)")
    verdict = "VIABLE" if eq == n else "NOT VIABLE"
    print(f"  => {verdict}\n")
    return eq == n


def screen_waypoints():
    from trajspec.adapters.waypoints import WaypointsRuntime
    from trajspec.nusc_data import VAL_PKL, load_infos, pack_sample_full, token_index

    rt = WaypointsRuntime()
    tok = rt.processor.tokenizer
    infos = load_infos(VAL_PKL)
    t2i = token_index(infos)
    s = None
    for idx in range(0, 400, 8):
        s = pack_sample_full(infos, idx, t2i)
        if s is not None:
            break
    inp = {k: (v.to(rt.device) if torch.is_tensor(v) else v) for k, v in rt._inputs(s).items()}
    plen = int(inp["input_ids"].shape[1])

    cand = tok(
        "Planned Trajectory:\n[(0.40,-0.02), (0.75,-0.04), (1.04,-0.06), (1.28,-0.08)]",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(rt.device)
    K = int(cand.shape[1])

    def one_shot():
        with torch.no_grad():
            pre = rt.model(**inp, use_cache=True)
            pos, _ = rt.model.model.get_rope_index(
                inp["input_ids"], inp["mm_token_type_ids"],
                image_grid_thw=inp.get("image_grid_thw"), attention_mask=inp.get("attention_mask"),
            )
            start = int(pos[0, 0, -1]) + 1
            base = torch.arange(start, start + K, device=rt.device).view(1, 1, -1).expand(3, 1, K)
            am = torch.ones(1, plen + K, device=rt.device, dtype=torch.long)
            mm = torch.zeros(1, K, dtype=torch.int32, device=rt.device)
            o = rt.model(input_ids=cand, past_key_values=pre.past_key_values, use_cache=True,
                         position_ids=base, attention_mask=am, mm_token_type_ids=mm)
        f = o.logits[0].float()
        return [int(pre.logits[0, -1].float().argmax())] + [int(f[i].argmax()) for i in range(K - 1)]

    def serial():
        with torch.no_grad():
            pre = rt.model(**inp, use_cache=True)
        past = pre.past_key_values
        ids = [int(pre.logits[0, -1].float().argmax())]
        for j in range(K - 1):
            with torch.no_grad():
                o = rt.model(input_ids=cand[:, j : j + 1], past_key_values=past, use_cache=True)
            past = o.past_key_values
            ids.append(int(o.logits[0, -1].float().argmax()))
        return ids

    return check(one_shot, serial, "waypoints/Qwen3-VL-4B", K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="waypoints", choices=["waypoints"])
    args = ap.parse_args()
    ok = screen_waypoints()
    print("RESULT:", "teacher supports batched verify" if ok else
          "teacher does NOT support batched verify -> reject for specdec")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
