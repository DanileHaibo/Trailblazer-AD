#!/usr/bin/env python3
"""Cache AutoVLA trajectory-only teacher+hidden on the OpenDrive 6019 val tokens.

Official use_cot=false format: prefill → force <answer> prefix → greedy action tokens.
Does not reuse the CoT teacher cache.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.adapters.autovla import AutoVLARuntime
from trajspec.draft_head import pack_extra, pack_speed
from trajspec.nusc_data import VAL_PKL, load_infos, pack_sample_full, token_index

OD_CACHE = ROOT / "results" / "cache_opendrive_nusc_val.pt"


def od_tokens() -> list[str]:
    blob = torch.load(OD_CACHE, map_location="cpu", weights_only=False)
    return [r["token"] for r in blob["rows"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    tokens = od_tokens()[args.start :: args.stride]
    if args.limit:
        tokens = tokens[: args.limit]
    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out.exists():
        rows = torch.load(out, map_location="cpu", weights_only=False)["rows"]
        print(f"[resume] {len(rows)} already in shard", flush=True)
    done = {r["token"] for r in rows}

    runner = None
    n_new = n_skip = 0
    for token in tokens:
        if token in done:
            continue
        if token not in tok2i:
            n_skip += 1
            continue
        feat = pack_sample_full(infos, tok2i[token], tok2i)
        if feat is None:
            n_skip += 1
            print(f"[skip] cannot pack {token}", flush=True)
            continue
        if runner is None:
            runner = AutoVLARuntime(device=args.device, temperature=None, horizon=args.horizon)
        teacher, info = runner.ar_action_generate(feat, horizon=args.horizon, skip_think=True)
        hidden = info.pop("hidden")
        rows.append(
            {
                "token": token,
                "idx": feat["idx"],
                "cmd": feat["driving_command"],
                "hidden": hidden.cpu().float(),
                "teacher": torch.as_tensor(teacher[: args.horizon], dtype=torch.float32),
                "gt": torch.as_tensor(feat["gt"][: args.horizon], dtype=torch.float32),
                "extra": pack_extra(feat),
                "speed": pack_speed(feat),
            }
        )
        done.add(token)
        n_new += 1
        if n_new % 10 == 0 or n_new == 1:
            print(
                f"[trajonly] shard={len(rows)} new={n_new} skip={n_skip} "
                f"e2e={info['t_e2e_ms']:.0f}ms dec={info['t_decode_ms']:.0f}ms last={token[:8]}",
                flush=True,
            )
        if n_new % 20 == 0:
            torch.save({"rows": rows, "split": "val", "hidden_size": runner.hidden_size, "horizon": args.horizon, "mode": "traj_only"}, out)
            print(f"[ckpt] {out} n={len(rows)}", flush=True)
    hid = runner.hidden_size if runner is not None else 2048
    torch.save({"rows": rows, "split": "val", "hidden_size": hid, "horizon": args.horizon, "mode": "traj_only"}, out)
    print(f"[done] {out} n={len(rows)} new={n_new} skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()
