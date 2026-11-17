#!/usr/bin/env python3
"""AutoVLA native (no speculative decoding) over the full nuScenes val split.

Runs the teacher's own generate() on every frame in the OpenDrive val order
(6019) so the result is directly comparable to a spec run over the same tokens.
Writes one JSONL record per frame: token, native xy, optional gt.

This is the "native" arm of native-vs-TrajSpecDec; the teacher weights are the
official AutoVLA PDMS checkpoint, unmodified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

import numpy as np
import torch

ROOT = Path("/home/xinhchen2/hhb/TrajSpecDec")
sys.path.insert(0, str(ROOT))

from trajspec.adapters.autovla import AutoVLARuntime  # noqa: E402
from trajspec.nusc_data import VAL_PKL, load_infos, pack_sample_full, token_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--od-cache", default=str(ROOT / "results/cache_opendrive_nusc_val.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/native_autovla_full_6019.jsonl"))
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out.exists():
        with out.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["token"])
                except Exception:
                    pass
        print(f"[resume] {len(done)} already done", flush=True)
    elif out.exists():
        out.unlink()

    order = [r["token"] for r in torch.load(args.od_cache, map_location="cpu", weights_only=False)["rows"]]
    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    print(f"[data] {len(order)} tokens, infos={len(infos)}", flush=True)

    runner = AutoVLARuntime(device="cuda")
    n_done = 0
    t0 = time.time()
    for ci in range(args.start, len(order)):
        token = order[ci]
        if token in done:
            continue
        if token not in tok2i:
            continue
        s = pack_sample_full(infos, tok2i[token], tok2i)
        if s is None:
            continue
        with torch.inference_mode():
            xy = runner.predict(s)
        rec = {
            "token": token,
            "cache_index": ci,
            "native_xy": np.asarray(xy[:6], dtype=np.float32).tolist(),
            "gt": np.asarray(s["gt"][:6], dtype=np.float32).tolist(),
        }
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        n_done += 1
        if n_done % 20 == 0 or n_done == 1:
            el = time.time() - t0
            print(f"[{ci+1}/{len(order)}] n={n_done} t={el:.0f}s {el/max(n_done,1):.2f}s/it", flush=True)
        if args.max_samples and n_done >= args.max_samples:
            break

    print(f"[done] {out} rows={n_done} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
