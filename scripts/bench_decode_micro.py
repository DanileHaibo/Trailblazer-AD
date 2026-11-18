#!/usr/bin/env python3
"""Where does the 17 ms per decoded token actually go?

Microbenchmark of the bare Qwen2-0.5B stack on this L20:

    1 token   -> 16.8 ms (layers) + 0.75 ms (lm_head)
    620 tokens -> 28.2 ms total

A 1-token forward costing 60% of a 620-token prefill is not arithmetic -- it is
launch overhead: 24 layers x ~6 kernels x ~4 us, plus per-call Python. That means
the largest available speedups on this model are *system-level*, not algorithmic,
which is exactly what the published 4-12x driving-VLA papers rely on.

This script measures what each system-level lever is worth here:
  baseline      eager, fp32
  bf16 LM       decoder stack in bf16 (UniAD ops have no bf16 kernel, so it stays fp32)
  compiled      torch.compile(mode="reduce-overhead") on the decoder stack
  cuda_graph    capture the steady-state decode step into a CUDA graph and replay

CUDA-graph capture needs a static KV cache, so we allocate the full KV once and
write into it by index -- the same trick SGLang/vLLM use.
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

import torch

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

CKPT = "/data2/hf_overlap_models/OpenDriveVLA-0.5B"


def bench(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-len", type=int, default=620)
    p.add_argument("--out", default=str(ROOT / "results/decode_microbench.json"))
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
    m = model.model
    L = args.prompt_len
    out = {"device": torch.cuda.get_device_name(0), "prompt_len": L, "results": {}}

    # ---------------- baseline eager fp32 ----------------
    ids1 = torch.randint(0, 1000, (1, 1), device=device)
    idsL = torch.randint(0, 1000, (1, L), device=device)
    with torch.inference_mode():
        t_1 = bench(lambda: m(input_ids=ids1, use_cache=False))
        t_L = bench(lambda: m(input_ids=idsL, use_cache=False))
    out["results"]["eager_fp32"] = {"one_token_ms": round(t_1, 3), "prefill_ms": round(t_L, 3)}
    print(f"eager fp32: 1tok={t_1:.2f}ms  {L}tok={t_L:.2f}ms", flush=True)

    # ---------------- bf16 decoder stack ----------------
    m.layers.to(torch.bfloat16)
    m.norm.to(torch.bfloat16)
    m.embed_tokens.to(torch.bfloat16)
    model.lm_head.to(torch.bfloat16)
    with torch.inference_mode():
        t_1 = bench(lambda: m(input_ids=ids1, use_cache=False))
        t_L = bench(lambda: m(input_ids=idsL, use_cache=False))
    out["results"]["bf16_stack"] = {"one_token_ms": round(t_1, 3), "prefill_ms": round(t_L, 3)}
    print(f"bf16 stack: 1tok={t_1:.2f}ms  {L}tok={t_L:.2f}ms", flush=True)

    # ---------------- compiled ----------------
    try:
        cm = torch.compile(m, mode="reduce-overhead", fullgraph=False)
        with torch.inference_mode():
            t_1 = bench(lambda: cm(input_ids=ids1, use_cache=False))
            t_L = bench(lambda: cm(input_ids=idsL, use_cache=False))
        out["results"]["compiled_bf16"] = {"one_token_ms": round(t_1, 3), "prefill_ms": round(t_L, 3)}
        print(f"compiled bf16: 1tok={t_1:.2f}ms  {L}tok={t_L:.2f}ms", flush=True)
    except Exception as e:  # noqa: BLE001
        out["results"]["compiled_bf16"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print("compile failed:", e, flush=True)

    # ---------------- CUDA graph over the decode step ----------------
    # Static KV cache so the graph sees identical shapes every replay.
    try:
        from transformers.cache_utils import StaticCache

        max_len = L + 128
        dtype = torch.bfloat16
        cache = StaticCache(config=m.config, max_cache_len=max_len, device=device, dtype=dtype)
        # prime: run a real prefill into the static cache
        pe = m.embed_tokens(idsL).to(dtype)
        with torch.inference_mode():
            m(input_ids=None, inputs_embeds=pe, past_key_values=cache, use_cache=True,
              cache_position=torch.arange(L, device=device))

        step_ids = torch.randint(0, 1000, (1, 1), device=device)
        em = m.embed_tokens(step_ids).to(dtype)
        pos1 = torch.tensor([[L]], device=device)
        attn1 = torch.ones(1, L + 1, device=device, dtype=torch.long)

        def _step():
            return m(inputs_embeds=em, attention_mask=attn1, position_ids=pos1,
                     past_key_values=cache, use_cache=True,
                     cache_position=torch.tensor([L], device=device))

        with torch.inference_mode():
            eager_step = bench(_step, n=20, warmup=5)
        out["results"]["static_cache_eager_bf16"] = {"one_token_ms": round(eager_step, 3)}
        print(f"static-cache eager bf16 1tok: {eager_step:.2f}ms", flush=True)

        g = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            _step()  # side stream warmup per PyTorch docs
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                _step()
        with torch.inference_mode():
            graph_ms = bench(lambda: g.replay(), n=50, warmup=10)
        out["results"]["cuda_graph_bf16"] = {"one_token_ms": round(graph_ms, 3)}
        print(f"CUDA-graph replay 1tok: {graph_ms:.2f}ms", flush=True)
    except Exception as e:  # noqa: BLE001
        out["results"]["cuda_graph_bf16"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        print("cuda graph failed:", type(e).__name__, str(e)[:300], flush=True)

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, indent=2))
    print("[write]", args.out)


if __name__ == "__main__":
    main()
