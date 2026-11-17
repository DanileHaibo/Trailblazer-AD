#!/usr/bin/env python3
"""Dump a few nuScenes cases: GT vs AutoVLA teacher vs draft, plus action tokens."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.accept import AcceptConfig, longest_prefix_accept
from trajspec.adapters.autovla import AutoVLARuntime
from trajspec.draft_head import DraftHead
from trajspec.nusc_data import VAL_PKL, load_infos, pack_sample, token_index

CASES = [10, 43, 73, 253]  # accept 1 / 0 / 0 / 2 from nusc_openloop.json
CKPT = ROOT / "results" / "draft_autovla_nusc.pt"
LON, LAT = 0.5, 0.3


def fmt(arr, nd=2):
    a = np.asarray(arr, dtype=np.float32)
    return np.round(a, nd).tolist()


def main():
    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    runner = AutoVLARuntime(device="cuda")
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    head = DraftHead(int(blob["hidden_size"]), horizon=6).cuda()
    head.load_state_dict(blob["state_dict"])
    head.eval()
    cfg = AcceptConfig(lon_m=LON, lat_m=LAT)
    cb = runner.action_tokenizer.code_book
    print(f"[codebook] shape={tuple(cb.shape)} n_bins={cb.shape[0]} patch={tuple(cb.shape[1:])}", flush=True)

    for idx in CASES:
        s = pack_sample(infos, idx, tok2i)
        if s is None:
            print(f"\n=== idx={idx} SKIP (pack failed) ===")
            continue
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _StopAfterActions(StoppingCriteria):
            def __init__(self, start_id, n, prompt_len):
                super().__init__()
                self.start_id, self.n, self.prompt_len = start_id, n, prompt_len

            def __call__(self, input_ids, scores, **kwargs):
                gen = input_ids[0, self.prompt_len :]
                return bool((gen >= self.start_id).sum().item() >= self.n)

        inputs = runner.get_prompt(s)
        model_inputs = runner._to_dev(inputs)
        prompt_len = int(inputs["input_ids"].shape[1])
        outputs = runner.vlm.generate(
            **model_inputs,
            max_new_tokens=192,
            do_sample=False,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList(
                [_StopAfterActions(runner.action_start_id, runner.horizon, prompt_len)]
            ),
        )
        trimmed = outputs[0, prompt_len:]
        if trimmed.numel() and trimmed[-1].item() == runner.processor.tokenizer.eos_token_id:
            trimmed = trimmed[:-1]
        act = trimmed[trimmed >= runner.action_start_id][:10]
        ids = []
        for tid in act.cpu().tolist():
            tok = runner.processor.tokenizer.decode(tid)
            ids.append(int(tok.split("_")[1].replace(">", "")))
        teacher = runner._decode_actions(trimmed)
        print(f"action tokens <action_i> i={ids}  raw_ids={act.cpu().tolist()}")
        hidden = runner.prefill_hidden(s)
        with torch.no_grad():
            draft = head(hidden.cuda())[0, :6].cpu().numpy()
        teacher6 = np.asarray(teacher[:6], dtype=np.float32)
        gt = np.asarray(s["gt"][:6], dtype=np.float32)
        dlt = teacher6 - draft
        dlg = gt - teacher6
        n_acc = longest_prefix_accept(
            torch.as_tensor(draft), torch.as_tensor(teacher6), cfg
        )
        lon_ok = np.abs(dlt[:, 0]) <= LON
        lat_ok = np.abs(dlt[:, 1]) <= LAT
        ok = lon_ok & lat_ok
        print(f"\n=== idx={s['idx']} token={s['token'][:8]} cmd={s['driving_command']} v={s['vehicle_velocity'][0]:.2f} m/s ===")
        print(f"GT      {fmt(gt)}")
        print(f"teacher {fmt(teacher6)}")
        print(f"draft   {fmt(draft)}")
        print(f"d-t dx  {fmt(dlt[:, 0])}  |dx|<={LON} {lon_ok.tolist()}")
        print(f"d-t dy  {fmt(dlt[:, 1])}  |dy|<={LAT} {lat_ok.tolist()}")
        print(f"accept  prefix={int(n_acc)}/6  ok={ok.tolist()}")
        print(f"t-gt L2 {fmt(np.linalg.norm(dlg, axis=-1), 3)}")
        print(f"d-gt L2 {fmt(np.linalg.norm(gt - draft, axis=-1), 3)}")
        print(f"d-t  L2 {fmt(np.linalg.norm(dlt, axis=-1), 3)}")
        print(f"teacher end {fmt(teacher6[-1])}  draft end {fmt(draft[-1])}  GT end {fmt(gt[-1])}")


if __name__ == "__main__":
    main()
