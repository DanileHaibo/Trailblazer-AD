#!/usr/bin/env python3
"""How much of the teacher's 3 s plan is predictable from *ego state alone*?

Trains the same head but with `hidden` zeroed out, so the only inputs are the
16-d ego kinematics. Three probes:

  kin_only   hidden := 0            -> pure kinematics ceiling
  hidden_only extra := 0            -> pure VLM-prefill ceiling
  both       full inputs            -> current setting

If kin_only already reaches ~3.5 and both only 3.9, the hidden adds little and
the real bottleneck is that the *teacher target itself* is not recoverable from
one prefill state (it depends on the 6 autoregressive steps it has not taken).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/xinhchen2/hhb/TrajSpecDec")
sys.path.insert(0, str(ROOT))

from scripts.run_full_specdec import load_cache  # noqa: E402
from trajspec.accept import AcceptConfig, longest_prefix_accept  # noqa: E402
from trajspec.draft_head import DraftHead  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(head, hs, extra, speed, ys, cfg, bs=512):
    head.eval()
    accs = []
    for i in range(0, hs.shape[0], bs):
        pred = head(
            hs[i : i + bs].to(DEV), extra=extra[i : i + bs].to(DEV), speed=speed[i : i + bs].to(DEV)
        ).cpu()
        for j in range(pred.shape[0]):
            accs.append(longest_prefix_accept(pred[j], ys[i + j], cfg))
    return np.asarray(accs)


def train(head, hs, extra, speed, ys, cfg, steps, lr=1.5e-3, batch=96, val=None, log_every=2000):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-5)
    w = 1.0 + 3.5 * (extra[:, 1] + extra[:, 2]).clamp(0, 1)
    w = (w / w.sum()).cpu()
    w_t = torch.tensor([2.4, 2.0, 1.6, 1.2, 0.9, 0.7], device=DEV).view(1, 6, 1)
    w_xy = torch.tensor([1.15, 2.0], device=DEV).view(1, 1, 2)

    best, best_state = -1.0, None
    t0 = time.time()
    for step in range(steps):
        idx = torch.multinomial(w, batch, replacement=True)
        pred = head(hs[idx].to(DEV), extra=extra[idx].to(DEV), speed=speed[idx].to(DEV))
        tgt = ys[idx].to(DEV)
        err = torch.nn.functional.smooth_l1_loss(pred, tgt, beta=0.2, reduction="none")
        loss = (err * w_t * w_xy).mean()
        box = (pred - tgt).abs()
        hinge = torch.relu(box[:, :, 0] - 0.5) + 1.4 * torch.relu(box[:, :, 1] - 0.3)
        loss = loss + 1.6 * (hinge * w_t.view(1, 6)).mean()
        d_pred, d_tgt = pred[:, 1:] - pred[:, :-1], tgt[:, 1:] - tgt[:, :-1]
        loss = loss + 0.55 * torch.nn.functional.smooth_l1_loss(d_pred, d_tgt, beta=0.15)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if val is not None and (step + 1) % log_every == 0:
            a = evaluate(head, *val, cfg)
            print(
                f"    step {step+1:>6} loss={float(loss):.4f} val={a.mean():.3f} full6={np.mean(a>=6):.3f} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
            if a.mean() > best:
                best = float(a.mean())
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    return float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="auto", choices=["auto", "od"])
    ap.add_argument("--modes", nargs="+", default=["kin_only", "hidden_only", "both"])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = AcceptConfig(lon_m=0.5, lat_m=0.3)
    if args.which == "auto":
        tr = load_cache(ROOT / "results/cache_autovla_nusc_train.pt", refresh_extra=True)
        va = load_cache(ROOT / "results/cache_autovla_nusc_val.pt", refresh_extra=True)
        _, hs, extra, speed, ys, _, _ = tr
        _, vhs, vex, vsp, vys, _, _ = va
        val = (vhs, vex, vsp, vys)
    else:
        pack = load_cache(ROOT / "results/cache_opendrive_nusc_val.pt", refresh_extra=True)
        _, hs, extra, speed, ys, _, _ = pack
        g = torch.Generator().manual_seed(17)
        perm = torch.randperm(hs.shape[0], generator=g)
        nv = int(round(hs.shape[0] * 0.15))
        vi, ti = perm[:nv], perm[nv:]
        hs, extra, speed, ys = hs[ti], extra[ti], speed[ti], ys[ti]
        val = (hs[vi], extra[vi], speed[vi], ys[vi])

    out = []
    for mode in args.modes:
        h_tr, e_tr = hs.clone(), extra.clone()
        h_va, e_va = val[0].clone(), val[1].clone()
        if mode == "kin_only":
            h_tr.zero_()
            h_va.zero_()
        elif mode == "hidden_only":
            e_tr.zero_()
            e_va.zero_()
        v = (h_va, e_va, val[2], val[3])

        print(f"\n[mode {mode}] which={args.which} n={h_tr.shape[0]}", flush=True)
        torch.manual_seed(0)
        head = DraftHead(h_tr.shape[-1], horizon=6, width=512).to(DEV)
        best = train(head, h_tr, e_tr, speed, ys, cfg, args.steps, val=v)
        a = evaluate(head, *v, cfg)
        rec = {
            "which": args.which,
            "mode": mode,
            "n": int(h_tr.shape[0]),
            "val_accept": round(float(a.mean()), 4),
            "val_full6": round(float(np.mean(a >= 6)), 4),
            "dist": np.bincount(a, minlength=7).tolist(),
        }
        print(f"[RESULT {mode}] {json.dumps(rec)}", flush=True)
        out.append(rec)

    Path(ROOT / f"results/diag_inputs_{args.which}{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== input ablation ({args.which}) ===")
    for r in out:
        print(f"  {r['mode']:<14} val_accept={r['val_accept']:.3f} full6={r['val_full6']:.3f}")


if __name__ == "__main__":
    main()
