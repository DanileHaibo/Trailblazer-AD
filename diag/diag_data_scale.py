#!/usr/bin/env python3
"""Data-scaling test: does more *teacher-labelled* data lift held-out accept?

The capacity test showed a 19M-param head memorises the train split
(train accept 5.997) while val goes *down* -> the limit is data, not model size.
Here we keep the same head and scale the number of training frames.

Sources
  nusc_train  2500 frames, nuScenes train split   (in-domain)
  navtest    12146 frames, 136 unseen logs        (out-of-domain, free extra)

The val set is never in any train source (token-disjoint, verified).
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


def sub_pack(pack, k):
    return (list(pack[0])[:k], pack[1][:k], pack[2][:k], pack[3][:k], pack[4][:k], pack[5][:k])


def concat_packs(packs):
    toks, hs, ex, sp, ys, gt = [], [], [], [], [], []
    for p in packs:
        t, h, e, s, y, g = p[:6]
        toks += list(t)
        hs.append(h)
        ex.append(e)
        sp.append(s)
        ys.append(y)
        gt.append(g)
    d = max(x.shape[-1] for x in ex)
    ex = [torch.nn.functional.pad(x, (0, d - x.shape[-1])) for x in ex]
    return toks, torch.cat(hs, 0).float(), torch.cat(ex, 0).float(), torch.cat(sp, 0).float(), torch.cat(ys, 0).float(), torch.cat(gt, 0).float()


@torch.no_grad()
def evaluate(head, hs, extra, speed, ys, cfg, bs=512):
    head.eval()
    accs, errs = [], []
    for i in range(0, hs.shape[0], bs):
        pred = head(
            hs[i : i + bs].to(DEV), extra=extra[i : i + bs].to(DEV), speed=speed[i : i + bs].to(DEV)
        ).cpu()
        errs.append((pred - ys[i : i + bs]).abs().mean(dim=(1, 2)).numpy())
        for j in range(pred.shape[0]):
            accs.append(longest_prefix_accept(pred[j], ys[i + j], cfg))
    return np.asarray(accs), np.concatenate(errs)


def train(head, pack, cfg, steps, lr=1.5e-3, batch=96, log_every=2000, val=None):
    _, hs, extra, speed, ys = pack[:5]
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-5)
    turn = (extra[:, 1] + extra[:, 2]).clamp(0, 1)
    w = 1.0 + 3.5 * turn
    w = (w / w.sum()).cpu()

    best, best_state = -1.0, None
    t0 = time.time()
    for step in range(steps):
        idx = torch.multinomial(w, batch, replacement=True)
        pred = head(hs[idx].to(DEV), extra=extra[idx].to(DEV), speed=speed[idx].to(DEV))
        tgt = ys[idx].to(DEV)
        w_t = torch.tensor([2.4, 2.0, 1.6, 1.2, 0.9, 0.7], device=DEV).view(1, 6, 1)
        w_xy = torch.tensor([1.15, 2.0], device=DEV).view(1, 1, 2)
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
            a, _ = evaluate(head, *val, cfg)
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
    ap.add_argument("--configs", nargs="+", default=["nusc2500", "nusc2500+nav4000", "navtest_all"])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = AcceptConfig(lon_m=0.5, lat_m=0.3)

    vpack = load_cache(ROOT / "results/cache_autovla_nusc_val.pt", refresh_extra=True)
    _, vhs, vex, vsp, vys, _, _ = vpack
    val = (vhs, vex, vsp, vys)
    print(f"[val] n={vhs.shape[0]} (token-disjoint from all train sources)", flush=True)

    nusc = load_cache(ROOT / "results/cache_autovla_nusc_train.pt", refresh_extra=True)
    nav = load_cache(ROOT / "results/cache_autovla_navtest.pt", refresh_extra=True)
    vtoks = set(vpack[0])
    nav_only = [t for t in nav[0] if t not in vtoks]
    print(f"[navtest] {len(nav[0])} rows, {len(nav_only)} token-disjoint from val", flush=True)

    out = []
    for conf in args.configs:
        if conf == "nusc2500":
            pack = nusc
        elif conf.startswith("nusc2500+nav"):
            k = int(conf.split("nav")[1])
            pack = concat_packs([nusc, sub_pack(nav, k)])
        elif conf == "navtest_all":
            pack = nav
        elif conf == "nusc+navall":
            pack = concat_packs([nusc, nav])
        else:
            raise ValueError(conf)

        n = pack[1].shape[0]
        print(f"\n[cfg {conf}] train n={n} steps={args.steps}", flush=True)
        torch.manual_seed(0)
        head = DraftHead(pack[1].shape[-1], horizon=6, width=args.width).to(DEV)
        best = train(head, pack, cfg, args.steps, val=val)
        a, err = evaluate(head, *val, cfg)
        rec = {
            "config": conf,
            "train_n": int(n),
            "steps": args.steps,
            "val_accept": round(float(a.mean()), 4),
            "val_full6": round(float(np.mean(a >= 6)), 4),
            "val_mae": round(float(err.mean()), 4),
            "best_during_train": round(best, 4),
            "dist": np.bincount(a, minlength=7).tolist(),
        }
        print(f"[RESULT {conf}] {json.dumps(rec)}", flush=True)
        out.append(rec)
        torch.save(
            {"state_dict": head.state_dict(), "hidden_size": pack[1].shape[-1], "width": args.width, "config": conf},
            ROOT / f"results/diag_data_{conf.replace('+','_')}{args.tag}.pt",
        )

    Path(ROOT / f"results/diag_data_scaling{args.tag}.json").write_text(json.dumps(out, indent=2))
    print("\n=== data scaling ===")
    for r in out:
        print(
            f"  {r['config']:<22} n={r['train_n']:<6} val_accept={r['val_accept']:.3f} "
            f"full6={r['val_full6']:.3f} mae={r['val_mae']:.3f}"
        )


if __name__ == "__main__":
    main()
