#!/usr/bin/env python3
"""Dump AutoVLA teacher / draft / spec XY trajectories for NAVSIM PDMS."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.accept import AcceptConfig
from trajspec.draft_head import DraftHead
from trajspec.eval_loop import spec_from_draft
from trajspec.spec_loop import train_step  # noqa: F401  keep import path stable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=ROOT / "results" / "cache_autovla_navtest.pt")
    p.add_argument("--ckpt", type=Path, default=ROOT / "results" / "specdec_autovla_navsim.pt")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "navsim_trajs.pt")
    p.add_argument("--horizon", type=int, default=10)
    args = p.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    rows = cache["rows"]
    hidden = int(ckpt.get("hidden_size", cache.get("hidden_size", 2048)))
    head = DraftHead(hidden, horizon=args.horizon, extra_size=int(ckpt.get("extra_size", 16)))
    head.load_state_dict(ckpt["head"])
    head.eval()
    cfg = AcceptConfig(lon_m=0.50, lat_m=0.30)

    tokens, teacher, draft, spec, n_acc = [], [], [], [], []
    with torch.no_grad():
        for i, r in enumerate(rows):
            y = r["teacher"][: args.horizon].float()
            e = r["extra"].float().view(-1)
            if e.numel() < 16:
                e = torch.nn.functional.pad(e, (0, 16 - e.numel()))
            e = e[:16]
            sp = r["speed"].view(-1)[:1].float()
            d = head(r["hidden"].float().unsqueeze(0), extra=e.unsqueeze(0), speed=sp)[0].cpu()
            s, n, _ = spec_from_draft(d, y, cfg, horizon=args.horizon)
            tokens.append(r["token"])
            teacher.append(y[:, :2].cpu())
            draft.append(d[:, :2].cpu())
            spec.append(torch.as_tensor(s)[:, :2].cpu())
            n_acc.append(int(n))
            if (i + 1) % 1000 == 0:
                print(f"[dump] {i+1}/{len(rows)}", flush=True)
    blob = {
        "tokens": tokens,
        "teacher": torch.stack(teacher),
        "draft": torch.stack(draft),
        "spec": torch.stack(spec),
        "n_accept": torch.tensor(n_acc),
        "horizon": args.horizon,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(f"[done] {args.out} n={len(tokens)} mean_n_accept={float(blob['n_accept'].float().mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
