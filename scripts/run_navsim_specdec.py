#!/usr/bin/env python3
"""Train NAVSIM AutoVLA draft on cached teacher trajectories and report accept."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from trajspec.accept import AcceptConfig
from trajspec.draft_head import DraftHead
from trajspec.eval_loop import spec_from_draft
from trajspec.spec_loop import train_step


def _hash_mask(n: int, frac=0.15, seed=17):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(64, int(round(n * frac))) if n > 128 else max(8, n // 5)
    return perm[n_val:], perm[:n_val]


def load_cache(path: Path, horizon: int):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    rows = blob["rows"]
    tokens = [r["token"] for r in rows]
    hs = torch.stack([r["hidden"] for r in rows]).float()
    ys = torch.stack([r["teacher"][:horizon] for r in rows]).float()

    def _pad16(t):
        v = t.float().view(-1)
        if v.numel() >= 16:
            return v[:16]
        return torch.nn.functional.pad(v, (0, 16 - v.numel()))

    extra = torch.stack([_pad16(r["extra"]) for r in rows])
    speed = torch.stack([r["speed"].view(-1)[:1].float() for r in rows]).view(-1)
    gt = torch.stack([torch.as_tensor(r.get("gt", r["teacher"]))[:horizon].float() for r in rows])
    return tokens, hs, extra, speed, ys, gt, int(blob.get("hidden_size", hs.shape[-1]))


def l2_at(pred, gt):
    pred = torch.as_tensor(pred)[:, :2]
    gt = torch.as_tensor(gt)[:, :2]
    n = min(pred.shape[0], gt.shape[0])
    d = torch.linalg.norm(pred[:n] - gt[:n], dim=-1)
    out = {}
    for k in (1, 2, 3, 4, 5):
        m = min(2 * k, n)
        if m > 0:
            out[f"l2_{k}s"] = float(d[:m].mean())
    out["ade"] = float(d.mean()) if n else None
    return out


def _mean_rows(rows):
    keys = sorted({k for r in rows for k in r})
    return {k: round(float(np.mean([r[k] for r in rows if r.get(k) is not None])), 4) for k in keys}


def eval_split(head, hs, extra, speed, ys, gt, cfg: AcceptConfig, horizon: int):
    head.eval()
    n_acc = []
    t_rows, d_rows, s_rows = [], [], []
    with torch.no_grad():
        for i in range(hs.shape[0]):
            h = hs[i : i + 1].to(DEVICE)
            e = extra[i : i + 1].to(DEVICE)
            sp = speed[i : i + 1].to(DEVICE)
            teacher = ys[i]
            draft = head(h, extra=e, speed=sp)[0].cpu()
            spec, n, _ = spec_from_draft(draft, teacher, cfg, horizon=horizon)
            n_acc.append(int(n))
            t_rows.append(l2_at(teacher, gt[i]))
            d_rows.append(l2_at(draft, gt[i]))
            s_rows.append(l2_at(spec, gt[i]))
    dist = {str(k): round(n_acc.count(k) / max(1, len(n_acc)), 3) for k in range(horizon + 1)}
    return {
        "n": len(n_acc),
        "mean_n_accept": round(float(np.mean(n_acc)), 3) if n_acc else 0.0,
        "accept_dist": dist,
        "teacher_vs_gt": _mean_rows(t_rows) if t_rows else {},
        "draft_vs_gt": _mean_rows(d_rows) if d_rows else {},
        "spec_vs_gt": _mean_rows(s_rows) if s_rows else {},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=ROOT / "results" / "cache_autovla_navtest.pt")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "specdec_autovla_navsim.json")
    p.add_argument("--ckpt", type=Path, default=ROOT / "results" / "specdec_autovla_navsim.pt")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--target-accept", type=float, default=3.5)
    args = p.parse_args()

    tokens, hs, extra, speed, ys, gt, hidden = load_cache(args.cache, args.horizon)
    tr_idx, va_idx = _hash_mask(len(tokens))
    print(
        f"[data] n={len(tokens)} train={len(tr_idx)} holdout={len(va_idx)} hidden={hidden} H={args.horizon}",
        flush=True,
    )
    cfg = AcceptConfig(lon_m=0.50, lat_m=0.30)
    head = DraftHead(hidden, horizon=args.horizon, width=512).to(DEVICE)
    hist = []
    best = -1.0
    opt = torch.optim.AdamW(head.parameters(), lr=1.5e-3, weight_decay=2e-4)
    batch = min(32, int(tr_idx.numel()))
    for rnd in range(args.rounds):
        head.train()
        losses = []
        for _ in range(args.steps):
            idx = tr_idx[torch.randint(0, len(tr_idx), (batch,))]
            loss = train_step(
                head,
                hs[idx].to(DEVICE),
                ys[idx].to(DEVICE),
                opt,
                extra=extra[idx].to(DEVICE),
                speed=speed[idx].to(DEVICE),
            )
            losses.append(float(loss))
        summary = eval_split(
            head, hs[va_idx], extra[va_idx], speed[va_idx], ys[va_idx], gt[va_idx], cfg, args.horizon
        )
        summary["round"] = rnd
        summary["loss"] = float(np.mean(losses[-200:])) if losses else 0.0
        hist.append({k: summary[k] for k in ("round", "loss", "mean_n_accept", "accept_dist")})
        print(f"[round {rnd}] loss={summary['loss']:.4f} accept={summary['mean_n_accept']:.3f}", flush=True)
        if summary["mean_n_accept"] >= best:
            best = summary["mean_n_accept"]
            torch.save(
                {"head": head.state_dict(), "hidden_size": hidden, "horizon": args.horizon, "extra_size": 16},
                args.ckpt,
            )
        if summary["mean_n_accept"] >= args.target_accept:
            break
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    head.load_state_dict(blob["head"])
    holdout = eval_split(head, hs[va_idx], extra[va_idx], speed[va_idx], ys[va_idx], gt[va_idx], cfg, args.horizon)
    full = eval_split(head, hs, extra, speed, ys, gt, cfg, args.horizon)
    out = {
        "name": "autovla_navsim",
        "cache": str(args.cache),
        "ckpt": str(args.ckpt),
        "n_cache": len(tokens),
        "n_holdout": int(len(va_idx)),
        "accept": {"lon_m": 0.50, "lat_m": 0.30},
        "holdout": holdout,
        "full_cache": full,
        "train_hist": hist,
        "note": "Draft supervised on AutoVLA teacher (not GT). Holdout is 15% of navtest cache.",
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"n_cache": out["n_cache"], "n_holdout": out["n_holdout"], "holdout": holdout}, indent=2), flush=True)


if __name__ == "__main__":
    main()
