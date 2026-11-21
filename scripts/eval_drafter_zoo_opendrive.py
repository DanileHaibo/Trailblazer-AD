#!/usr/bin/env python3
"""Zero-parameter drafts vs the neural DraftHead, same accept rule, same split.

Zora (arXiv 2611.11098) argues the draft model is the wrong place to spend compute
and replaces it with retrieval: look up a similar past ego state, reuse the actions
that followed it. That is a direct challenge to TrajSpecDec's learned DraftHead, so
it belongs in the comparison as a first-class baseline rather than a footnote.

Three drafters, one accept rule (lon 0.50 m / lat 0.30 m), one holdout:

  bicycle   physical prior only: constant speed and yaw-rate rollout
  rag       k-NN over ego kinematics (speed, accel, yaw-rate, kappa, command),
            draft = the teacher trajectory of the nearest *training* frame
  neural    the trained DraftHead (ours)

Retrieval uses only training-split frames, so the holdout stays clean. Distance is
computed in the same standardized 16-d `extra` vector the neural head consumes, so
the comparison is "same inputs, different function class".
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.accept import AcceptConfig, longest_prefix_accept
from trajspec.metrics import l2_horizon
from trajspec.draft_head import DraftHead, pack_extra, pack_speed
from trajspec.eval_loop import l2_uniad_at
from trajspec.nusc_data import extras_for_tokens
from trajspec.sd_baselines import hash_holdout_idx, seam_blend

CACHE = ROOT / "results/cache_opendrive_nusc_val.pt"
HEAD = ROOT / "results/specdec_opendrive_heldout.pt"
HORIZON = 6
DT = 0.5


def load_cache(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    rows = blob["rows"]
    tokens = [r["token"] for r in rows]
    hs = torch.stack([r["hidden"] for r in rows]).float()
    ys = torch.stack([r["teacher"][:HORIZON] for r in rows]).float()
    extra = []
    for r in rows:
        e = r["extra"].float().view(-1)[:16]
        extra.append(torch.nn.functional.pad(e, (0, 16 - e.numel())))
    extra = torch.stack(extra)
    speed = torch.stack([r["speed"].float().view(-1)[:1] for r in rows]).view(-1)
    gt = []
    for r in rows:
        g = r.get("gt")
        if g is None:
            gt.append(torch.zeros(HORIZON, 2))
        else:
            g = torch.as_tensor(g).float()
            gt.append(g[:HORIZON] if g.numel() >= HORIZON * 2 else torch.zeros(HORIZON, 2))
    gt = torch.stack(gt)

    # Some cached `extra` rows are only 6-d (older packing). Zero-padding those
    # silently corrupts the draft head's input, so rebuild every row from the VAD
    # pkls exactly like eval_sd_baselines_offline.py does.
    rebuilt = extras_for_tokens(tokens)
    hit = 0
    for i, tok in enumerate(tokens):
        if tok in rebuilt:
            e, s, g = rebuilt[tok]
            extra[i] = e.float().view(-1)[:16]
            speed[i] = s.view(-1)[:1].float()
            if g.abs().sum() > 0.05:
                gt[i] = g[:HORIZON].float()
            hit += 1
    print(f"[data] extras rebuilt {hit}/{len(tokens)}", flush=True)

    return {
        "tokens": tokens,
        "hs": hs,
        "ys": ys,
        "extra": extra,
        "speed": speed,
        "gt": gt,
        "hidden_size": int(blob.get("hidden_size", hs.shape[-1])),
    }


def bicycle_prior(extra: torch.Tensor, horizon: int = HORIZON, dt: float = DT) -> torch.Tensor:
    """Constant-speed, constant-yaw-rate rollout from ego kinematics.

    extra layout (see draft_head.pack_extra): v/15 at [0], yaw_rate at [8], kappa/3 at [9].
    """
    out = torch.zeros(horizon, 2)
    v = float(extra[0].item()) * 15.0
    yaw_rate = float(extra[8].item()) if extra.numel() > 8 else 0.0
    heading = 0.0
    pos = torch.zeros(2)
    cur_v = v
    for k in range(horizon):
        heading = heading + yaw_rate * dt
        step = torch.tensor([cur_v * dt * math.cos(heading), cur_v * dt * math.sin(heading)])
        pos = pos + step
        out[k] = pos
    return out


def standardize(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd


class RagDrafter:
    """k-NN retrieval drafter over standardized ego-kinematics features."""

    def __init__(self, feats: torch.Tensor, trajs: torch.Tensor, k: int = 1):
        self.z = standardize(feats)
        self.trajs = trajs
        self.k = int(k)

    @torch.no_grad()
    def draft(self, query: torch.Tensor) -> torch.Tensor:
        q = (query.view(1, -1) - self.z.mean(dim=0, keepdim=True)) / self.z.std(dim=0, keepdim=True).clamp_min(1e-6)
        d = ((self.z - q) ** 2).sum(dim=-1)
        idx = torch.topk(d, k=min(self.k, d.numel()), largest=False).indices
        return self.trajs[idx].mean(dim=0)


def eval_drafter(name, drafts, ys, gt, cfg, speed, horizon=HORIZON):
    rows = []
    for i, draft in enumerate(drafts):
        teacher = ys[i]
        n_acc = longest_prefix_accept(draft[:horizon], teacher[:horizon], cfg)
        if n_acc >= horizon:
            spec = draft[:horizon].clone()
        else:
            spec = seam_blend(draft[:horizon], teacher[:horizon], n_acc, horizon=horizon)
        g = gt[i].numpy()
        has_gt = bool(np.linalg.norm(g) > 0.2)
        spec_np = spec.numpy()
        teacher_np = teacher.numpy()
        rows.append(
            {
                "n_accept": int(n_acc),
                "has_gt": has_gt,
                "ade_vs_teacher": round(float(np.linalg.norm(spec_np - teacher_np, axis=-1).mean()), 4),
                "spec_ade": round(float(l2_horizon(spec_np, g)["ade"]), 4) if has_gt else None,
                "teacher_ade": round(float(l2_horizon(teacher_np, g)["ade"]), 4) if has_gt else None,
            }
        )
    accs = [r["n_accept"] for r in rows]
    gt_rows = [r for r in rows if r["has_gt"]]
    # decode-step accounting: 1 parallel verify + remaining serial steps
    steps = [1 if a >= horizon else 1 + max(horizon - a - 1, 0) for a in accs]
    out = {
        "drafter": name,
        "n": len(rows),
        "mean_n_accept": round(float(np.mean(accs)), 3),
        "frac_accept_ge2": round(float(np.mean([a >= 2 for a in accs])), 3),
        "frac_full6": round(float(np.mean([a >= horizon for a in accs])), 3),
        "frac_zero": round(float(np.mean([a == 0 for a in accs])), 3),
        "decode_steps": round(float(np.mean(steps)), 3),
        "decode_speedup": round(float(horizon / np.mean(steps)), 3),
        "ade_vs_teacher": round(float(np.mean([r["ade_vs_teacher"] for r in rows])), 4),
        "spec_ade": round(float(np.mean([r["spec_ade"] for r in gt_rows])), 4) if gt_rows else None,
        "teacher_ade": round(float(np.mean([r["teacher_ade"] for r in gt_rows])), 4) if gt_rows else None,
        "n_with_gt": len(gt_rows),
    }
    print(
        f"[{name:8s}] accept={out['mean_n_accept']:.3f} full6={out['frac_full6']:.3f} "
        f"zero={out['frac_zero']:.3f} steps={out['decode_steps']:.2f} "
        f"ADE_T={out['ade_vs_teacher']:.4f} ADE_G={out['spec_ade']}",
        flush=True,
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=1, help="RAG neighbours to average")
    p.add_argument("--lon", type=float, default=0.5)
    p.add_argument("--lat", type=float, default=0.3)
    p.add_argument("--out", default=str(ROOT / "results/drafter_zoo_opendrive.json"))
    args = p.parse_args()
    if not Path(args.out).is_absolute():
        args.out = str(ROOT / args.out)

    cfg = AcceptConfig(lon_m=args.lon, lat_m=args.lat)
    split = load_cache(CACHE)
    n = len(split["tokens"])
    hold = hash_holdout_idx(n, frac=0.15, seed=17)
    mask = torch.ones(n, dtype=torch.bool)
    mask[hold] = False
    tr = mask.nonzero(as_tuple=False).view(-1)
    te = hold
    print(f"[data] total={n} train={len(tr)} holdout={len(te)}", flush=True)

    feats_tr = split["extra"][tr]
    trajs_tr = split["ys"][tr]

    # ---- drafters ----
    results = []

    drafts = [bicycle_prior(split["extra"][i]) for i in te.tolist()]
    results.append(eval_drafter("bicycle", drafts, split["ys"][te], split["gt"][te], cfg, split["speed"][te]))

    rag = RagDrafter(feats_tr, trajs_tr, k=args.k)
    drafts = [rag.draft(split["extra"][i]) for i in te.tolist()]
    results.append(eval_drafter(f"rag(k={args.k})", drafts, split["ys"][te], split["gt"][te], cfg, split["speed"][te]))

    blob = torch.load(HEAD, map_location="cpu", weights_only=False)
    head = DraftHead(int(blob["hidden_size"]), horizon=HORIZON, width=512)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    drafts = []
    with torch.no_grad():
        for j, i in enumerate(te.tolist()):
            pred = head(
                split["hs"][i : i + 1],
                extra=split["extra"][i : i + 1],
                speed=split["speed"][i : i + 1],
            )[0]
            drafts.append(pred[:HORIZON].float().cpu())
    results.append(eval_drafter("neural", drafts, split["ys"][te], split["gt"][te], cfg, split["speed"][te]))

    summary = {
        "protocol": {
            "cache": str(CACHE),
            "split": "hash holdout 15% seed=17; RAG retrieval pool = the other 85%",
            "accept": {"lon_m": args.lon, "lat_m": args.lat},
            "rag_k": args.k,
            "note": (
                "All three drafters share one accept rule and one seam-repair tail, so the only "
                "difference is the draft distribution. decode_steps = 1 verify + remaining serial "
                "waypoints; decode_speedup is step accounting, not wall clock."
            ),
        },
        "results": {r["drafter"]: r for r in results},
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
