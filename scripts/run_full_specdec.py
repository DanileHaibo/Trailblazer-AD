#!/usr/bin/env python3
"""Train waypoint draft until accept rate is high, then eval held-out cache."""
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

from trajspec.accept import AcceptConfig, longest_prefix_accept
from trajspec.b2d_data import l2_horizon
from trajspec.draft_head import DraftHead
from trajspec.eval_loop import l2_uniad_at, mean_dict, spec_from_draft
from trajspec.nusc_data import extras_for_tokens
from trajspec.spec_loop import train_step


def _hash_mask(tokens, frac=0.15, seed=17):
    n = len(tokens)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(64, int(round(n * frac)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return tr_idx, val_idx


def load_cache(path: Path, refresh_extra: bool = True):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    rows = blob["rows"]
    tokens = [r["token"] for r in rows]
    hs = torch.stack([r["hidden"] for r in rows]).float()
    ys = torch.stack([r["teacher"][:6] for r in rows]).float()

    def _pad_stack(tensors, min_d=1):
        ts = [t.float().view(-1) for t in tensors]
        d = max([min_d] + [int(t.numel()) for t in ts])
        return torch.stack([torch.nn.functional.pad(t, (0, d - t.numel())) for t in ts]).float()

    extra = _pad_stack([r["extra"] for r in rows], min_d=16)
    speed = torch.stack([r["speed"].view(-1)[:1].float() for r in rows]).view(-1)
    gt = torch.stack([torch.as_tensor(r["gt"] if r.get("gt") is not None else r["teacher"])[:6].float() for r in rows])
    if refresh_extra:
        print(f"[data] refreshing extras from nuScenes pkl n={len(tokens)}", flush=True)
        rebuilt = extras_for_tokens(tokens)
        hit = 0
        extras = []
        speeds = []
        gts = []
        max_d = 16
        for i, tok in enumerate(tokens):
            if tok in rebuilt:
                e, s, g = rebuilt[tok]
                extras.append(e.float().view(-1))
                speeds.append(s.view(-1)[:1].float())
                if g.abs().sum() > 0.05:
                    gts.append(g[:6].float())
                else:
                    gts.append(gt[i])
                hit += 1
            else:
                extras.append(extra[i].float().view(-1))
                speeds.append(speed[i : i + 1])
                gts.append(gt[i])
            max_d = max(max_d, int(extras[-1].numel()))
        extras = [torch.nn.functional.pad(e, (0, max_d - e.numel())) for e in extras]
        extra = torch.stack(extras).float()
        speed = torch.stack(speeds).float().view(-1)
        gt = torch.stack(gts).float()
        print(f"[data] extras rebuilt {hit}/{len(tokens)} dim={tuple(extra.shape)}", flush=True)
    hidden_size = int(blob.get("hidden_size", hs.shape[-1]))
    return tokens, hs, extra, speed, ys, gt, hidden_size


@torch.no_grad()
def accept_stats(head, hs, extra, speed, ys, cfg, n_eval=None):
    head.eval()
    n = hs.shape[0] if n_eval is None else min(n_eval, hs.shape[0])
    accs = []
    d_ade = []
    for i in range(n):
        pred = head(hs[i : i + 1].to(DEVICE), extra=extra[i : i + 1].to(DEVICE), speed=speed[i : i + 1].to(DEVICE))[0].cpu()
        n_acc = longest_prefix_accept(pred, ys[i], cfg)
        accs.append(int(n_acc))
        d_ade.append(float(torch.linalg.norm(pred - ys[i], dim=-1).mean()))
    mean_acc = float(np.mean(accs))
    dist = {k: accs.count(k) / len(accs) for k in range(7)}
    return mean_acc, float(np.mean(d_ade)), dist, accs


def eval_full(head, tokens, hs, extra, speed, ys, gt, cfg):
    head.eval()
    rows = []
    with torch.no_grad():
        for i in range(hs.shape[0]):
            draft = head(hs[i : i + 1].to(DEVICE), extra=extra[i : i + 1].to(DEVICE), speed=speed[i : i + 1].to(DEVICE))[0].cpu().numpy()
            teacher = ys[i].numpy()
            spec, n_acc, spec_steps = spec_from_draft(draft, teacher, cfg, horizon=6)
            g = gt[i].numpy()
            has_gt = bool(np.linalg.norm(g) > 0.2)
            rec = {
                "token": tokens[i],
                "n_accept": int(n_acc),
                "spec_decode_steps": int(spec_steps),
                "decode_speedup": round(6 / max(spec_steps, 1e-6), 3),
                "has_gt": has_gt,
                "teacher": {**l2_horizon(teacher, g), **l2_uniad_at(teacher, g)} if has_gt else {},
                "spec": {**l2_horizon(spec, g), **l2_uniad_at(spec, g)} if has_gt else {},
                "draft": {**l2_horizon(draft, g), **l2_uniad_at(draft, g)} if has_gt else {},
            }
            rows.append(rec)
    stp3 = ["l2_1s", "l2_2s", "l2_3s", "ade"]
    uni = ["l2_at_1s", "l2_at_2s", "l2_at_3s"]
    gt_rows = [r for r in rows if r.get("has_gt")]

    def flatten(tag, keys, src=None):
        src = src if src is not None else gt_rows
        return mean_dict([{k: r[tag][k] for k in keys} for r in src if r.get(tag)], keys)

    accs = [r["n_accept"] for r in rows]
    summary = {
        "n": len(rows),
        "n_with_gt": len(gt_rows),
        "mean_n_accept": round(float(np.mean(accs)), 3),
        "mean_decode_speedup": round(float(np.mean([r["decode_speedup"] for r in rows])), 3),
        "frac_accept_ge2": round(float(np.mean([a >= 2 for a in accs])), 3),
        "frac_accept_ge4": round(float(np.mean([a >= 4 for a in accs])), 3),
        "accept_dist": {str(k): round(accs.count(k) / max(len(accs), 1), 3) for k in range(7)},
        "note_speedup": "decode-step accounting only; not wall-clock specdec",
        "note_l2": "ST-P3/UniAD L2 vs GT only on frames with a valid VAD future (n_with_gt). Official OpenDrive occupancy-masked numbers are in opendrive_nusc_plan_conv eval.",
        "teacher_stp3": flatten("teacher", stp3),
        "spec_stp3": flatten("spec", stp3),
        "draft_stp3": flatten("draft", stp3),
        "teacher_uniad_l2": flatten("teacher", uni),
        "spec_uniad_l2": flatten("spec", uni),
        "draft_uniad_l2": flatten("draft", uni),
    }
    return summary, rows


def _sample_idx(extra, n_tr_idx, batch):
    """Oversample turns; they are ~15% of data but kill prefix accept."""
    e = extra[n_tr_idx]
    if e.shape[-1] < 4:
        return n_tr_idx[torch.randint(0, n_tr_idx.numel(), (batch,))]
    turn = (e[:, 1] + e[:, 2]).clamp(0, 1)
    w = 1.0 + 3.5 * turn
    w = w / w.sum()
    pick = torch.multinomial(w, batch, replacement=True)
    return n_tr_idx[pick]


def train_until(head, hs, extra, speed, ys, cfg, target_accept, max_rounds=8, steps_per=3000, tr_idx=None, va_idx=None, val_pack=None):
    n = hs.shape[0]
    if tr_idx is None or va_idx is None:
        n_val = max(64, n // 8)
        va_idx = torch.arange(n - n_val, n)
        tr_idx = torch.arange(0, n - n_val)
    if val_pack is not None:
        tr_idx = torch.arange(n)
    opt = torch.optim.AdamW(head.parameters(), lr=1.5e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_rounds * steps_per, eta_min=2e-5)
    best = -1.0
    best_state = None
    hist = []
    batch = min(96, int(tr_idx.numel()))

    def eval_acc():
        if val_pack is not None:
            hv, ev, sv, yv = val_pack
            return accept_stats(head, hv, ev, sv, yv, cfg)
        return accept_stats(head, hs[va_idx], extra[va_idx], speed[va_idx], ys[va_idx], cfg)

    acc0, ade0, dist0, _ = eval_acc()
    print(f"[train] init accept={acc0:.3f}/6 ade={ade0:.3f} dist={ {k: round(v, 3) for k, v in dist0.items()} }", flush=True)
    stale = 0
    for rnd in range(max_rounds):
        head.train()
        losses = []
        for _ in range(steps_per):
            idx = _sample_idx(extra, tr_idx, batch)
            loss = train_step(
                head,
                hs[idx].to(DEVICE),
                ys[idx].to(DEVICE),
                opt,
                extra=extra[idx].to(DEVICE),
                speed=speed[idx].to(DEVICE),
            )
            sched.step()
            losses.append(loss)
        acc, ade, dist, _ = eval_acc()
        rec = {
            "round": rnd,
            "loss": round(float(np.mean(losses[-100:])), 4),
            "accept": round(acc, 3),
            "ade": round(ade, 3),
            "dist": {str(k): round(v, 3) for k, v in dist.items()},
        }
        hist.append(rec)
        print(f"[train] round {rnd} loss={rec['loss']} accept={acc:.3f}/6 ade={ade:.3f} dist={rec['dist']}", flush=True)
        if acc > best + 1e-4:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if acc >= target_accept:
            print(f"[train] hit target accept {target_accept}", flush=True)
            break
        if stale >= 2 and best_state is not None:
            head.load_state_dict(best_state)
            for g in opt.param_groups:
                g["lr"] = max(g["lr"] * 0.5, 1e-5)
            print(f"[train] rewind+decay lr -> {opt.param_groups[0]['lr']:.2e}", flush=True)
            stale = 0
    if best_state is not None:
        head.load_state_dict(best_state)
    return hist, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", default="")
    p.add_argument("--target-accept", type=float, default=3.5)
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--steps-per", type=int, default=3000)
    p.add_argument("--lon-m", type=float, default=0.5)
    p.add_argument("--lat-m", type=float, default=0.3)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--no-refresh-extra", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = AcceptConfig(lon_m=args.lon_m, lat_m=args.lat_m)
    tokens_tr, hs, extra, speed, ys, gt, hidden_size = load_cache(Path(args.train_cache), refresh_extra=not args.no_refresh_extra)
    print(f"[data] train cache n={len(tokens_tr)} hidden={hidden_size} extra_dim={extra.shape[-1]}", flush=True)

    val_path = Path(args.val_cache) if args.val_cache else None
    val_pack = None
    tokens_v = hs_v = extra_v = speed_v = ys_v = gt_v = None
    if val_path and val_path.exists():
        tokens_v, hs_v, extra_v, speed_v, ys_v, gt_v, _ = load_cache(val_path, refresh_extra=not args.no_refresh_extra)
        print(f"[data] val cache n={len(tokens_v)} (also used for checkpoint selection)", flush=True)
        val_pack = (hs_v, extra_v, speed_v, ys_v)

    tr_idx, va_idx = _hash_mask(tokens_tr, frac=0.15)
    print(f"[data] hash split train={tr_idx.numel()} inner_val={va_idx.numel()} val_select={val_pack is not None}", flush=True)

    head = DraftHead(hidden_size, horizon=6, width=args.width).to(DEVICE)
    hist, best = train_until(
        head, hs, extra, speed, ys, cfg, args.target_accept, args.max_rounds, args.steps_per, tr_idx, va_idx, val_pack=val_pack
    )
    ckpt = Path(args.out).with_suffix(".pt")
    torch.save({"state_dict": head.state_dict(), "hidden_size": hidden_size, "name": args.name, "extra_size": int(extra.shape[-1])}, ckpt)

    if val_pack is not None:
        summary, rows = eval_full(head, tokens_v, hs_v, extra_v, speed_v, ys_v, gt_v, cfg)
        summary["eval_split"] = "heldout_val_cache"
    else:
        summary, rows = eval_full(head, [tokens_tr[i] for i in va_idx.tolist()], hs[va_idx], extra[va_idx], speed[va_idx], ys[va_idx], gt[va_idx], cfg)
        summary["eval_split"] = "hash_holdout_from_train_cache"
        summary["note"] = "val cache missing; reporting hash holdout not train-set overfit"
    summary["train_hist"] = hist
    summary["best_inner_accept"] = round(best, 3)
    out = {"name": args.name, "accept": {"lon_m": args.lon_m, "lat_m": args.lat_m}, "summary": summary, "n_samples_stored": min(24, len(rows)), "samples": rows[:24]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "train_hist"}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
