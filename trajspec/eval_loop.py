"""Shared train/eval loop for waypoint specdec open-loop."""
from __future__ import annotations

import time

import numpy as np
import torch

from trajspec.accept import AcceptConfig, longest_prefix_accept
from trajspec.metrics import l2_horizon
from trajspec.draft_head import DraftHead
from trajspec.spec_loop import train_step


def mean_dict(rows, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = round(float(np.mean(vals)), 4) if vals else None
    return out


def l2_uniad_at(pred: np.ndarray, gt: np.ndarray) -> dict:
    """UniAD-style L2 at exactly 1s/2s/3s (2 Hz waypoints index 1, 3, 5)."""
    n = min(len(pred), len(gt), 6)
    if n == 0:
        return {"l2_at_1s": None, "l2_at_2s": None, "l2_at_3s": None}
    d = np.linalg.norm(pred[:n, :2] - gt[:n, :2], axis=-1)

    def at(i):
        return float(d[i]) if i < n else None

    return {"l2_at_1s": at(1), "l2_at_2s": at(3), "l2_at_3s": at(5)}


def spec_from_draft(draft, teacher, cfg: AcceptConfig, horizon: int = 6, seam: str = "blend"):
    d = torch.as_tensor(draft[:, :2], dtype=torch.float32)
    t = torch.as_tensor(teacher[:, :2], dtype=torch.float32)
    k = min(d.shape[0], t.shape[0], horizon)
    n = longest_prefix_accept(d[:k], t[:k], cfg)
    if n >= k:
        spec = d[:k]
        steps = 1
    else:
        from trajspec.sd_baselines import seam_blend

        if str(seam) == "blend":
            spec = seam_blend(d[:k], t[:k], n, horizon=k)
        else:
            parts = []
            if n > 0:
                parts.append(d[:n])
            parts.append(t[n : n + 1])
            if n + 1 < t.shape[0]:
                parts.append(t[n + 1 : k])
            spec = torch.cat(parts, dim=0)[:k]
        steps = 1 + max(k - n - 1, 0)
    return spec.numpy(), n, steps


def collect_and_train(runner, samples, steps=800, supervise="teacher", horizon=6):
    cache_h, cache_y = [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        h = runner.prefill_hidden(s)
        if supervise == "teacher":
            y = torch.as_tensor(runner.predict(s)[:horizon], dtype=torch.float32)
        else:
            y = torch.as_tensor(s["gt"][:horizon], dtype=torch.float32)
        if y.shape[0] < horizon:
            pad = y[-1:].repeat(horizon - y.shape[0], 1) if y.shape[0] else torch.zeros(horizon, 2)
            y = torch.cat([y, pad], dim=0)[:horizon]
        cache_h.append(h)
        cache_y.append(y[:horizon])
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  cache {i+1}/{len(samples)}  elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"[cache] {len(cache_h)} samples in {time.time()-t0:.1f}s", flush=True)
    hs = torch.stack(cache_h, dim=0)
    ys = torch.stack(cache_y, dim=0)
    head = DraftHead(int(hs.shape[-1]), horizon=horizon).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3)
    losses = []
    for i in range(steps):
        idx = torch.randint(0, hs.shape[0], (min(32, hs.shape[0]),))
        loss = train_step(head, hs[idx].cuda(), ys[idx].cuda(), opt)
        losses.append(loss)
    head.eval()
    return head, losses


def eval_split(name, runner, head, samples, accept_cfg, horizon=6):
    rows = []
    for i, s in enumerate(samples):
        t_gen0 = time.perf_counter()
        teacher = runner.predict(s)
        t_gen = (time.perf_counter() - t_gen0) * 1000
        t_h0 = time.perf_counter()
        hidden = runner.prefill_hidden(s)
        with torch.no_grad():
            draft = head(hidden.cuda())[0, :horizon].cpu().numpy()
        t_draft = (time.perf_counter() - t_h0) * 1000
        spec, n_acc, spec_steps = spec_from_draft(draft, teacher, accept_cfg, horizon=horizon)
        gt = s["gt"]
        rec = {
            "idx": s.get("idx"),
            "token": s.get("token"),
            "folder": s.get("folder"),
            "n_accept": int(n_acc),
            "spec_decode_steps": int(spec_steps),
            "decode_speedup": round(horizon / max(spec_steps, 1e-6), 3),
            "gen_ms": round(t_gen, 1),
            "prefill_draft_ms": round(t_draft, 1),
            "teacher": {**l2_horizon(teacher, gt), **l2_uniad_at(teacher, gt)},
            "spec": {**l2_horizon(spec, gt), **l2_uniad_at(spec, gt)},
            "draft": {**l2_horizon(draft, gt), **l2_uniad_at(draft, gt)},
        }
        rows.append(rec)
        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"  [{name}] {i+1}/{len(samples)} accept={n_acc} "
                f"teacherADE={rec['teacher']['ade']:.3f} specADE={rec['spec']['ade']:.3f} "
                f"gen={t_gen:.0f}ms",
                flush=True,
            )

    def flatten(tag, keys):
        return mean_dict([{k: r[tag][k] for k in keys} for r in rows], keys)

    stp3_keys = ["l2_1s", "l2_2s", "l2_3s", "ade"]
    uniad_keys = ["l2_at_1s", "l2_at_2s", "l2_at_3s"]
    summary = {
        "n": len(rows),
        "mean_n_accept": round(float(np.mean([r["n_accept"] for r in rows])), 3),
        "mean_decode_speedup": round(float(np.mean([r["decode_speedup"] for r in rows])), 3),
        "mean_gen_ms": round(float(np.mean([r["gen_ms"] for r in rows])), 1),
        "mean_prefill_draft_ms": round(float(np.mean([r["prefill_draft_ms"] for r in rows])), 1),
        "teacher_stp3": flatten("teacher", stp3_keys),
        "spec_stp3": flatten("spec", stp3_keys),
        "draft_stp3": flatten("draft", stp3_keys),
        "teacher_uniad_l2": flatten("teacher", uniad_keys),
        "spec_uniad_l2": flatten("spec", uniad_keys),
        "draft_uniad_l2": flatten("draft", uniad_keys),
        "teacher_openloop": flatten("teacher", stp3_keys),
        "spec_openloop": flatten("spec", stp3_keys),
        "draft_openloop": flatten("draft", stp3_keys),
    }
    return summary, rows
