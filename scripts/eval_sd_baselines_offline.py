#!/usr/bin/env python3
"""Holdout quality for SD accept policies. Same draft head, no extra teacher forwards.

OpenDriveVLA uses Qwen BPE of the serialized [(x,y),...] string (proxy for
token-exact / scaffold without live logits); R&B-EnCoRe shares that tokenization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.accept import AcceptConfig
from trajspec.metrics import l2_horizon
from trajspec.draft_head import DraftHead
from trajspec.eval_loop import l2_uniad_at, mean_dict
from trajspec.sd_baselines import (
    GEOMETRIC,
    KERV_LITE,
    SCAFFOLD,
    SPECVLA_ID,
    SPECVLA_POSE,
    TOKEN_EXACT,
    commit_spec,
    hash_holdout_idx,
    n_accept_opendrive,
)
from trajspec.serialize import TRAJ_END, TRAJ_START, format_opendrive_answer
from trajspec.nusc_data import extras_for_tokens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pad_stack(tensors, min_d=1):
    ts = [t.float().view(-1) for t in tensors]
    d = max([min_d] + [int(t.numel()) for t in ts])
    return torch.stack([torch.nn.functional.pad(t, (0, d - t.numel())) for t in ts]).float()


def load_split(cache: Path, frac=0.15, seed=17):
    blob = torch.load(cache, map_location="cpu", weights_only=False)
    rows = blob["rows"]
    tokens = [r["token"] for r in rows]
    hs = torch.stack([r["hidden"] for r in rows]).float()
    ys = torch.stack([r["teacher"][:6] for r in rows]).float()
    extra = _pad_stack([r["extra"] for r in rows], min_d=16)
    speed = torch.stack([r["speed"].view(-1)[:1].float() for r in rows]).view(-1)
    gt = torch.stack(
        [torch.as_tensor(r["gt"] if r.get("gt") is not None else r["teacher"])[:6].float() for r in rows]
    )
    rebuilt = extras_for_tokens(tokens)
    hit = 0
    extras, speeds, gts = [], [], []
    for i, tok in enumerate(tokens):
        if tok in rebuilt:
            e, s, g = rebuilt[tok]
            extras.append(e.float().view(-1))
            speeds.append(s.view(-1)[:1].float())
            gts.append(g[:6].float() if g.abs().sum() > 0.05 else gt[i])
            hit += 1
        else:
            extras.append(extra[i].float().view(-1))
            speeds.append(speed[i : i + 1])
            gts.append(gt[i])
    extra = _pad_stack(extras, min_d=16)
    speed = torch.stack(speeds).float().view(-1)
    gt = torch.stack(gts).float()
    print(f"[data] extras rebuilt {hit}/{len(tokens)} dim={tuple(extra.shape)}", flush=True)
    hidden_size = int(blob.get("hidden_size", hs.shape[-1]))
    extra_size = int(blob.get("extra_size", extra.shape[-1]))
    idx = hash_holdout_idx(len(tokens), frac=frac, seed=seed)
    return {
        "tokens": [tokens[i] for i in idx.tolist()],
        "hs": hs[idx],
        "ys": ys[idx],
        "extra": extra[idx],
        "speed": speed[idx],
        "gt": gt[idx],
        "hidden_size": hidden_size,
        "extra_size": extra_size,
        "n_full": len(tokens),
    }


def load_head(ckpt: Path, hidden_size: int, extra_size: int, device=DEVICE):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    head = DraftHead(
        int(blob.get("hidden_size", hidden_size)),
        horizon=6,
        extra_size=int(blob.get("extra_size", extra_size)),
        width=512,
    ).to(device)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    return head


def summarize_method(rows, tag):
    gt_rows = [r for r in rows if r.get("has_gt")]
    accs = [r["n_accept"] for r in rows]
    keys = ["l2_1s", "l2_2s", "l2_3s", "ade"]
    return {
        "n": len(rows),
        "n_with_gt": len(gt_rows),
        "mean_n_accept": round(float(np.mean(accs)), 3) if accs else 0.0,
        "frac_accept_ge2": round(float(np.mean([a >= 2 for a in accs])), 3) if accs else 0.0,
        "frac_accept_ge4": round(float(np.mean([a >= 4 for a in accs])), 3) if accs else 0.0,
        "accept_dist": {str(k): round(accs.count(k) / max(len(accs), 1), 3) for k in range(7)},
        "ade_vs_teacher": round(float(np.mean([r["ade_vs_teacher"] for r in rows])), 4),
        "spec_stp3": mean_dict([{k: r["spec"][k] for k in keys} for r in gt_rows if r.get("spec")], keys),
        "teacher_stp3": mean_dict([{k: r["teacher"][k] for k in keys} for r in gt_rows if r.get("teacher")], keys),
        "tag": tag,
    }


@torch.no_grad()
def eval_opendrive(split, head, tokenizer, cfg):
    modes = [
        ("AR greedy", None),
        ("Spec-VLA (exact)", TOKEN_EXACT),
        ("Scaffold-SD", SCAFFOLD),
        ("Spec-VLA (pose-relax)", SPECVLA_POSE),
        ("KERV-lite", KERV_LITE),
        ("TrajSpecDec (ours)", GEOMETRIC),
    ]
    hs, extra, speed, ys, gt = split["hs"], split["extra"], split["speed"], split["ys"], split["gt"]
    drafts = []
    for i in range(hs.shape[0]):
        pred = head(
            hs[i : i + 1].to(DEVICE),
            extra=extra[i : i + 1].to(DEVICE),
            speed=speed[i : i + 1].to(DEVICE),
        )[0].cpu()
        drafts.append(pred)

    def _ids(xy):
        text = f"{TRAJ_START}{format_opendrive_answer(xy)}{TRAJ_END}"
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return ids

    out = {}
    for name, mode in modes:
        rows = []
        for i, draft in enumerate(drafts):
            teacher = ys[i]
            g = gt[i].numpy()
            has_gt = bool(np.linalg.norm(g) > 0.2)
            teacher_np = teacher.numpy()
            if mode is None:
                spec = teacher_np
                n_acc = 0
            else:
                d_ids = _ids(draft)
                t_ids = _ids(teacher)
                n_acc = n_accept_opendrive(mode, draft, teacher, d_ids, t_ids, tokenizer, cfg, horizon=6)
                spec_t, n_acc, _ = commit_spec(draft, teacher, n_acc, mode, float(speed[i]), horizon=6)
                spec = spec_t.numpy()
            rec = {
                "n_accept": int(n_acc),
                "has_gt": has_gt,
                "ade_vs_teacher": round(float(np.linalg.norm(spec[:6] - teacher_np[:6], axis=-1).mean()), 4),
                "teacher": {**l2_horizon(teacher_np, g), **l2_uniad_at(teacher_np, g)} if has_gt else {},
                "spec": {**l2_horizon(spec, g), **l2_uniad_at(spec, g)} if has_gt else {},
            }
            rows.append(rec)
        out[name] = summarize_method(rows, name)
        print(f"[OpenDrive] {name}: accept={out[name]['mean_n_accept']} ADE_T={out[name]['ade_vs_teacher']} "
              f"ADE_G={out[name]['spec_stp3'].get('ade')}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id-radius", type=int, default=16)
    p.add_argument("--out", default=str(ROOT / "results/sd_baselines_offline.json"))
    args = p.parse_args()
    cfg = AcceptConfig(lon_m=0.5, lat_m=0.3)

    od_split = load_split(ROOT / "results/cache_opendrive_nusc_val.pt")
    od_head = load_head(ROOT / "results/specdec_opendrive_heldout.pt", od_split["hidden_size"], od_split["extra_size"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/data2/hf_overlap_models/OpenDriveVLA-0.5B", trust_remote_code=True
    )
    print(f"[data] OpenDrive holdout n={od_split['hs'].shape[0]}", flush=True)
    opendrive = eval_opendrive(od_split, od_head, tokenizer, cfg)

    blob = {
        "protocol": {
            "split": "hash holdout 15% seed=17 (same as draft training)",
            "accept": {"lon_m": 0.5, "lat_m": 0.3, "id_radius": args.id_radius},
            "note": (
                "Offline proxy: teacher suffix stands in for serial AR after reject. "
                "OpenDrive token-exact / scaffold tokenize the official 2-decimal string "
                "(live eval uses teacher-force logits)."
            ),
        },
        "opendrive": opendrive,
    }
    Path(args.out).write_text(json.dumps(blob, ensure_ascii=False, indent=2))
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
