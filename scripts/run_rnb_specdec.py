#!/usr/bin/env python3
"""Train RnB draft + score Teacher/Draft/Spec on official OpenDrive occupancy GT."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path("/home/xinhchen2/hhb/TrajSpecDec-paper")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PAPER / "scripts"))

from trajspec.accept import AcceptConfig
from trajspec.draft_head import DraftHead
from trajspec.eval_loop import spec_from_draft
from trajspec.serialize import ego_to_od_xy

from run_full_specdec import _hash_mask, load_cache, train_until, eval_full

OD = Path("/home/xinhchen2/hhb/OpenDriveVLA")
sys.path.insert(0, str(OD / "drivevla"))
from eval_share.metric import PlanningMetric  # noqa: E402

GT_DIR = OD / "drivevla/eval_share/gt"
RESULTS = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ego_to_od(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    return np.stack([-xy[:, 1], xy[:, 0]], axis=-1)


def load_gt_pack() -> dict:
    def _occ(name: str, extra_flip_hw: bool) -> dict:
        occ = pickle.load(open(GT_DIR / name, "rb"))
        out = {}
        for token, occupancy in occ.items():
            if not isinstance(occupancy, torch.Tensor):
                occupancy = torch.tensor(occupancy)
            occupancy = torch.flip(occupancy, [-1])
            if extra_flip_hw:
                occupancy = torch.flip(occupancy, [-2])
            out[token] = occupancy
        return out

    print("[gt] loading occupancy", flush=True)
    return {
        "gt": pickle.load(open(GT_DIR / "gt_traj.pkl", "rb")),
        "mask": pickle.load(open(GT_DIR / "gt_traj_mask.pkl", "rb")),
        "occ": _occ("planing_gt_segmentation_val", extra_flip_hw=False),
    }


def score_preds(pred_od: dict, subset: set[str] | None = None) -> dict:
    pack = load_gt_pack()
    gt, mask, occ = pack["gt"], pack["mask"], pack["occ"]
    metric = PlanningMetric(6)
    n = skipped = 0
    ts = 6
    for token, g in tqdm(gt.items(), ncols=80, leave=False, desc="score"):
        if subset is not None and token not in subset:
            continue
        if token not in pred_od or token not in occ:
            skipped += 1
            continue
        gt_trajectory = torch.as_tensor(g, dtype=torch.float32)
        gt_traj_mask = torch.as_tensor(mask[token], dtype=torch.float32)
        output = torch.as_tensor(pred_od[token], dtype=torch.float32)
        if output.ndim == 2:
            output = output.unsqueeze(0)
        output = output.reshape(gt_traj_mask.shape) if output.numel() == gt_traj_mask.numel() else output.view(1, -1, 2)
        occupancy = occ[token]
        if occupancy.ndim == 3:
            occupancy = occupancy.unsqueeze(0)
        if output.shape[1] % 2:
            output = output[:, 1:]
        if occupancy.shape[1] % 2:
            occupancy = occupancy[:, 1:]
        if gt_trajectory.ndim == 2:
            gt_trajectory = gt_trajectory.unsqueeze(0)
        if gt_traj_mask.ndim == 2:
            gt_traj_mask = gt_traj_mask.unsqueeze(0)
        if gt_trajectory.shape[1] % 2:
            gt_trajectory = gt_trajectory[:, 1:]
        if gt_traj_mask.shape[1] % 2:
            gt_traj_mask = gt_traj_mask[:, 1:]
        metric(
            output[:, :ts],
            gt_trajectory[:, :ts],
            occupancy[:, :ts],
            token,
            gt_traj_mask[:, :ts],
        )
        n += 1
    scores = metric.compute()
    l2 = scores["L2"].detach().cpu().numpy()
    col = scores["obj_box_col"].detach().cpu().numpy()
    uniad_l2 = [float(l2[1]), float(l2[3]), float(l2[5])]
    uniad_col = [float(col[1] * 100), float(col[3] * 100), float(col[5] * 100)]
    stp_l2 = [float(l2[:2].mean()), float(l2[:4].mean()), float(l2[:6].mean())]
    stp_col = [float(col[:2].mean() * 100), float(col[:4].mean() * 100), float(col[:6].mean() * 100)]
    return {
        "scored": n,
        "skipped_missing": skipped,
        "uniad": {
            "l2": [round(x, 4) for x in uniad_l2],
            "l2_avg": round(sum(uniad_l2) / 3, 4),
            "col": [round(x, 4) for x in uniad_col],
            "col_avg": round(sum(uniad_col) / 3, 4),
        },
        "stp3": {
            "l2": [round(x, 4) for x in stp_l2],
            "l2_avg": round(sum(stp_l2) / 3, 4),
            "col": [round(x, 4) for x in stp_col],
            "col_avg": round(sum(stp_col) / 3, 4),
        },
    }


def dump_from_cache(cache_path: Path, ckpt: Path):
    tokens, hs, extra, speed, ys, gt, _ = load_cache(cache_path, refresh_extra=True)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    head = DraftHead(int(blob["hidden_size"]), horizon=6, extra_size=int(blob.get("extra_size", 16)))
    head.load_state_dict(blob["state_dict"])
    head.to(DEVICE).eval()
    cfg = AcceptConfig(lon_m=0.50, lat_m=0.30)
    teacher_od, spec_od, draft_od = {}, {}, {}
    accs = []
    with torch.no_grad():
        for i in tqdm(range(hs.shape[0]), ncols=80, desc="dump"):
            draft = (
                head(
                    hs[i : i + 1].to(DEVICE),
                    extra=extra[i : i + 1].to(DEVICE),
                    speed=speed[i : i + 1].to(DEVICE),
                )[0]
                .cpu()
                .numpy()
            )
            teacher = ys[i].numpy()
            spec, n_acc, _ = spec_from_draft(draft, teacher, cfg, horizon=6)
            tok = tokens[i]
            teacher_od[tok] = ego_to_od(teacher)
            spec_od[tok] = ego_to_od(spec)
            draft_od[tok] = ego_to_od(draft)
            accs.append(int(n_acc))
    accept = {
        "n": len(tokens),
        "mean_n_accept": round(float(np.mean(accs)), 3),
        "frac_accept_ge2": round(float(np.mean([a >= 2 for a in accs])), 3),
        "frac_accept_ge4": round(float(np.mean([a >= 4 for a in accs])), 3),
        "frac_accept_6": round(float(np.mean([a >= 6 for a in accs])), 3),
        "accept_dist": {str(k): round(accs.count(k) / max(len(accs), 1), 3) for k in range(7)},
    }
    return tokens, teacher_od, spec_od, draft_od, accept


def round_report(rep: dict) -> dict:
    return {
        "n": rep["scored"],
        "uniad_l2": rep["uniad"]["l2"],
        "uniad_l2_avg": rep["uniad"]["l2_avg"],
        "uniad_col": rep["uniad"]["col"],
        "uniad_col_avg": rep["uniad"]["col_avg"],
        "stp3_l2": rep["stp3"]["l2"],
        "stp3_l2_avg": rep["stp3"]["l2_avg"],
        "stp3_col": rep["stp3"]["col"],
        "stp3_col_avg": rep["stp3"]["col_avg"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(RESULTS / "cache_rnb_nusc_val.pt"))
    p.add_argument("--ckpt", default=str(RESULTS / "specdec_rnb_heldout.pt"))
    p.add_argument("--out", default=str(RESULTS / "rnb_official_compare.json"))
    p.add_argument("--target-accept", type=float, default=3.5)
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--steps-per", type=int, default=3000)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    cache = Path(args.cache)
    ckpt = Path(args.ckpt)
    cfg = AcceptConfig(lon_m=0.50, lat_m=0.30)

    tokens, hs, extra, speed, ys, gt, hidden_size = load_cache(cache, refresh_extra=True)
    print(f"[data] n={len(tokens)} hidden={hidden_size}", flush=True)
    tr_idx, va_idx = _hash_mask(tokens, frac=0.15)
    print(f"[data] hash train={tr_idx.numel()} holdout={va_idx.numel()}", flush=True)

    if not args.skip_train:
        head = DraftHead(hidden_size, horizon=6, width=512).to(DEVICE)
        hist, best = train_until(
            head, hs, extra, speed, ys, cfg, args.target_accept, args.max_rounds, args.steps_per, tr_idx, va_idx
        )
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": head.state_dict(),
                "hidden_size": hidden_size,
                "name": "rnb_waypoints",
                "extra_size": int(extra.shape[-1]),
                "train_hist": hist,
                "best_inner_accept": best,
            },
            ckpt,
        )
        print(f"[write] {ckpt} best_accept={best:.3f}", flush=True)
        hold_sum, _ = eval_full(
            head,
            [tokens[i] for i in va_idx.tolist()],
            hs[va_idx],
            extra[va_idx],
            speed[va_idx],
            ys[va_idx],
            gt[va_idx],
            cfg,
        )
        print("[holdout vs VAD GT]", json.dumps({k: hold_sum[k] for k in hold_sum if "hist" not in k}, indent=2), flush=True)
    else:
        hist, best = [], None
        hold_sum = {}

    tokens_d, teacher, spec, draft, accept_full = dump_from_cache(cache, ckpt)
    holdout = {tokens_d[i] for i in va_idx.tolist()}
    full = {
        "teacher": score_preds(teacher),
        "spec": score_preds(spec),
        "draft": score_preds(draft),
    }
    hold = {
        "teacher": score_preds(teacher, subset=holdout),
        "spec": score_preds(spec, subset=holdout),
        "draft": score_preds(draft, subset=holdout),
    }
    out = {
        "model": "stanfordasl/nuscenes-waypoints-model (Qwen3-VL-4B, traj-only)",
        "protocol": "same as OpenDriveVLA: hash 15% holdout draft train, official UniAD vehicle occupancy GT",
        "accept_full_val": accept_full,
        "holdout_n": len(holdout),
        "holdout_inner": hold_sum,
        "best_train_accept": best,
        "full_val": {k: round_report(v) for k, v in full.items()},
        "hash_holdout": {k: round_report(v) for k, v in hold.items()},
        "opendrive_reference": {
            "teacher_stp3_l2_avg": 0.35,
            "draft_stp3_l2_avg": 0.72,
            "spec_stp3_l2_avg": 0.37,
            "holdout_mean_n_accept": 4.009,
        },
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({"full_val": out["full_val"], "hash_holdout": out["hash_holdout"], "accept": accept_full}, indent=2), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
