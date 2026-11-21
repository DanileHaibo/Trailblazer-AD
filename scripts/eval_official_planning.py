#!/usr/bin/env python3
"""Dump TrajSpecDec teacher/spec/draft on official nuScenes GT and score them.

OpenDriveVLA-0.5B official protocol: GPT-Driver/UniAD occupancy + traj mask,
UniAD (t=1/2/3s) and ST-P3 (cumulative) L2 / box-collision.

Every teacher is scored against the same occupancy GT after coordinate mapping
ego (+X forward, +Y left) into UniAD lidar (x right, y forward). Primary
Only the OpenDrive-style textual-waypoint teachers are covered here.
"""
from __future__ import annotations

import os

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OD = Path("/home/xinhchen2/hhb/OpenDriveVLA")
sys.path.insert(0, str(OD / "drivevla"))
from eval_share.metric import PlanningMetric  # noqa: E402

from trajspec.accept import AcceptConfig
from trajspec.draft_head import DraftHead
from trajspec.eval_loop import spec_from_draft

GT_DIR = OD / "drivevla/eval_share/gt"
RESULTS = ROOT / "results"

PAPER = {
    "opendrive_0.5b": {
        "dataset": "nuScenes val open-loop (6019 frames)",
        "stp3_l2": [0.15, 0.32, 0.57],
        "stp3_l2_avg": 0.35,
        "stp3_col": [0.01, 0.06, 0.20],
        "stp3_col_avg": 0.09,
        "uniad_l2": [0.21, 0.60, 1.22],
        "uniad_l2_avg": 0.68,
        "uniad_col": [0.00, 0.15, 0.63],
        "uniad_col_avg": 0.26,
        "source": "OpenDriveVLA paper Table I, OpenDriveVLA-0.5B",
    },
}


def ego_to_od(xy: np.ndarray) -> np.ndarray:
    """ego (+X forward, +Y left) -> UniAD/OpenDrive lidar (x right, y forward)."""
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    return np.stack([-xy[:, 1], xy[:, 0]], axis=-1)


def _hash_mask(tokens, frac=0.15, seed=17):
    n = len(tokens)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(64, int(round(n * frac)))
    return perm[n_val:], perm[:n_val]


_GT_CACHE: dict | None = None


def load_gt_pack() -> dict:
    global _GT_CACHE
    if _GT_CACHE is not None:
        return _GT_CACHE

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

    print("[gt] loading official occupancy + traj pickle", flush=True)
    _GT_CACHE = {
        "gt": pickle.load(open(GT_DIR / "gt_traj.pkl", "rb")),
        "mask": pickle.load(open(GT_DIR / "gt_traj_mask.pkl", "rb")),
        "occ_uniad": _occ("planing_gt_segmentation_val", extra_flip_hw=False),
        "occ_stp3": _occ("vad_gt_seg.pkl", extra_flip_hw=True),
    }
    print(f"[gt] n={len(_GT_CACHE['gt'])}", flush=True)
    return _GT_CACHE


def fmt_traj(xy: np.ndarray) -> str:
    pts = ", ".join(f"({x:.2f},{y:.2f})" for x, y in xy.reshape(-1, 2)[:6])
    return f"[{pts}]"


def score_preds(pred_od: dict[str, np.ndarray], only_vehicle: bool, subset: set[str] | None = None) -> dict:
    """Official OpenDriveVLA evaluation.py aggregation."""
    pack = load_gt_pack()
    gt, mask = pack["gt"], pack["mask"]
    occ = pack["occ_uniad"] if only_vehicle else pack["occ_stp3"]
    metric = PlanningMetric(6)
    n = 0
    skipped = 0
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
        output = output.reshape(gt_traj_mask.shape)
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
        "only_vehicle_occupancy": only_vehicle,
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
        "gt_collision_count": int(metric.gt_collision),
    }


def official_pair(pred_od, subset=None):
    return {
        "uniad_protocol": score_preds(pred_od, only_vehicle=True, subset=subset),
        "stp3_protocol": score_preds(pred_od, only_vehicle=False, subset=subset),
    }


def load_head(ckpt: Path, device: torch.device) -> DraftHead:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    head = DraftHead(int(blob["hidden_size"]), horizon=6, extra_size=int(blob.get("extra_size", 16)))
    head.load_state_dict(blob["state_dict"])
    head.to(device).eval()
    return head


def dump_from_cache(
    cache_path: Path,
    ckpt: Path,
    device: torch.device,
    refresh_extra: bool,
    to_od: bool,
):
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_full_specdec", ROOT / "scripts" / "run_full_specdec.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_full_specdec"] = mod
    spec.loader.exec_module(mod)
    load_cache = mod.load_cache

    tokens, hs, extra, speed, ys, gt, _ = load_cache(cache_path, refresh_extra=refresh_extra)
    head = load_head(ckpt, device)
    cfg = AcceptConfig(lon_m=0.50, lat_m=0.30)
    teacher_od, spec_od, draft_od = {}, {}, {}
    accs, steps = [], []
    with torch.no_grad():
        for i in tqdm(range(hs.shape[0]), ncols=80, desc=cache_path.name):
            draft = (
                head(
                    hs[i : i + 1].to(device),
                    extra=extra[i : i + 1].to(device),
                    speed=speed[i : i + 1].to(device),
                )[0]
                .cpu()
                .numpy()
            )
            teacher = ys[i].numpy()
            spec, n_acc, spec_steps = spec_from_draft(draft, teacher, cfg, horizon=6)
            tok = tokens[i]
            teacher_od[tok] = ego_to_od(teacher) if to_od else np.asarray(teacher, dtype=np.float32)
            spec_od[tok] = ego_to_od(spec) if to_od else np.asarray(spec, dtype=np.float32)
            draft_od[tok] = ego_to_od(draft) if to_od else np.asarray(draft, dtype=np.float32)
            accs.append(int(n_acc))
            steps.append(int(spec_steps))
    accept = {
        "n": len(tokens),
        "mean_n_accept": round(float(np.mean(accs)), 3),
        "mean_decode_speedup": round(float(np.mean([6 / max(s, 1e-6) for s in steps])), 3),
        "frac_accept_ge2": round(float(np.mean([a >= 2 for a in accs])), 3),
        "frac_accept_6": round(float(np.mean([a >= 6 for a in accs])), 3),
        "accept_dist": {str(k): round(accs.count(k) / max(len(accs), 1), 3) for k in range(7)},
    }
    return tokens, teacher_od, spec_od, draft_od, accept


def write_jsonl(path: Path, pred_od: dict):
    with path.open("w") as f:
        for tok, xy in pred_od.items():
            f.write(
                json.dumps(
                    {"id": f"{tok}_trajectory", "answer": [fmt_traj(xy)]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_pkl(path: Path, pred_od: dict):
    blob = {f"{tok}_trajectory": xy.reshape(1, 6, 2).astype(np.float32) for tok, xy in pred_od.items()}
    pickle.dump(blob, path.open("wb"))


def round_report(rep: dict) -> dict:
    """Paper-style table: UniAD occupancy for both aggregations (matches published OD numbers)."""
    uni = rep["uniad_protocol"]
    return {
        "n": uni["scored"],
        "uniad_l2": uni["uniad"]["l2"],
        "uniad_l2_avg": uni["uniad"]["l2_avg"],
        "uniad_col": uni["uniad"]["col"],
        "uniad_col_avg": uni["uniad"]["col_avg"],
        "stp3_l2": uni["stp3"]["l2"],
        "stp3_l2_avg": uni["stp3"]["l2_avg"],
        "stp3_col": uni["stp3"]["col"],
        "stp3_col_avg": uni["stp3"]["col_avg"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=RESULTS / "official_benchmark_compare.json")
    args = p.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)

    out = {
        "paper": PAPER,
        "method": {
            "accept": {"lon_m": 0.50, "lat_m": 0.30},
            "spec": "accepted draft prefix + teacher correction + teacher suffix",
            "coords": "OpenDrive cache is already UniAD (x right, y forward). R&B cache is ego (+X fwd, +Y left) and is mapped before scoring.",
        },
    }

    print("=== OpenDriveVLA-0.5B on official nuScenes val ===", flush=True)
    od_tokens, od_teacher, od_spec, od_draft, od_accept = dump_from_cache(
        RESULTS / "cache_opendrive_nusc_val.pt",
        RESULTS / "specdec_opendrive_heldout.pt",
        device,
        refresh_extra=True,
        to_od=False,
    )
    _, va_idx = _hash_mask(od_tokens)
    holdout = {od_tokens[i] for i in va_idx.tolist()}
    od_full = {
        "teacher": official_pair(od_teacher),
        "spec": official_pair(od_spec),
        "draft": official_pair(od_draft),
    }
    od_hold = {
        "teacher": official_pair(od_teacher, subset=holdout),
        "spec": official_pair(od_spec, subset=holdout),
        "draft": official_pair(od_draft, subset=holdout),
    }
    out["opendrive"] = {
        "accept_full_val": od_accept,
        "holdout_n": len(holdout),
        "full_val": {k: round_report(v) for k, v in od_full.items()},
        "hash_holdout": {k: round_report(v) for k, v in od_hold.items()},
    }

    print("=== R&B-EnCoRe-4B on official nuScenes occupancy GT ===", flush=True)
    rnb_cache = Path(os.environ.get("TRAJSPEC_RNB_CACHE", "/home/xinhchen2/hhb/TrajSpecDec/results/cache_rnb_nusc_val.pt"))
    rnb_ckpt = Path(os.environ.get("TRAJSPEC_RNB_DRAFT", "/home/xinhchen2/hhb/TrajSpecDec/results/specdec_rnb_full_retrain.pt"))
    if rnb_cache.exists() and rnb_ckpt.exists():
        rnb_tokens, rnb_teacher, rnb_spec, rnb_draft, rnb_accept = dump_from_cache(
            rnb_cache,
            rnb_ckpt,
            device,
            refresh_extra=True,
            to_od=True,
        )
        _, rnb_va = _hash_mask(rnb_tokens)
        rnb_holdout = {rnb_tokens[i] for i in rnb_va.tolist()}
        rnb_full = {
            "teacher": official_pair(rnb_teacher),
            "spec": official_pair(rnb_spec),
            "draft": official_pair(rnb_draft),
        }
        rnb_hold = {
            "teacher": official_pair(rnb_teacher, subset=rnb_holdout),
            "spec": official_pair(rnb_spec, subset=rnb_holdout),
            "draft": official_pair(rnb_draft, subset=rnb_holdout),
        }
        out["rnb"] = {
            "accept_full_val": rnb_accept,
            "n_cached": len(rnb_tokens),
            "holdout_n": len(rnb_holdout),
            "full_val": {k: round_report(v) for k, v in rnb_full.items()},
            "hash_holdout": {k: round_report(v) for k, v in rnb_hold.items()},
            "note": "Same protocol as OpenDrive: hash 15% holdout draft train, official occupancy GT, ego->UniAD coords.",
        }

    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(
        {
            "opendrive_full": out.get("opendrive", {}).get("full_val"),
            "rnb_full": out.get("rnb", {}).get("full_val"),
        },
        indent=2,
    ), flush=True)
    print("[write]", args.out, flush=True)


if __name__ == "__main__":
    main()
