"""nuScenes open-loop packing from VAD temporal infos (2 Hz, 6 steps, 3 s)."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

from trajspec.metrics import l2_horizon  # noqa: F401  re-export

# Override with TRAJSPEC_NUSC_ROOT to point at a local nuScenes copy.
NUSC_ROOT = Path(os.environ.get("TRAJSPEC_NUSC_ROOT", "/home/xinhchen2/hhb/Bench2DriveZoo/data/nuscenes"))
VAL_PKL = NUSC_ROOT / "vad_nuscenes_infos_temporal_val.pkl"
TRAIN_PKL = NUSC_ROOT / "vad_nuscenes_infos_temporal_train.pkl"


def load_infos(path: str | Path) -> list[dict]:
    with open(path, "rb") as f:
        blob = pickle.load(f)
    if isinstance(blob, dict) and "infos" in blob:
        return blob["infos"]
    return blob


def lidar_to_ego_xy(xy: np.ndarray) -> np.ndarray:
    """VAD nuScenes lidar after ego->lidar: +Y along drive, +X right (cmd x>=2 = right).

    Map to nuScenes ego convention (OpenDrive / R&B): +X forward, +Y left.
    """
    xy = np.asarray(xy, dtype=np.float32)
    out = np.empty_like(xy)
    out[..., 0] = xy[..., 1]
    out[..., 1] = -xy[..., 0]
    return out


def _cam_abs(rel: str) -> str:
    rel = str(rel).replace("./data/nuscenes/", "").lstrip("/")
    return str(NUSC_ROOT / rel)


def token_index(infos: list[dict]) -> dict[str, int]:
    return {info["token"]: i for i, info in enumerate(infos)}


def hist_indices(infos: list[dict], idx: int, tok2i: dict[str, int], n: int = 4) -> list[int] | None:
    chain = []
    cur = idx
    for _ in range(n):
        chain.append(cur)
        prev = infos[cur].get("prev") or ""
        if not prev or prev not in tok2i:
            break
        nxt = tok2i[prev]
        if infos[nxt]["scene_token"] != infos[idx]["scene_token"]:
            break
        cur = nxt
    if len(chain) < n:
        return None
    return list(reversed(chain))


def hist_abs_from_offsets(his_off: np.ndarray, n: int = 4) -> np.ndarray:
    """Offsets of past steps → absolute positions (current at origin), oldest first."""
    pos = np.zeros(2, dtype=np.float32)
    pts = []
    for off in reversed(np.asarray(his_off, dtype=np.float32)):
        pos = pos - off[:2]
        pts.append(pos.copy())
    pts = list(reversed(pts))
    while len(pts) < n:
        pts = [pts[0].copy() if pts else np.zeros(2, dtype=np.float32)] + pts
    return np.stack(pts[-n:], axis=0)


def pack_sample(infos: list[dict], idx: int, tok2i: dict[str, int]) -> dict | None:
    cur = infos[idx]
    if not cur.get("fut_valid_flag", True):
        return None
    fut_off = np.asarray(cur["gt_ego_fut_trajs"], dtype=np.float32)
    fut_mask = np.asarray(cur["gt_ego_fut_masks"], dtype=np.float32)
    if fut_off.shape[0] < 6 or fut_mask[:6].sum() < 6:
        return None
    hist_idx = hist_indices(infos, idx, tok2i, n=4)
    if hist_idx is None:
        return None
    images = {"front_camera": [], "front_left_camera": [], "front_right_camera": []}
    for j in hist_idx:
        cams = infos[j]["cams"]
        images["front_camera"].append(_cam_abs(cams["CAM_FRONT"]["data_path"]))
        images["front_left_camera"].append(_cam_abs(cams["CAM_FRONT_LEFT"]["data_path"]))
        images["front_right_camera"].append(_cam_abs(cams["CAM_FRONT_RIGHT"]["data_path"]))
        if not Path(images["front_camera"][-1]).exists():
            return None
    gt = lidar_to_ego_xy(np.cumsum(fut_off[:6], axis=0))
    his_off = np.asarray(cur["gt_ego_his_trajs"], dtype=np.float32)
    hist = lidar_to_ego_xy(hist_abs_from_offsets(his_off, n=4))
    cmd = np.asarray(cur["gt_ego_fut_cmd"], dtype=np.float32).reshape(-1)
    if cmd[0] > 0:
        mission, driving = "turn right", "turn right"
    elif cmd[1] > 0:
        mission, driving = "turn left", "turn left"
    else:
        mission, driving = "keep forward", "go straight"
    lcf = np.asarray(cur["gt_ego_lcf_feat"], dtype=np.float32).reshape(-1)
    # OpenDrive prompt uses lcf[0:2]*0.5 as (vx,vy); v0 at [7] is longitudinal m/s.
    v0 = float(lcf[7]) if lcf.size > 7 else float(np.linalg.norm(gt[0]))
    return {
        "images": images,
        "sensor_data_path": "",
        "vehicle_velocity": [v0, 0.0],
        "vehicle_acceleration": [0.0, 0.0],
        "driving_command": driving,
        "mission_goal": mission,
        "hist": hist.astype(np.float32),
        "lcf": lcf,
        "gt": gt.astype(np.float32),
        "gt_mask": fut_mask[:6].astype(np.float32),
        "cmd": cmd.tolist(),
        "idx": idx,
        "token": cur["token"],
        "folder": cur.get("scene_token", ""),
        "frame_idx": cur.get("frame_idx", 0),
    }


def hist_indices_pad(infos: list[dict], idx: int, tok2i: dict[str, int], n: int = 4) -> list[int]:
    """Walk back as far as the scene allows, then repeat the oldest frame to length n."""
    chain = []
    cur = idx
    for _ in range(n):
        chain.append(cur)
        prev = infos[cur].get("prev") or ""
        if not prev or prev not in tok2i:
            break
        nxt = tok2i[prev]
        if infos[nxt]["scene_token"] != infos[idx]["scene_token"]:
            break
        cur = nxt
    chain = list(reversed(chain))
    while len(chain) < n:
        chain = [chain[0]] + chain
    return chain[-n:]


def pack_sample_full(infos: list[dict], idx: int, tok2i: dict[str, int]) -> dict | None:
    """Cover every nuScenes val frame (OpenDrive 6019). Pad short history by repeating."""
    s = pack_sample(infos, idx, tok2i)
    if s is not None:
        return s
    cur = infos[idx]
    hist_idx = hist_indices_pad(infos, idx, tok2i, n=4)
    images = {"front_camera": [], "front_left_camera": [], "front_right_camera": []}
    for j in hist_idx:
        cams = infos[j].get("cams") or {}
        front = cams.get("CAM_FRONT") or {}
        path = _cam_abs(front.get("data_path", ""))
        if not Path(path).exists():
            return None
        images["front_camera"].append(path)
        images["front_left_camera"].append(_cam_abs((cams.get("CAM_FRONT_LEFT") or front).get("data_path", "")))
        images["front_right_camera"].append(_cam_abs((cams.get("CAM_FRONT_RIGHT") or front).get("data_path", "")))
    fut_off = np.asarray(cur.get("gt_ego_fut_trajs", np.zeros((6, 2))), dtype=np.float32).reshape(-1, 2)
    if fut_off.shape[0] < 6:
        fut_off = np.vstack([fut_off, np.repeat(fut_off[-1:], 6 - fut_off.shape[0], axis=0) if fut_off.shape[0] else np.zeros((6, 2), np.float32)])
    fut_mask = np.asarray(cur.get("gt_ego_fut_masks", np.ones(6)), dtype=np.float32).reshape(-1)
    if fut_mask.size < 6:
        fut_mask = np.pad(fut_mask, (0, 6 - fut_mask.size))
    gt = lidar_to_ego_xy(np.cumsum(fut_off[:6], axis=0))
    his_off = np.asarray(cur.get("gt_ego_his_trajs", np.zeros((4, 2))), dtype=np.float32)
    hist = lidar_to_ego_xy(hist_abs_from_offsets(his_off, n=4)) if his_off.size >= 2 else np.zeros((4, 2), np.float32)
    cmd = np.asarray(cur.get("gt_ego_fut_cmd", [0, 0, 1]), dtype=np.float32).reshape(-1)
    if cmd.size < 3:
        cmd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if cmd[0] > 0:
        mission, driving = "turn right", "turn right"
    elif cmd[1] > 0:
        mission, driving = "turn left", "turn left"
    else:
        mission, driving = "keep forward", "go straight"
    lcf = np.asarray(cur.get("gt_ego_lcf_feat", np.zeros(9)), dtype=np.float32).reshape(-1)
    v0 = float(lcf[7]) if lcf.size > 7 else 0.0
    return {
        "images": images,
        "sensor_data_path": "",
        "vehicle_velocity": [v0, 0.0],
        "vehicle_acceleration": [0.0, 0.0],
        "driving_command": driving,
        "mission_goal": mission,
        "hist": hist.astype(np.float32),
        "lcf": lcf,
        "gt": gt.astype(np.float32),
        "gt_mask": fut_mask[:6].astype(np.float32),
        "cmd": cmd.tolist(),
        "idx": idx,
        "token": cur["token"],
        "folder": cur.get("scene_token", ""),
        "frame_idx": cur.get("frame_idx", 0),
    }


def pack_sample_rnb(infos: list[dict], idx: int, tok2i: dict[str, int]) -> dict | None:
    """RnB only needs the current front image; keep all 6019 val frames when possible."""
    cur = infos[idx]
    cam = cur.get("cams", {}).get("CAM_FRONT", {})
    path = _cam_abs(cam.get("data_path", ""))
    if not Path(path).exists():
        return None
    s = pack_sample(infos, idx, tok2i)
    if s is not None:
        return s
    fut_off = np.asarray(cur.get("gt_ego_fut_trajs", np.zeros((6, 2))), dtype=np.float32).reshape(-1, 2)
    if fut_off.shape[0] < 6:
        fut_off = np.vstack([fut_off, np.zeros((6 - fut_off.shape[0], 2), np.float32)])
    fut_mask = np.asarray(cur.get("gt_ego_fut_masks", np.ones(6)), dtype=np.float32).reshape(-1)
    if fut_mask.size < 6:
        fut_mask = np.pad(fut_mask, (0, 6 - fut_mask.size))
    gt = lidar_to_ego_xy(np.cumsum(fut_off[:6], axis=0))
    his_off = np.asarray(cur.get("gt_ego_his_trajs", np.zeros((4, 2))), dtype=np.float32)
    hist = lidar_to_ego_xy(hist_abs_from_offsets(his_off, n=4)) if his_off.size >= 2 else np.zeros((4, 2), np.float32)
    cmd = np.asarray(cur.get("gt_ego_fut_cmd", [0, 0, 1]), dtype=np.float32).reshape(-1)
    if cmd.size < 3:
        cmd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if cmd[0] > 0:
        mission, driving = "turn right", "turn right"
    elif cmd[1] > 0:
        mission, driving = "turn left", "turn left"
    else:
        mission, driving = "keep forward", "go straight"
    lcf = np.asarray(cur.get("gt_ego_lcf_feat", np.zeros(9)), dtype=np.float32).reshape(-1)
    v0 = float(lcf[7]) if lcf.size > 7 else 0.0
    return {
        "images": {"front_camera": [path], "front_left_camera": [], "front_right_camera": []},
        "sensor_data_path": "",
        "vehicle_velocity": [v0, 0.0],
        "vehicle_acceleration": [0.0, 0.0],
        "driving_command": driving,
        "mission_goal": mission,
        "hist": hist.astype(np.float32),
        "lcf": lcf,
        "gt": gt.astype(np.float32),
        "gt_mask": fut_mask[:6].astype(np.float32),
        "cmd": cmd.tolist(),
        "idx": idx,
        "token": cur["token"],
        "folder": cur.get("scene_token", ""),
        "frame_idx": cur.get("frame_idx", 0),
    }


def pack_kinematics(cur: dict) -> dict | None:
    """Ego kinematics without images; used to rebuild draft extras."""
    lcf = np.asarray(cur.get("gt_ego_lcf_feat", []), dtype=np.float32).reshape(-1)
    if lcf.size < 8:
        return None
    cmd = np.asarray(cur.get("gt_ego_fut_cmd", [0, 0, 1]), dtype=np.float32).reshape(-1)
    if cmd.size < 3:
        cmd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if cmd[0] > 0:
        driving = "turn right"
    elif cmd[1] > 0:
        driving = "turn left"
    else:
        driving = "go straight"
    v0 = float(lcf[7]) if lcf.size > 7 else 0.0
    hist = np.zeros((4, 2), dtype=np.float32)
    his_off = np.asarray(cur.get("gt_ego_his_trajs", []), dtype=np.float32)
    if his_off.size >= 2:
        hist = lidar_to_ego_xy(hist_abs_from_offsets(his_off, n=4))
    gt = None
    fut_off = np.asarray(cur.get("gt_ego_fut_trajs", []), dtype=np.float32)
    fut_mask = np.asarray(cur.get("gt_ego_fut_masks", []), dtype=np.float32)
    if fut_off.shape[0] >= 6:
        gt = lidar_to_ego_xy(np.cumsum(fut_off[:6], axis=0))
        if fut_mask.size >= 6 and fut_mask[:6].sum() < 6:
            gt = None
    return {
        "vehicle_velocity": [v0, 0.0],
        "driving_command": driving,
        "hist": hist,
        "lcf": lcf,
        "gt": gt,
        "token": cur.get("token", ""),
    }


def extras_for_tokens(tokens: list[str]) -> dict[str, tuple]:
    """Rebuild pack_extra / pack_speed / gt for cached tokens from VAD pkls."""
    import torch

    from trajspec.draft_head import pack_extra, pack_speed

    need = set(tokens)
    found: dict[str, tuple] = {}
    for pkl in (VAL_PKL, TRAIN_PKL):
        if not need or not pkl.exists():
            continue
        infos = load_infos(pkl)
        tok2i = token_index(infos)
        hit = [t for t in list(need) if t in tok2i]
        for t in hit:
            s = pack_kinematics(infos[tok2i[t]])
            need.discard(t)
            if s is None:
                continue
            g = s.get("gt")
            g_t = torch.as_tensor(g[:6], dtype=torch.float32) if g is not None else torch.zeros(6, 2)
            found[t] = (pack_extra(s), pack_speed(s), g_t)
        del infos, tok2i
    return found


def subsample_valid(infos: list[dict], stride: int, limit: int, start: int = 0):
    tok2i = token_index(infos)
    out = []
    i = start
    while i < len(infos) and len(out) < limit:
        feat = pack_sample(infos, i, tok2i)
        if feat is not None:
            out.append(feat)
            i += stride
        else:
            i += 1
    return out
