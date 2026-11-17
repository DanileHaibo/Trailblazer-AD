"""Bench2Drive open-loop GT and camera packing. Matches VAD B2D (2 Hz, 6 steps, 3 s)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

CAM_FRONT = "CAM_FRONT"
CAM_FL = "CAM_FRONT_LEFT"
CAM_FR = "CAM_FRONT_RIGHT"
CMD_TEXT = {
    0: "go straight",
    1: "turn left",
    2: "turn right",
    3: "go straight",
    4: "follow the lane",
    5: "change lane to the left",
    6: "change lane to the right",
}


def load_infos(path: str | Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def ego_trajs(infos: list[dict], idx: int, sample_rate: int = 5, past: int = 2, future: int = 6):
    """Port of B2D_VAD_Dataset.get_ego_trajs. Returns offsets [T,2], mask, command."""
    adj = range(idx - past * sample_rate, idx + (future + 1) * sample_rate, sample_rate)
    cur = infos[idx]
    track = np.zeros((past + future + 1, 2), dtype=np.float32)
    mask = np.zeros(past + future + 1, dtype=np.float32)
    w2l_cur = cur["sensors"]["LIDAR_TOP"]["world2lidar"]
    for j, adj_idx in enumerate(adj):
        if adj_idx < 0 or adj_idx >= len(infos):
            break
        fr = infos[adj_idx]
        if fr["folder"] != cur["folder"]:
            break
        w2l_adj = fr["sensors"]["LIDAR_TOP"]["world2lidar"]
        adj2cur = w2l_cur @ np.linalg.inv(w2l_adj)
        track[j] = adj2cur[0:2, 3]
        mask[j] = 1
    offset = track[1:] - track[:-1]
    for j in range(past - 1, -1, -1):
        if mask[j] == 0:
            offset[j] = offset[j + 1]
    for j in range(past, past + future):
        if mask[j + 1] == 0:
            offset[j] = 0
    cmd = int(cur.get("command_near", 4))
    return offset[:past].copy(), offset[past:].copy(), mask[-future:].copy(), cmd


def abs_future(fut_off: np.ndarray) -> np.ndarray:
    return np.cumsum(fut_off, axis=0)


def lidar_xy_to_ego(xy: np.ndarray) -> np.ndarray:
    """B2D lidar is +Y forward, +X ≈ left (see lidar2ego). Map to nuScenes-style +X forward, +Y left."""
    xy = np.asarray(xy, dtype=np.float32)
    out = np.empty_like(xy)
    out[..., 0] = xy[..., 1]
    out[..., 1] = xy[..., 0]
    return out


def ego_hist_abs(infos: list[dict], idx: int, past: int = 4, sample_rate: int = 5) -> np.ndarray:
    """Past ego xy in the current lidar frame, oldest first. [past, 2]."""
    cur = infos[idx]
    w2l_cur = cur["sensors"]["LIDAR_TOP"]["world2lidar"]
    pts = []
    for k in range(past, 0, -1):
        adj_idx = idx - k * sample_rate
        if adj_idx < 0 or adj_idx >= len(infos) or infos[adj_idx]["folder"] != cur["folder"]:
            pts.append(np.zeros(2, dtype=np.float32))
            continue
        w2l_adj = infos[adj_idx]["sensors"]["LIDAR_TOP"]["world2lidar"]
        adj2cur = w2l_cur @ np.linalg.inv(w2l_adj)
        pts.append(adj2cur[0:2, 3].astype(np.float32))
    return np.stack(pts, axis=0)


def cam_path(info: dict, cam: str, data_root: Path) -> Path:
    rel = info["sensors"][cam]["data_path"]
    return data_root / rel


def history_indices(idx: int, n_hist: int = 4, sample_rate: int = 5) -> list[int]:
    out = [max(0, idx - (n_hist - 1 - k) * sample_rate) for k in range(n_hist)]
    return out


def pack_autovla_features(infos, idx, data_root: Path) -> dict | None:
    cur = infos[idx]
    hist_idx = history_indices(idx)
    # All history frames must stay on the same route.
    if any(infos[j]["folder"] != cur["folder"] for j in hist_idx):
        return None
    _, fut_off, fut_mask, cmd = ego_trajs(infos, idx)
    if fut_mask.sum() < 6:
        return None
    gt = lidar_xy_to_ego(abs_future(fut_off))
    hist_xy = lidar_xy_to_ego(ego_hist_abs(infos, idx, past=4))
    images = {"front_camera": [], "front_left_camera": [], "front_right_camera": []}
    for j in hist_idx:
        images["front_camera"].append(str(cam_path(infos[j], CAM_FRONT, data_root)))
        images["front_left_camera"].append(str(cam_path(infos[j], CAM_FL, data_root)))
        images["front_right_camera"].append(str(cam_path(infos[j], CAM_FR, data_root)))
        if not Path(images["front_camera"][-1]).exists():
            return None
    vel = np.asarray(cur["ego_vel"], dtype=np.float32).reshape(-1)
    acc = np.asarray(cur["ego_accel"], dtype=np.float32).reshape(-1)
    cmd_i = int(cmd)
    if cmd_i == 1 or cmd_i == 5:
        mission = "turn left"
    elif cmd_i == 2 or cmd_i == 6:
        mission = "turn right"
    else:
        mission = "keep forward"
    return {
        "images": images,
        "sensor_data_path": "",
        "vehicle_velocity": vel[:2].tolist(),
        "vehicle_acceleration": acc[:2].tolist(),
        "driving_command": CMD_TEXT.get(int(cmd), "follow the lane"),
        "mission_goal": mission,
        "hist": hist_xy.astype(np.float32),
        "gt": gt.astype(np.float32),
        "gt_mask": fut_mask.astype(np.float32),
        "cmd": int(cmd),
        "idx": idx,
        "folder": cur["folder"],
        "frame_idx": cur["frame_idx"],
        "feat": np.array(
            [
                float(np.linalg.norm(vel[:2])) / 15.0,
                float(cmd) / 6.0,
                float(vel[0]) / 10.0 if vel.size else 0.0,
                float(vel[1]) / 10.0 if vel.size > 1 else 0.0,
            ],
            dtype=np.float32,
        ),
    }


def subsample_valid(infos, data_root: Path, stride: int, limit: int, start: int = 0):
    out = []
    i = start
    while i < len(infos) and len(out) < limit:
        feat = pack_autovla_features(infos, i, data_root)
        if feat is not None:
            out.append(feat)
            i += stride
        else:
            i += 1
    return out


def l2_horizon(pred: np.ndarray, gt: np.ndarray) -> dict:
    n = min(len(pred), len(gt), 6)
    if n == 0:
        return {"l2_1s": None, "l2_2s": None, "l2_3s": None, "ade": None}
    d = np.linalg.norm(pred[:n, :2] - gt[:n, :2], axis=-1)
    def mean_upto(t_pts):
        k = min(t_pts, n)
        return float(d[:k].mean()) if k else None
    return {
        "l2_1s": mean_upto(2),
        "l2_2s": mean_upto(4),
        "l2_3s": mean_upto(6),
        "ade": float(d.mean()),
        "l2_at": d.tolist(),
    }
