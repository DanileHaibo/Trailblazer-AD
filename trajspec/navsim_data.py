"""NAVSIM / OpenScene helpers: scene filter, nocot JSON, AutoVLA features."""
from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path("/data2/navsim")
LOG_DIR = ROOT / "navsim_logs" / "test"
SENSOR_DIR = ROOT / "sensor_blobs" / "test"
JSON_DIR = ROOT / "navtest_nocot"
NAVTEST_YAML = Path(
    "/home/xinhchen2/hhb/AutoVLA/navsim/navsim/planning/script/config/common/"
    "train_test_split/scene_filter/navtest.yaml"
)

CAM_MAP = {
    "front": "CAM_F0",
    "front_left": "CAM_L0",
    "front_right": "CAM_R0",
    "left": "CAM_L1",
    "right": "CAM_R1",
    "back_left": "CAM_L2",
    "back_right": "CAM_R2",
    "back": "CAM_B0",
}

CMD_NAMES = ("TURN LEFT", "KEEP FORWARD", "TURN RIGHT", "UNKNOWN")


def quat_yaw(q: Iterable[float]) -> float:
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def global_to_ego(origin_xyh: np.ndarray, poses_xyh: np.ndarray) -> np.ndarray:
    ox, oy, oh = [float(v) for v in origin_xyh]
    theta = -oh
    c, s = math.cos(theta), math.sin(theta)
    rel = np.asarray(poses_xyh, dtype=np.float64) - np.array([ox, oy, oh], dtype=np.float64)
    xy = rel[:, :2] @ np.array([[c, -s], [s, c]], dtype=np.float64).T
    heading = np.arctan2(np.sin(rel[:, 2]), np.cos(rel[:, 2]))
    return np.concatenate([xy, heading[:, None]], axis=1).astype(np.float32)


def frame_xyh(frame: dict) -> np.ndarray:
    t = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    yaw = quat_yaw(frame["ego2global_rotation"])
    return np.array([t[0], t[1], yaw], dtype=np.float64)


def command_name(cmd) -> str:
    arr = np.asarray(cmd).reshape(-1)
    idx = int(np.argmax(arr)) if arr.size else 1
    if 0 <= idx < len(CMD_NAMES):
        return CMD_NAMES[idx]
    return "KEEP FORWARD"


def load_navtest_filter(path: Path = NAVTEST_YAML) -> tuple[set[str], set[str] | None]:
    """Return (log_names, tokens). tokens is None if the yaml has no token list."""
    text = path.read_text()
    log_block = re.search(r"^log_names:\s*\n((?:^\s+- .*\n)+)", text, flags=re.M)
    tok_block = re.search(r"^tokens:\s*\n((?:^\s+- .*\n)+)", text, flags=re.M)
    log_names = set(re.findall(r"- '([^']+)'", log_block.group(1) if log_block else ""))
    tokens = set(re.findall(r"- '([^']+)'", tok_block.group(1) if tok_block else "")) or None
    return log_names, tokens


def load_navtest_log_names(path: Path = NAVTEST_YAML) -> set[str]:
    logs, _ = load_navtest_filter(path)
    return logs


def iter_navtest_scenes(
    log_dir: Path = LOG_DIR,
    num_history: int = 4,
    num_future: int = 10,
    frame_interval: int = 1,
    has_route: bool = True,
    log_names: set[str] | None = None,
):
    n_frames = num_history + num_future
    if log_names is None:
        log_names, tokens = load_navtest_filter()
    else:
        tokens = None
    pkls = sorted(log_dir.glob("*.pkl"))
    pkls = [p for p in pkls if p.stem in log_names]
    for pkl_path in pkls:
        frames = pickle.load(open(pkl_path, "rb"))
        for start in range(0, len(frames), frame_interval):
            window = frames[start : start + n_frames]
            if len(window) < n_frames:
                continue
            current = window[num_history - 1]
            if has_route and len(current.get("roadblock_ids") or []) == 0:
                continue
            if tokens is not None and current["token"] not in tokens:
                continue
            yield pkl_path.stem, window, current


def _cam_paths(window: list[dict], cam_key: str, num_history: int = 4) -> list[str]:
    out = []
    for fr in window[:num_history]:
        out.append(fr["cams"][cam_key]["data_path"])
    return out


def scene_to_json(window: list[dict], current: dict, num_history: int = 4, num_future: int = 10) -> dict:
    dyn = list(current["ego_dynamic_state"])
    origin = frame_xyh(current)
    fut = np.stack([frame_xyh(fr) for fr in window[num_history : num_history + num_future]], axis=0)
    his = np.stack([frame_xyh(fr) for fr in window[:num_history]], axis=0)
    gt = global_to_ego(origin, fut)
    hist_local = global_to_ego(origin, his)
    rec: dict[str, Any] = {
        "token": current["token"],
        "dataset_name": "nuplan",
        "log_name": current["log_name"],
        "map_location": current["map_location"],
        "cot_output": [],
        "velocity": [float(dyn[0]), float(dyn[1])],
        "acceleration": [float(dyn[2]), float(dyn[3])] if len(dyn) >= 4 else [0.0, 0.0],
        "instruction": command_name(current["driving_command"]).lower(),
        "gt_trajectory": gt.tolist(),
        "his_trajectory": hist_local.tolist(),
    }
    for name, cam in CAM_MAP.items():
        rec[f"{name}_camera_paths"] = _cam_paths(window, cam, num_history)
    return rec


def json_to_autovla_features(rec: dict, sensor_root: Path = SENSOR_DIR, official_nuplan_cams: bool = False) -> dict:
    """Pack AutoVLARuntime features. Default uses CAM_F0/L0/R0 (prompt cameras)."""
    if official_nuplan_cams:
        fl = rec.get("left_camera_paths") or rec["front_left_camera_paths"]
        fr = rec.get("right_camera_paths") or rec["front_right_camera_paths"]
    else:
        fl = rec["front_left_camera_paths"]
        fr = rec["front_right_camera_paths"]
    images = {
        "front_camera": rec["front_camera_paths"],
        "front_left_camera": fl,
        "front_right_camera": fr,
    }
    hist = np.asarray(rec.get("his_trajectory") or [], dtype=np.float32)
    return {
        "token": rec["token"],
        "images": images,
        "sensor_data_path": str(sensor_root),
        "vehicle_velocity": rec["velocity"],
        "vehicle_acceleration": rec["acceleration"],
        "driving_command": rec["instruction"],
        "gt": np.asarray(rec["gt_trajectory"], dtype=np.float32)[:, :2],
        "hist": hist[:, :2] if hist.ndim == 2 and hist.shape[-1] >= 2 else hist,
        "log_name": rec.get("log_name"),
    }


def load_json_dir(json_dir: Path = JSON_DIR) -> list[dict]:
    recs = []
    for p in sorted(json_dir.glob("*.json")):
        recs.append(json.loads(p.read_text()))
    return recs
