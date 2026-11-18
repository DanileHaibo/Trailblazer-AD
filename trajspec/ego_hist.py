#!/usr/bin/env python3
"""Build a 4-step (2 s) ego history from the prev-chain instead of the 2-step
`gt_ego_his_trajs` that the VAD temporal pkl ships.

The RnB-EnCoRe nuScenes checkpoint was trained with
`Historical Trajectory (last 2 seconds): [(...), (...), (...), (...)]`
i.e. four 0.5 s steps. VAD's pkl only carries two, so pack_sample_full pads by
repeating the first point and the model sees a truncated history - one of the
suspected causes of the L2 gap (ours ~1.07 vs paper 0.72).

This module recomputes the history in the ego frame from `ego2global_*` poses
walking back along `prev`, so four real samples are available.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _quat_yaw(q) -> float:
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _pose(info: dict) -> np.ndarray:
    t = np.asarray(info["ego2global_translation"], dtype=np.float64)
    return np.array([t[0], t[1], _quat_yaw(info["ego2global_rotation"])], dtype=np.float64)


def ego_history_4(infos: list[dict], idx: int, tok2i: dict[str, int], n: int = 4) -> np.ndarray | None:
    """Return [n,2] past positions in the current ego frame (oldest first)."""
    chain = [idx]
    cur = idx
    for _ in range(n):
        prev = infos[cur].get("prev") or ""
        if not prev or prev not in tok2i:
            break
        nxt = tok2i[prev]
        if infos[nxt]["scene_token"] != infos[idx]["scene_token"]:
            break
        chain.append(nxt)
        cur = nxt
    if len(chain) < 2:
        return None

    poses = np.stack([_pose(infos[i]) for i in chain])  # current .. oldest
    ox, oy, oh = poses[0]
    rel = poses - np.array([ox, oy, oh])
    c, s = math.cos(-oh), math.sin(-oh)
    xy = np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], axis=-1)
    xy = xy[1:][::-1]  # drop current, oldest first
    if xy.shape[0] < n:
        xy = np.vstack([np.repeat(xy[:1], n - xy.shape[0], axis=0), xy])
    return xy[-n:].astype(np.float32)
