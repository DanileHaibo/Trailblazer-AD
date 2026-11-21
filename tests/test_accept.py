from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajspec.accept import AcceptConfig, longest_prefix_accept
from trajspec.draft_head import DraftHead
from trajspec.scenes import kinematic_trajectory, sample_scenes
from trajspec.serialize import format_opendrive_traj, parse_opendrive_traj
from trajspec.spec_loop import speculative_trajectory


class StubWaypointTeacher:
    """Deterministic in-process teacher for the spec_loop contract.

    Mirrors the real textual-waypoint teachers (OpenDriveVLA / R&B): the same
    scene always yields the same 6-waypoint trajectory, so accept and tail
    behaviour can be tested without loading a 4B checkpoint.
    """

    horizon = 6
    hidden_size = 32

    def __init__(self):
        self._traj = {}

    def prefill(self, scene: dict):
        h = torch.zeros(1, self.hidden_size)
        traj = kinematic_trajectory(
            float(scene.get("vehicle_velocity", [3.0])[0] or 3.0),
            float(scene.get("yaw_rate", 0.0)),
            self.horizon,
        )
        key = id(scene)
        self._traj[key] = traj
        return h.squeeze(0), {"scene": key}

    def teacher_waypoints(self, scene: dict, kv, hidden):
        key = kv["scene"]
        if key not in self._traj:
            self.prefill(scene)
        return self._traj[key]

    def continue_decode(self, scene: dict, kv, accepted, remaining: int):
        if remaining <= 0:
            return torch.zeros(0, 2)
        traj = self.teacher_waypoints(scene, kv, None)
        start = int(accepted.shape[0])
        return traj[start : start + remaining]


def test_longest_prefix_partial():
    draft = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 2.0]])
    teacher = torch.tensor([[1.02, 0.01], [2.05, -0.02], [3.0, 1.8], [4.0, 2.0]])
    n = longest_prefix_accept(draft, teacher, AcceptConfig(lon_m=0.1, lat_m=0.1))
    assert n == 2, n


def test_longest_prefix_all():
    a = torch.tensor([[0.0, 0.0], [1.0, 0.1]])
    assert longest_prefix_accept(a, a) == 2


def test_opendrive_roundtrip():
    traj = torch.tensor([[0.12, -0.03], [1.50, 0.20], [3.01, 0.40], [4.4, 0.5], [5.9, 0.55], [7.2, 0.6]])
    text = format_opendrive_traj(traj)
    assert text.startswith("<traj_start>[(")
    back = parse_opendrive_traj(text, horizon=6)
    assert back.shape == (6, 2)
    assert torch.allclose(back, traj, atol=6e-3)


def test_spec_never_worse_than_threshold():
    teacher = StubWaypointTeacher()
    scenes = sample_scenes(8, teacher.horizon, seed=7)
    head = DraftHead(teacher.hidden_size, horizon=teacher.horizon)
    cfg = AcceptConfig(lon_m=0.35, lat_m=0.20)
    for sc in scenes:
        r = speculative_trajectory(teacher, head, sc, cfg)
        k = min(r.n_accept, r.draft.shape[0], r.teacher_full.shape[0])
        if k:
            err = (r.draft[:k] - r.teacher_full[:k]).abs()
            assert err[:, 0].max() <= cfg.lon_m + 1e-5
            assert err[:, 1].max() <= cfg.lat_m + 1e-5
        assert r.trajectory.shape[0] == teacher.horizon
        assert r.spec_decode_steps <= r.baseline_decode_steps


def test_teacher_horizon_and_shape():
    teacher = StubWaypointTeacher()
    sc = sample_scenes(1, teacher.horizon, seed=0)[0]
    h, kv = teacher.prefill(sc)
    wp = teacher.teacher_waypoints(sc, kv, h)
    assert wp.shape == (6, 2)
    tail = teacher.continue_decode(sc, kv, wp[:2], 4)
    assert tail.shape == (4, 2)


if __name__ == "__main__":
    test_longest_prefix_partial()
    test_longest_prefix_all()
    test_opendrive_roundtrip()
    test_spec_never_worse_than_threshold()
    test_teacher_horizon_and_shape()
    print("ok")
