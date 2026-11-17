from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajspec.accept import AcceptConfig, longest_prefix_accept
from trajspec.adapters.toy import AutoVLAStyleTeacher, OpenDriveStyleTeacher
from trajspec.draft_head import DraftHead
from trajspec.scenes import sample_scenes
from trajspec.serialize import format_opendrive_traj, parse_opendrive_traj
from trajspec.spec_loop import speculative_trajectory


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
    teacher = OpenDriveStyleTeacher()
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


def test_autovla_quantized_teacher_length():
    teacher = AutoVLAStyleTeacher()
    sc = sample_scenes(1, teacher.horizon, seed=0)[0]
    h, kv = teacher.prefill(sc)
    wp = teacher.teacher_waypoints(sc, kv, h)
    assert wp.shape == (10, 2)
    ids = teacher.codebook.encode(wp)
    assert len(ids) == 10
    assert "<action_" in teacher.codebook.format_tokens(ids)


if __name__ == "__main__":
    test_longest_prefix_partial()
    test_longest_prefix_all()
    test_opendrive_roundtrip()
    test_spec_never_worse_than_threshold()
    test_autovla_quantized_teacher_length()
    print("ok")
