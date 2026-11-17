"""Seam-correction unit tests (trajspec.sd_baselines.seam_blend)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajspec.sd_baselines import commit_spec, seam_blend  # noqa: E402


def _trajs():
    d = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.1], [4.0, 0.2], [5.0, 0.3], [6.0, 0.4]])
    t = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.5, 0.5], [5.0, 1.0], [6.5, 1.6], [8.0, 2.3]])
    return d, t


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 6])
def test_finite_and_shape(n):
    d, t = _trajs()
    out = seam_blend(d, t, n)
    assert out.shape == (6, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_prefix_preserved(n):
    d, t = _trajs()
    out = seam_blend(d, t, n)
    assert torch.allclose(out[:n], d[:n])


def test_endpoint_is_teacher():
    d, t = _trajs()
    for n in (1, 2, 3, 5, 6):
        out = seam_blend(d, t, n)
        expected = t[-1] if n < 6 else d[-1]
        assert torch.allclose(out[-1], expected, atol=1e-5)


def test_zero_accept_is_pure_teacher():
    d, t = _trajs()
    assert torch.allclose(seam_blend(d, t, 0), t)


def test_full_accept_is_pure_draft():
    d, t = _trajs()
    assert torch.allclose(seam_blend(d, t, 6), d)


def test_smoother_than_hard_splice():
    """The blend must not create a bigger jump than the hard splice does."""
    d, t = _trajs()
    n = 2
    hard = torch.cat([d[:n], t[n : n + 1], t[n + 1 :]], 0)[:6]
    blended = seam_blend(d, t, n)

    def jump(x):
        steps = (x[1:] - x[:-1]).norm(dim=-1)
        return float((steps[1:] - steps[:-1]).abs().max())

    assert jump(blended) < jump(hard)


def test_zero_velocity_is_finite():
    out = seam_blend(torch.zeros(6, 2), torch.ones(6, 2), 3)
    assert torch.isfinite(out).all()


def test_commit_spec_defaults_to_blend():
    d, t = _trajs()
    spec, n_acc, n_serial = commit_spec(d, t, 2, "geometric")
    assert n_acc == 2
    assert n_serial == 3
    assert torch.allclose(spec, seam_blend(d, t, 2))


def test_commit_spec_hard_reproduces_paper():
    d, t = _trajs()
    spec, _, _ = commit_spec(d, t, 2, "geometric", seam="hard")
    hard = torch.cat([d[:2], t[2:3], t[3:]], 0)[:6]
    assert torch.allclose(spec, hard)
