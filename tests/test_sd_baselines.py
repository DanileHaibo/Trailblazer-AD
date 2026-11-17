from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajspec.accept import AcceptConfig
from trajspec.sd_baselines import (
    GEOMETRIC,
    KERV_LITE,
    TOKEN_EXACT,
    commit_spec,
    kinematic_complete,
    longest_id_prefix,
    n_accept_autovla,
)


def test_id_prefix():
    assert longest_id_prefix([1, 2, 3], [1, 2, 9], radius=0) == 2
    assert longest_id_prefix([10, 20], [12, 20], radius=2) == 2
    assert longest_id_prefix([10, 20], [14, 20], radius=2) == 0


def test_token_exact_stricter_than_geometry():
    draft = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    teacher = torch.tensor([[0.02, 0.01], [1.04, -0.02], [2.0, 0.0]])
    cfg = AcceptConfig(lon_m=0.5, lat_m=0.3)
    n_geo = n_accept_autovla(GEOMETRIC, draft, teacher, [0, 1, 2], [9, 8, 2], cfg)
    n_tok = n_accept_autovla(TOKEN_EXACT, draft, teacher, [0, 1, 2], [9, 8, 2], cfg)
    assert n_geo == 3
    assert n_tok == 0


def test_kerv_no_serial():
    draft = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.5], [4.0, 1.0], [5.0, 1.5], [6.0, 2.0]])
    teacher = torch.zeros(6, 2)
    spec, n, serial = commit_spec(draft, teacher, 2, KERV_LITE, speed_ms=4.0, horizon=6)
    assert n == 2
    assert serial == 0
    assert spec.shape == (6, 2)
    assert torch.allclose(spec[:2], draft[:2])
    # remaining points continue forward, not copied from teacher zeros
    assert float(spec[5, 0]) > 3.0


def test_kinematic_from_empty():
    draft = torch.zeros(6, 2)
    out = kinematic_complete(draft, 0, speed_ms=10.0, horizon=6, dt=0.5)
    assert out.shape == (6, 2)
    assert abs(float(out[0, 0]) - 5.0) < 1e-4


if __name__ == "__main__":
    test_id_prefix()
    test_token_exact_stricter_than_geometry()
    test_kerv_no_serial()
    test_kinematic_from_empty()
    print("ok")
