"""Accept policies for SD baselines on OpenDriveVLA / R&B-EnCoRe.

All methods share the same one-shot DraftHead proposal and (in live eval) the
same teacher-force verify. Only the accept rule and the reject recovery differ.

  token_exact   — Leviathan/Chen greedy SD; draft token must equal teacher token
  specvla_id    — Spec-VLA relaxed accept on token ids |Δid| ≤ r
  scaffold      — OpenDrive: punctuation/structure tokens may mismatch
  geometric     — TrajSpecDec lon/lat metres (ours)
  kerv_lite     — geometric accept, then bicycle-fill the tail (no AR re-decode)
  kerv_bonus    — geometric accept, teacher bonus at reject, bicycle after (no AR)
  geom_blend    — geometric accept, seam-blend into verify teacher_xy (no AR)
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

from trajspec.accept import AcceptConfig, longest_prefix_accept

GEOMETRIC = "geometric"
TOKEN_EXACT = "token_exact"
SPECVLA_ID = "specvla_id"
SPECVLA_POSE = "specvla_pose"
SCAFFOLD = "scaffold"
KERV_LITE = "kerv_lite"
KERV_BONUS = "kerv_bonus"
GEOM_BLEND = "geom_blend"

# Spec-VLA uses r=5–9 on 256 discretized bins; the waypoint tokenizers here
# put a whole coordinate in one token, so a small id radius is already a large
# relaxation.
DEFAULT_ID_RADIUS = 16
DEFAULT_POSE_M = 0.50


def longest_id_prefix(
    draft_ids: Sequence[int],
    teacher_ids: Sequence[int],
    radius: int = 0,
) -> int:
    n = 0
    for a, b in zip(draft_ids, teacher_ids):
        if abs(int(a) - int(b)) > int(radius):
            break
        n += 1
    return n


def longest_pose_prefix(draft_xy: torch.Tensor, teacher_xy: torch.Tensor, tau_m: float = DEFAULT_POSE_M) -> int:
    """Spec-VLA-style relax in metres: accept while Euclidean error ≤ τ."""
    d = torch.as_tensor(draft_xy[:, :2], dtype=torch.float32)
    t = torch.as_tensor(teacher_xy[:, :2], dtype=torch.float32)
    k = min(int(d.shape[0]), int(t.shape[0]))
    if k == 0:
        return 0
    err = (d[:k] - t[:k]).norm(dim=-1)
    bad = (err > float(tau_m)).nonzero(as_tuple=False)
    if bad.numel() == 0:
        return k
    return int(bad[0, 0].item())


def is_scaffold_text(text: str) -> bool:
    """True if a decoded token has no digit — brackets, commas, traj tags, signs."""
    s = (text or "").strip()
    if not s:
        return True
    return not any(ch.isdigit() for ch in s)


def longest_scaffold_prefix(
    draft_ids: Sequence[int],
    teacher_ids: Sequence[int],
    tokenizer,
) -> int:
    """Accept an exact token match, or a pair of non-numeric (syntax) tokens."""
    n = 0
    for a, b in zip(draft_ids, teacher_ids):
        if int(a) == int(b):
            n += 1
            continue
        da = tokenizer.decode([int(a)], skip_special_tokens=False)
        tb = tokenizer.decode([int(b)], skip_special_tokens=False)
        if is_scaffold_text(da) and is_scaffold_text(tb):
            n += 1
            continue
        break
    return n


def waypoints_in_token_prefix(tokenizer, token_ids: Sequence[int]) -> int:
    from trajspec.serialize import extract_opendrive_points

    if not token_ids:
        return 0
    text = tokenizer.decode(list(token_ids), skip_special_tokens=False)
    return len(extract_opendrive_points(text))


def n_accept_opendrive(
    mode: str,
    draft_xy: torch.Tensor,
    teacher_xy: torch.Tensor,
    draft_ids: Sequence[int],
    teacher_ids: Sequence[int],
    tokenizer,
    cfg: AcceptConfig | None = None,
    horizon: int = 6,
    pose_m: float = DEFAULT_POSE_M,
) -> int:
    mode = str(mode)
    k = min(int(draft_xy.shape[0]), int(teacher_xy.shape[0]), horizon)
    if mode in (GEOMETRIC, KERV_LITE):
        return longest_prefix_accept(draft_xy[:k], teacher_xy[:k], cfg)
    if mode == SPECVLA_POSE:
        return longest_pose_prefix(draft_xy[:k], teacher_xy[:k], tau_m=pose_m)
    if mode == TOKEN_EXACT:
        n_tok = longest_id_prefix(draft_ids, teacher_ids, radius=0)
        return min(waypoints_in_token_prefix(tokenizer, draft_ids[:n_tok]), k)
    if mode in (SCAFFOLD, SPECVLA_ID):
        n_tok = longest_scaffold_prefix(draft_ids, teacher_ids, tokenizer)
        return min(waypoints_in_token_prefix(tokenizer, draft_ids[:n_tok]), k)
    raise ValueError(f"unknown OpenDrive accept mode {mode}")


def kinematic_complete(
    draft_xy: torch.Tensor,
    n_acc: int,
    speed_ms: float,
    horizon: int = 6,
    dt: float = 0.5,
) -> torch.Tensor:
    """KERV-lite tail: constant-speed bicycle along the last accepted heading."""
    d = draft_xy[:horizon].float().reshape(-1, 2)
    k = int(d.shape[0])
    n_acc = max(0, min(int(n_acc), k))
    out = d.clone()
    v = max(float(speed_ms), 0.0)
    if n_acc <= 0:
        heading = 0.0
        pos = d.new_zeros(2)
        start = 0
    else:
        pos = d[n_acc - 1].clone()
        if n_acc >= 2:
            delta = d[n_acc - 1] - d[n_acc - 2]
        else:
            delta = d[0]
        heading = math.atan2(float(delta[1]), float(delta[0] + 1e-6))
        if v < 1e-3:
            v = float(delta.norm()) / dt
        start = n_acc
    c, s = math.cos(heading), math.sin(heading)
    step = d.new_tensor([v * dt * c, v * dt * s])
    cur = pos
    for i in range(start, k):
        cur = cur + step
        out[i] = cur
    return out


def seam_blend(
    draft_xy: torch.Tensor,
    teacher_xy: torch.Tensor,
    n_acc: int,
    horizon: int = 6,
) -> torch.Tensor:
    """Splice the accepted draft prefix into the teacher suffix without a jump.

    The paper's splice (`draft[:n] + teacher[n:]`) is C0/C1 discontinuous: the
    path teleports from the draft's position onto the teacher's at the reject
    index. Measured on held-out data this costs 0.02-0.03 m L2 and, on live
    frames, up to 1.2 m on the worst sample.

    Repair: extrapolate the accepted prefix along its own last velocity, then
    linearly hand over to the teacher over the remaining steps:

        out[j] = (1 - a_j) * (d[n-1] + v * (j-n+1)) + a_j * t[j],
        a_j    = (j-n+1) / (k-n)

    so `a` reaches 1 exactly at the last waypoint and the endpoint stays the
    teacher's. Verified on both teachers: lower L2 and lower step-curvature than
    the hard splice (see diag/diag_seam.py).
    """
    d = torch.as_tensor(draft_xy[:, :2], dtype=torch.float32)
    t = torch.as_tensor(teacher_xy[:, :2], dtype=torch.float32)
    k = min(int(d.shape[0]), int(t.shape[0]), horizon)
    n = max(0, min(int(n_acc), k))
    if n <= 0:
        return t[:k].clone()
    if n >= k:
        return d[:k].clone()
    out = d.new_empty(k, 2)
    out[:n] = d[:n]
    v = d[n - 1] - d[n - 2] if n >= 2 else d[0]
    span = max(k - n, 1)
    for j in range(n, k):
        a = (j - n + 1) / span
        extrap = d[n - 1] + v * (j - n + 1)
        out[j] = (1.0 - a) * extrap + a * t[j]
    return out


def commit_spec(
    draft_xy: torch.Tensor,
    teacher_xy: torch.Tensor,
    n_acc: int,
    mode: str,
    speed_ms: float = 0.0,
    horizon: int = 6,
    seam: str = "blend",
) -> tuple[torch.Tensor, int, int]:
    """Return (spec_xy, n_accept, n_serial_waypoints).

    Standard SD: keep draft[:n], take teacher[n] as the bonus token, copy the
    remaining teacher suffix (offline proxy of serial AR). KERV-lite skips the
    suffix and fills with a bicycle instead.

    `seam="blend"` (default) hands over to the teacher along the draft's own
    velocity instead of hard-copying teacher[n:]; `seam="hard"` reproduces the
    paper's original splice.
    """
    d = torch.as_tensor(draft_xy[:, :2], dtype=torch.float32)
    t = torch.as_tensor(teacher_xy[:, :2], dtype=torch.float32)
    k = min(int(d.shape[0]), int(t.shape[0]), horizon)
    n = max(0, min(int(n_acc), k))
    if str(mode) == KERV_LITE:
        spec = kinematic_complete(d[:k], n, speed_ms, horizon=k)
        return spec, n, 0
    if str(mode) == KERV_BONUS:
        # Keep accepted draft, splice teacher bonus point, bicycle-fill the rest.
        # No LLM serial tail — bicycle-fill keeps E2E below the AR baseline.
        if n >= k:
            return d[:k], n, 0
        out = d[:k].clone()
        out[n] = t[n]
        if n + 1 < k:
            # Bicycle from the teacher bonus along draft (or teacher) heading.
            v = max(float(speed_ms), 0.0)
            if n >= 1:
                delta = out[n] - out[n - 1]
            else:
                delta = out[n]
            heading = math.atan2(float(delta[1]), float(delta[0] + 1e-6))
            if v < 1e-3:
                v = float(delta.norm()) / 0.5
            step = out.new_tensor([v * 0.5 * math.cos(heading), v * 0.5 * math.sin(heading)])
            cur = out[n].clone()
            for i in range(n + 1, k):
                cur = cur + step
                out[i] = cur
        return out, n, 0
    if str(mode) == GEOM_BLEND:
        # Free seam into the parallel-verify teacher trajectory (no serial AR).
        if n >= k:
            return d[:k], n, 0
        spec = seam_blend(d[:k], t[:k], n, horizon=k)
        return spec, n, 0
    if n >= k:
        return d[:k], n, 0
    if str(seam) == "blend":
        spec = seam_blend(d[:k], t[:k], n, horizon=k)
        return spec, n, max(k - n - 1, 0)
    parts = []
    if n > 0:
        parts.append(d[:n])
    parts.append(t[n : n + 1])
    if n + 1 < k:
        parts.append(t[n + 1 : k])
    spec = torch.cat(parts, dim=0)[:k]
    n_serial = max(k - n - 1, 0)
    return spec, n, n_serial


def hash_holdout_idx(n: int, frac: float = 0.15, seed: int = 17) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(64, int(round(n * frac)))
    return perm[:n_val]
