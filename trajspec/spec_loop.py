"""Prefill -> draft head -> waypoint-level verify -> continue AR decode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn

from trajspec.accept import AcceptConfig, longest_prefix_accept
from trajspec.draft_head import DraftHead, pack_extra, pack_speed


class ARTeacher(Protocol):
    """Minimal interface an AR driving VLA must expose."""

    horizon: int
    hidden_size: int

    def prefill(self, scene: dict) -> tuple[torch.Tensor, object]:
        """Return (last_hidden [D], kv_cache)."""
        ...

    def teacher_waypoints(self, scene: dict, kv: object, hidden: torch.Tensor) -> torch.Tensor:
        """Greedy AR trajectory of the original VLA. Shape [H, 2]."""
        ...

    def continue_decode(
        self,
        scene: dict,
        kv: object,
        accepted: torch.Tensor,
        remaining: int,
    ) -> torch.Tensor:
        """AR-decode `remaining` waypoints after an accepted prefix. [R, 2]."""
        ...


@dataclass
class SpecResult:
    trajectory: torch.Tensor
    n_accept: int
    draft: torch.Tensor
    teacher_full: torch.Tensor
    verify_teacher_prefix: torch.Tensor
    # Cost model in "decode steps" so we can report speedup without a GPU VLA.
    baseline_decode_steps: int
    spec_decode_steps: int
    extra: dict = field(default_factory=dict)

    @property
    def decode_speedup(self) -> float:
        # Prefill is shared; speculative decoding only shrinks the serial decode.
        # One parallel verify counts as 1 decode step (classic specdec accounting).
        spec = max(self.spec_decode_steps, 1e-6)
        return self.baseline_decode_steps / spec


@torch.no_grad()
def speculative_trajectory(
    teacher: ARTeacher,
    draft_head: DraftHead,
    scene: dict,
    accept_cfg: AcceptConfig | None = None,
    draft_k: int | None = None,
) -> SpecResult:
    """Draft K near-horizon points, verify against the AR teacher, continue.

    Verification is waypoint-level, not token-level:
      accept draft[:n] iff each point is within (lon, lat) metres of the
      teacher's greedy waypoint at that index. The first failure is replaced
      by the teacher's waypoint, then the teacher AR-decodes the suffix.
    """
    accept_cfg = accept_cfg or AcceptConfig()
    hidden, kv = teacher.prefill(scene)
    draft_k = draft_k or draft_head.horizon
    extra = pack_extra(scene).to(device=hidden.device)
    speed = pack_speed(scene).to(device=hidden.device)
    draft = draft_head(hidden, extra=extra, speed=speed)[0, :draft_k]  # [K, 2]

    # Teacher's own full AR trajectory is the distribution we must stay on.
    teacher_full = teacher.teacher_waypoints(scene, kv, hidden)
    H = int(teacher.horizon)
    verify_prefix = teacher_full[: draft.shape[0]]

    n_acc = longest_prefix_accept(draft, verify_prefix, accept_cfg)
    # Keep draft points the teacher agreed with (they are within threshold).
    # At the first reject, take the teacher's waypoint (specdec correction).
    if n_acc >= draft.shape[0] and n_acc >= H:
        traj = draft[:H]
        spec_steps = 1  # one parallel verify, nothing left to decode
        extra_reject = False
    elif n_acc >= H:
        traj = draft[:H]
        spec_steps = 1
        extra_reject = False
    else:
        parts = []
        if n_acc > 0:
            parts.append(draft[:n_acc])
        # Teacher correction at the reject index, then AR the rest.
        reject_idx = n_acc
        if reject_idx < H:
            parts.append(teacher_full[reject_idx : reject_idx + 1])
            rest = H - reject_idx - 1
            if rest > 0:
                accepted_so_far = teacher_full[: reject_idx + 1]
                suffix = teacher.continue_decode(scene, kv, accepted_so_far, rest)
                if suffix.numel() > 0:
                    parts.append(suffix[:rest])
        hard = torch.cat(parts, dim=0)[:H]
        # Hand over to the teacher along the draft's own velocity so the splice
        # is continuous (see trajspec.sd_baselines.seam_blend).
        from trajspec.sd_baselines import seam_blend

        traj = seam_blend(draft[:H], hard, n_acc, horizon=H) if n_acc < H else hard
        # 1 verify + remaining serial decode tokens after the correction
        spec_steps = 1 + max(H - n_acc - 1, 0)
        extra_reject = True

    if traj.shape[0] < H:
        pad = teacher_full[traj.shape[0] : H]
        traj = torch.cat([traj, pad], dim=0) if pad.numel() else traj

    return SpecResult(
        trajectory=traj,
        n_accept=n_acc,
        draft=draft,
        teacher_full=teacher_full,
        verify_teacher_prefix=verify_prefix,
        baseline_decode_steps=H,
        spec_decode_steps=spec_steps,
        extra={"corrected": extra_reject, "hidden_norm": float(hidden.float().norm())},
    )


def train_step(
    draft_head: DraftHead,
    hidden: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    extra: torch.Tensor | None = None,
    speed: torch.Tensor | None = None,
) -> float:
    """Supervise the draft head on teacher near-horizon waypoints."""
    draft_head.train()
    pred = draft_head(hidden, extra=extra, speed=speed)
    k = min(pred.shape[1], target.shape[1])
    tgt = target[:, :k].float()
    if tgt.ndim == 2:
        tgt = tgt.unsqueeze(0)
    w_list = [2.4, 2.0, 1.6, 1.2, 0.9, 0.7] + [0.55] * max(0, k - 6)
    w_t = torch.tensor(w_list[:k], device=pred.device).view(1, k, 1)
    w_xy = torch.tensor([1.15, 2.0], device=pred.device).view(1, 1, 2)
    err = nn.functional.smooth_l1_loss(pred[:, :k], tgt, beta=0.2, reduction="none")
    loss = (err * w_t * w_xy).mean()
    # Accept-aligned hinge: push every waypoint inside the metre box.
    box = (pred[:, :k] - tgt).abs()
    lon_m, lat_m = 0.50, 0.30
    hinge = torch.relu(box[:, :, 0] - lon_m) + 1.4 * torch.relu(box[:, :, 1] - lat_m)
    loss = loss + 1.6 * (hinge * w_t.view(1, k)).mean()
    if k >= 2:
        d_pred = pred[:, 1:k] - pred[:, : k - 1]
        d_tgt = tgt[:, 1:k] - tgt[:, : k - 1]
        loss = loss + 0.55 * nn.functional.smooth_l1_loss(d_pred, d_tgt, beta=0.15)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(draft_head.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())
