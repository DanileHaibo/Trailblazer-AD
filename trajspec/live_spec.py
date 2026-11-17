"""Live speculative decode: one teacher-force forward verifies K draft tokens.

Prefill logits verify draft[0]; the K-token teacher-force logits verify draft[1:].
That is one parallel verify, not K serial generates. Reject → take teacher token
at that index, then serial-decode the suffix from a short continuation forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from trajspec.accept import AcceptConfig, longest_prefix_accept


@dataclass
class LiveSpecResult:
    trajectory: torch.Tensor
    draft: torch.Tensor
    teacher_verify: torch.Tensor
    n_accept: int
    n_verify_tokens: int
    n_serial_tail: int
    action_ids: list[int]
    times_ms: dict = field(default_factory=dict)

    @property
    def spec_decode_steps(self) -> int:
        return 1 + max(self.n_serial_tail, 0)


def greedy_action_id(logits: torch.Tensor, action_start: int, n_bins: int) -> int:
    """Argmax inside the action-token slice."""
    sl = logits[action_start : action_start + n_bins]
    return int(sl.argmax().item())


def verify_ids_from_logits(
    prefill_logits: torch.Tensor,
    force_logits: torch.Tensor,
    draft_ids: list[int],
    action_start: int,
    n_bins: int,
) -> list[int]:
    """Teacher greedy action id at each of the K draft positions.

    force_logits: [K, vocab] from one forward of the K draft tokens with prefill KV.
    force_logits[i] = p(next | prompt, draft_ids[:i+1]), so it verifies draft_ids[i+1].
    """
    k = len(draft_ids)
    teacher_ids = [greedy_action_id(prefill_logits, action_start, n_bins)]
    for i in range(k - 1):
        teacher_ids.append(greedy_action_id(force_logits[i], action_start, n_bins))
    return teacher_ids[:k]


def live_accept(
    draft_xy: torch.Tensor,
    teacher_xy: torch.Tensor,
    cfg: AcceptConfig,
) -> int:
    return longest_prefix_accept(draft_xy.float(), teacher_xy.float(), cfg)
