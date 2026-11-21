"""Waypoint-level speculative decoding for AR driving VLAs."""

from trajspec.accept import AcceptConfig, longest_prefix_accept
from trajspec.draft_head import DraftHead
from trajspec.spec_loop import SpecResult, speculative_trajectory
from trajspec.serialize import (
    format_opendrive_traj,
    parse_opendrive_traj,
)

__all__ = [
    "AcceptConfig",
    "longest_prefix_accept",
    "DraftHead",
    "SpecResult",
    "speculative_trajectory",
    "format_opendrive_traj",
    "parse_opendrive_traj",
]
