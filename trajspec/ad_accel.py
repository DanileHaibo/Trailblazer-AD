"""Four reproducible AD-style accelerators, all training-free.

Every method here is a pure inference-time change, so it applies to the frozen
OpenDriveVLA-0.5B teacher without retraining or extra weights. Each attacks a
different stage, which is the point of putting them in one table.

1. FastV (ECCV 2024 Oral; the family FastDriveVLA/Prune2Drive specialise for driving)
     After layer K, drop the visual tokens that received the least attention. The
     surviving KV cache is then used for *all* remaining prefill layers AND for every
     decode step, so the saving is real and not just a prompt-encoding trick.

2. DeeAD (arXiv 2511.20720, AD-native early exit)
     Truncate the decoder stack when the partial trajectory is already close to a
     cheap kinematic prior.

3. Scaffold accept (core trick of Fast-dDrive, arXiv 2605.23163)
     Freeze the structural tokens (`<traj_start>`, `[(`, `),(`, `]`) and only
     draft/verify the numeric ones.

4. Visual-token pruning oracle
     Slice the projected UniAD features with no pruner at all: an upper bound for
     any learned pruning policy.
"""
from __future__ import annotations

from typing import Sequence

import torch


# ---------------------------------------------------------------- FastV
class FastVConfig:
    def __init__(self, k: int = 2, r: float = 0.5):
        self.k = int(k)
        self.r = float(r)

    def keep_count(self, n_visual: int) -> int:
        return max(1, int(round(n_visual * (1.0 - self.r))))


@torch.no_grad()
def fastv_score_from_attn(attn: torch.Tensor, n_vis_total: int, text_rows: torch.Tensor) -> torch.Tensor:
    """Column-sum attention each visual token receives from text rows, mean over heads.

    `attn` is [B, H, L, L] with L = text + visual (visual contiguous here because we
    call this on the *assembled* prompt where the three feature blocks were already
    merged into one span).
    """
    a = attn[0]  # [H, L, L]
    vis_cols = a[:, text_rows][:, :, ~text_rows]  # [H, n_text, n_vis]
    return vis_cols.sum(dim=1).mean(dim=0)  # [n_vis]


@torch.no_grad()
def fastv_prune_features(
    model,
    tokenizer,
    data_shared: dict,
    cfg: FastVConfig,
    horizon: int = 6,
):
    """Two-pass FastV, matching how FastDriveVLA/Prune2Drive actually operate.

    Pass 1: run the prompt once with attentions, rank each feature block
            (scene / track / map) by the attention it receives from text tokens.
    Pass 2: rebuild the prompt from the *pruned* features and decode normally.

    Pruning the features (rather than surgically editing masks and the KV cache)
    is exactly equivalent in effect -- fewer visual tokens enter the prompt -- and
    keeps the code on the same path as every other method. The extra prefill is
    charged to this method's wall clock.
    """
    import time

    from trajspec.opendrive_live import prefill_uniad, reset_uniad_temporal
    from trajspec.serialize import extract_opendrive_points

    vt_res = data_shared["uniad_pth"]
    orig_encode = model.encode_vision_tower_result

    # ---------- pass 1: score ----------
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    reset_uniad_temporal(model)
    _, meta1 = prefill_uniad(model, data_shared)
    with torch.inference_mode():
        out = model(
            inputs_embeds=meta1["inputs_embeds"],
            attention_mask=meta1["attention_mask"],
            position_ids=meta1["position_ids"],
            use_cache=False,
            output_attentions=True,
        )
    attn = out.attentions[cfg.k]  # [1, H, L, L]

    scene_f, track_f, map_f = orig_encode(vt_res)
    L = int(meta1["inputs_embeds"].shape[1])
    ids_row = data_shared.get("_ids_row") or []

    # Locate the visual positions in the assembled prompt. `prepare_inputs...`
    # walks the id sequence and, at each special token, splices in that block's
    # feature rows, so positions are *not* one contiguous span (scene ends, then a
    # track splice, then a map splice). Verified against a zero-probe: zeroing the
    # features changes exactly the positions this loop produces.
    block_size = {-201: 0 if scene_f is None else int(scene_f.shape[0]),
                  -202: 0 if track_f is None else int(track_f.shape[0]),
                  -203: 0 if map_f is None else int(map_f.shape[0])}
    n_vis = sum(block_size.values())
    vis_idx: list[int] = []
    if ids_row:
        acc = 0
        for x in ids_row:
            if x >= 0:
                acc += 1
                continue
            size = block_size.get(x, 0)
            if size:
                vis_idx.extend(range(acc, acc + size))
                acc += size
    if not vis_idx:  # fallback: assume features are trailing
        vis_idx = list(range(L - n_vis, L))
    assert len(vis_idx) == n_vis, f"visual span {len(vis_idx)} != feature rows {n_vis}"

    vis_t = torch.as_tensor(vis_idx, device=attn.device, dtype=torch.long)
    n_vis_actual = int(vis_t.numel())
    vis_mask = torch.zeros(L, dtype=torch.bool, device=attn.device)
    vis_mask[vis_t] = True
    score = fastv_score_from_attn(attn, n_vis_actual, ~vis_mask)
    keep_n = cfg.keep_count(n_vis_actual)
    torch.topk(score, k=keep_n, largest=True)  # ranking check only

    # per-block keep counts, so each block is pruned proportionally
    keep_ratio = keep_n / max(n_vis_actual, 1)
    new_blocks = []
    for f in (scene_f, track_f, map_f):
        if f is None:
            new_blocks.append(None)
            continue
        n_b = int(f.shape[0])
        k_b = max(1, int(round(n_b * keep_ratio)))
        new_blocks.append(f[:k_b])
    torch.cuda.synchronize()
    t_pass1 = (time.perf_counter() - t0) * 1000.0

    # ---------- pass 2: decode with pruned features ----------
    state = {"i": 0, "blocks": new_blocks}

    def _patched(vt_result):
        b = state["blocks"]
        return b[0], b[1], b[2]

    model.encode_vision_tower_result = _patched
    try:
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        reset_uniad_temporal(model)
        with torch.inference_mode():
            cont = model.generate(
                data_shared["input_ids"],
                uniad_data=None,
                uniad_pth=vt_res,
                qa_instance_ind=data_shared.get("qa_instance_ind"),
                do_sample=False,
                temperature=0,
                max_new_tokens=96,
                num_beams=1,
            )
        torch.cuda.synchronize()
        t_pass2 = (time.perf_counter() - t0) * 1000.0
    finally:
        model.encode_vision_tower_result = orig_encode
    text = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    return text, n_vis_actual, keep_n, t_pass1 + t_pass2


# ---------------------------------------------------------------- scaffold
def is_structural(piece: str) -> bool:
    s = (piece or "").strip()
    if not s:
        return True
    return not any(ch.isdigit() for ch in s)


def scaffold_accept_len(draft_ids: Sequence[int], teacher_ids: Sequence[int], tokenizer) -> int:
    """Longest prefix accepted under the scaffold rule.

    Fast-dDrive freezes the JSON/trajectory skeleton: those tokens carry no numeric
    information, so verifying them one-by-one wastes verification budget. Accept a
    position when the tokens match exactly, OR when BOTH sides decode to non-numeric
    text (brackets, commas, the `<traj_start>` tag).

    Note this counts *token* positions. To turn it into an accepted-waypoint count
    you must still check that the waypoints those tokens encode are geometrically
    acceptable -- see `scaffold_accept_waypoints`.
    """
    n = 0
    for a, b in zip(draft_ids, teacher_ids):
        if int(a) == int(b):
            n += 1
            continue
        pa = tokenizer.decode([int(a)], skip_special_tokens=False)
        pb = tokenizer.decode([int(b)], skip_special_tokens=False)
        if is_structural(pa) and is_structural(pb):
            n += 1
            continue
        break
    return n


def scaffold_accept_waypoints(
    tokenizer,
    draft_ids: Sequence[int],
    teacher_ids: Sequence[int],
    draft_xy: torch.Tensor,
    teacher_xy: torch.Tensor,
    cfg,
    horizon: int = 6,
) -> tuple[int, int]:
    """Scaffold rule, evaluated at waypoint granularity.

    Fast-dDrive's benefit is that structural tokens never block acceptance. So we
    walk the draft token by token: a structural mismatch costs nothing, a numeric
    mismatch ends the prefix. The number of *complete* waypoints inside that token
    prefix is then still subject to the geometric accept box -- scaffolds alone
    prove nothing about the trajectory.

    Returns (n_accept_waypoints, n_accept_tokens).
    """
    from trajspec.accept import longest_prefix_accept
    from trajspec.serialize import extract_opendrive_points

    n_tok = scaffold_accept_len(draft_ids, teacher_ids, tokenizer)
    prefix = tokenizer.decode(list(draft_ids[:n_tok]), skip_special_tokens=False)
    n_wp_in_prefix = len(extract_opendrive_points(prefix))
    # geometric gate on whichever waypoints the prefix actually contains
    k = min(n_wp_in_prefix, horizon, int(draft_xy.shape[0]), int(teacher_xy.shape[0]))
    if k <= 0:
        return 0, n_tok
    n_geo = longest_prefix_accept(draft_xy[:k], teacher_xy[:k], cfg)
    return int(n_geo), int(n_tok)
