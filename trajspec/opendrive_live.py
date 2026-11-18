"""Live OpenDriveVLA specdec: one teacher-force of the draft [(x,y),...] string."""
from __future__ import annotations

import time
from typing import Any

import torch

from trajspec.sd_baselines import GEOMETRIC, KERV_LITE, commit_spec, n_accept_opendrive
from trajspec.serialize import (
    TRAJ_END,
    TRAJ_START,
    extract_opendrive_points,
    format_opendrive_answer,
    ego_to_od_xy,
    od_to_ego_xy,
    parse_opendrive_traj,
)


def _ms(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def _cache_seq_len(past) -> int:
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        n = past.get_seq_length()
        if n:
            return int(n)
    if hasattr(past, "key_cache") and past.key_cache:
        return int(past.key_cache[0].shape[-2])
    return int(past[0][0].shape[-2])


def _clone_past(past):
    if past is None:
        return None
    import copy

    return copy.deepcopy(past)


def _waypoint_prefix_len(tokenizer, xy_od: torch.Tensor, n_keep: int) -> int:
    """Token count of `<traj_start>[(p0),...,(p_{n_keep-1})` (no closing) for n_keep>0."""
    if n_keep <= 0:
        ids = tokenizer(TRAJ_START, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)
    pts = xy_od[:n_keep].tolist()
    inner = ",".join(f"({x:.2f},{y:.2f})" for x, y in (p[:2] for p in pts))
    text = f"{TRAJ_START}[{inner}"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def reset_uniad_temporal(model) -> None:
    """Clear UniAD's persistent temporal state before an independent frame.

    `UniADTrack.prev_frame_info` persists `prev_bev` / `prev_pos` / `scene_token`
    across forward calls, and is only cleared on a scene switch. Evaluating by
    index (stride sampling, a non-contiguous frame list, or running generate()
    before a re-prefill) therefore feeds the tower the wrong BEV context.

    Measured impact on OpenDriveVLA-0.5B: the same frame prefilled twice around a
    `generate()` call differs by 0.65 in logit space, which flipped the decoded
    token chain after ~5-20 tokens. Every AR-vs-spec comparison is invalid unless
    this state is reset per frame.
    """
    vt = model.get_vision_tower() if hasattr(model, "get_vision_tower") else None
    # Actual path in OpenDriveVLA:
    #   LlavaQwen.model.vision_tower -> UniadTrackMapVisionTower
    #     .vision_tower -> UniadTrackMapModel
    #       .vision_model -> UniADTrack (owns prev_frame_info + track_base)
    # Walk a few levels down instead of hard-coding, so this survives refactors.
    stack = [vt, getattr(vt, "vision_tower", None), getattr(getattr(vt, "vision_tower", None), "vision_model", None)]
    seen = 0
    for obj in stack:
        if obj is None:
            continue
        info = getattr(obj, "prev_frame_info", None)
        if isinstance(info, dict):
            info["prev_bev"] = None
            info["scene_token"] = None
            info["prev_pos"] = 0
            info["prev_angle"] = 0
            seen += 1
        # Track ids are assigned incrementally and index per-query embeddings, so a
        # stale max_obj_id also shifts the features.
        tb = getattr(obj, "track_base", None)
        if tb is not None and hasattr(tb, "max_obj_id"):
            tb.max_obj_id = 0
    if seen == 0:
        # Fall back to a full module walk if the layout changed.
        for _name, mod in model.named_modules():
            info = getattr(mod, "prev_frame_info", None)
            if isinstance(info, dict):
                info["prev_bev"] = None
                info["scene_token"] = None
                info["prev_pos"] = 0
                info["prev_angle"] = 0
            tb = getattr(mod, "track_base", None)
            if tb is not None and hasattr(tb, "max_obj_id"):
                tb.max_obj_id = 0


@torch.inference_mode()
def prefill_uniad(model, data) -> tuple[Any, dict]:
    ids = data["input_ids"]
    (
        _,
        position_ids,
        attention_mask,
        _,
        inputs_embeds,
        _,
    ) = model.prepare_inputs_labels_for_multimodal_uniad_vlm(
        ids,
        None,
        None,
        None,
        None,
        None,
        uniad_data=data.get("uniad_data"),
        uniad_pth=data.get("uniad_pth"),
        qa_instance_ind=data.get("qa_instance_ind"),
    )
    out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        output_hidden_states=True,
    )
    info = {
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "inputs_embeds": inputs_embeds,
        "prompt_len": int(inputs_embeds.shape[1]),
    }
    return out, info


def _continue(
    model,
    token_ids: torch.Tensor,
    past,
    cache_len: int,
    attn_dtype,
):
    t = int(token_ids.shape[1])
    device = token_ids.device
    embeds = model.get_model().embed_tokens(token_ids)
    attn = torch.ones(1, cache_len + t, device=device, dtype=attn_dtype)
    pos = torch.arange(cache_len, cache_len + t, device=device).unsqueeze(0)
    return model(
        inputs_embeds=embeds,
        attention_mask=attn,
        position_ids=pos,
        past_key_values=past,
        use_cache=True,
    )


def _greedy_id(logits: torch.Tensor) -> int:
    return int(logits.float().view(-1).argmax().item())


class TeacherDecoder:
    """Greedy decode that matches the official `generate()` exactly.

    `generation_config.json` ships `repetition_penalty = 1.1`. `generate()`
    applies it to every step even with `do_sample=False`, but the SD verify path
    originally used a bare argmax. That made the "teacher" the verify path
    compares against a *different* decoder from the one that produced the
    reference trajectory: on 10/10 frames the two chains diverged within the
    first 20 tokens, giving 0.41 m mean ADE between them, and a full-reject frame
    (n_accept = 0) still differed from AR greedy by 0.15 m.

    This processor reproduces the HF op faithfully: the penalty is applied once
    per token across the *whole* history (prompt + generated), not just the new
    tokens.
    """

    def __init__(self, model, penalty: float | None = None):
        gc = getattr(model, "generation_config", None)
        if penalty is None:
            penalty = float(getattr(gc, "repetition_penalty", 1.0) or 1.0)
        self.penalty = float(penalty)
        self._proc = None
        if self.penalty != 1.0:
            from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

            self._proc = RepetitionPenaltyLogitsProcessor(self.penalty)

    def __call__(self, logits: torch.Tensor, history_ids) -> int:
        lg = logits.float().view(1, -1).clone()
        if self._proc is not None and len(history_ids):
            # The multimodal prompt contains negative placeholder ids
            # (IMAGE/SCENE/TRACK/MAP = -200..-204) which are not vocab indices.
            hist = torch.as_tensor([int(t) for t in history_ids if int(t) >= 0],
                                   device=lg.device, dtype=torch.long)
            if hist.numel():
                lg = self._proc(hist.unsqueeze(0), lg)
        return int(lg.view(-1).argmax().item())


@torch.inference_mode()
def spec_propose_opendrive(
    model,
    tokenizer,
    data,
    draft_head,
    extra: torch.Tensor,
    speed: torch.Tensor,
    horizon: int = 6,
):
    device = extra.device if extra.is_cuda else next(model.parameters()).device
    attn_dtype = torch.long

    def _pre():
        return prefill_uniad(model, data)

    t_pre, (pre, meta) = _ms(_pre)
    hidden = pre.hidden_states[-1][:, -1].float()
    prefill_logits = pre.logits[0, -1].float()
    prefill_past = pre.past_key_values
    cache_len = _cache_seq_len(prefill_past) or int(meta["prompt_len"])
    if meta["attention_mask"] is not None:
        attn_dtype = meta["attention_mask"].dtype

    extra_b = extra.to(device=device, dtype=torch.float32).view(1, -1)
    speed_b = speed.to(device=device, dtype=torch.float32).view(-1)

    def _draft():
        return draft_head(hidden, extra=extra_b, speed=speed_b)[0, :horizon]

    t_draft, draft_ego = _ms(_draft)
    draft_ego = draft_ego.float().cpu()
    draft_od = ego_to_od_xy(draft_ego)
    draft_text = f"{TRAJ_START}{format_opendrive_answer(draft_od)}{TRAJ_END}"
    draft_ids = tokenizer(draft_text, add_special_tokens=False)["input_ids"]
    if isinstance(draft_ids[0], list):
        draft_ids = draft_ids[0]
    tok = torch.tensor([draft_ids], device=device, dtype=torch.long)

    def _verify():
        return _continue(model, tok, _clone_past(prefill_past), cache_len, attn_dtype)

    t_ver, ver = _ms(_verify)
    force_logits = ver.logits[0].float()
    # Match generate(): repetition_penalty is active even at do_sample=False.
    dec = TeacherDecoder(model)
    prompt_hist = [int(t) for t in data["input_ids"][0].tolist()]
    teacher_tok = [dec(prefill_logits, prompt_hist)]
    for i in range(min(len(draft_ids), force_logits.shape[0]) - 1):
        teacher_tok.append(dec(force_logits[i], prompt_hist + teacher_tok))
    teacher_text = tokenizer.decode(teacher_tok, skip_special_tokens=False)
    teacher_od = parse_opendrive_traj(teacher_text, horizon=horizon)
    teacher_ego = od_to_ego_xy(teacher_od)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "horizon": horizon,
        "device": device,
        "attn_dtype": attn_dtype,
        "cache_len": cache_len,
        "prefill_logits": prefill_logits,
        "prefill_past": prefill_past,
        "draft_ego": draft_ego,
        "draft_od": draft_od,
        "draft_text": draft_text,
        "draft_ids": draft_ids,
        "teacher_tok": teacher_tok,
        "teacher_text": teacher_text,
        "teacher_ego": teacher_ego,
        "speed_ms": float(speed_b.view(-1)[0].item()),
        "prompt_ids": [int(t) for t in data["input_ids"][0].tolist()],
        # Kept so a full-reject frame can rerun the *official* generate() rather than
        # a hand-rolled loop. A manual loop is not bit-equivalent (measured 0.09 m ADE
        # gap on n_accept==0 frames), which would make the table look inconsistent.
        "raw_data": data,
        "t_prefill_ms": t_pre,
        "t_draft_ms": t_draft,
        "t_verify_ms": t_ver,
    }


@torch.inference_mode()
def spec_commit_opendrive(pack: dict, accept_cfg, accept_mode: str = "geometric", max_new: int = 96,
                          min_accept: int = 2):
    tokenizer = pack["tokenizer"]
    model = pack["model"]
    horizon = int(pack["horizon"])
    draft_ego = pack["draft_ego"]
    teacher_ego = pack["teacher_ego"]
    draft_ids = pack["draft_ids"]
    teacher_tok = pack["teacher_tok"]
    n_acc = n_accept_opendrive(
        accept_mode,
        draft_ego[:horizon],
        teacher_ego[:horizon],
        draft_ids,
        teacher_tok,
        tokenizer,
        accept_cfg,
        horizon=horizon,
    )
    t_tail = 0.0
    n_serial = 0
    splice_acc = int(n_acc)
    draft_text = pack["draft_text"]
    final_text = draft_text
    if str(accept_mode) == KERV_LITE:
        spec_ego, n_acc, n_serial = commit_spec(
            draft_ego, teacher_ego, n_acc, KERV_LITE, speed_ms=pack["speed_ms"], horizon=horizon
        )
    elif n_acc >= horizon:
        spec_ego = draft_ego[:horizon]
    else:
        # `min_accept` changes both the splice and the tail's condition:
        #   * splicing from a 1-waypoint prefix drives the tail from a single
        #     displacement vector (~+0.99 m ADE over 8 frames, measured), and
        #   * decoding the tail *conditioned on* that same 1-waypoint prefix
        #     propagates the error, so zeroing only the splice is not enough.
        # Below `min_accept` we therefore fall back to a full reject: decode the
        # tail from scratch and keep the teacher suffix. Such frames get no
        # speedup, which is the honest cost of refusing a bad draft.
        full_reject = n_acc < min_accept
        splice_acc = 0 if full_reject else n_acc
        keep = 0 if full_reject else _waypoint_prefix_len(tokenizer, pack["draft_od"], n_acc)
        prefix = list(draft_ids[:keep])
        device = pack["device"]
        cache_len = pack["cache_len"]
        attn_dtype = pack["attn_dtype"]

        if full_reject and pack.get("raw_data") is not None:
            # Nothing was accepted, so the correct output *is* the teacher trajectory.
            # Call the official generate() instead of re-deriving it: the hand-rolled
            # loop below is not bit-equivalent (0.09 m ADE gap measured), and a
            # full-reject frame must reproduce AR greedy exactly by construction.
            raw = pack["raw_data"]

            def _official():
                with torch.inference_mode():
                    cont = model.generate(
                        raw["input_ids"],
                        uniad_data=raw.get("uniad_data"),
                        uniad_pth=raw.get("uniad_pth"),
                        qa_instance_ind=raw.get("qa_instance_ind"),
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=max_new,
                        num_beams=1,
                    )
                return tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

            t_tail, gen_text = _ms(_official)
            final_text = gen_text
            spec_ego = od_to_ego_xy(parse_opendrive_traj(gen_text, horizon=horizon))
            n_serial = 0
            prefix = []
            keep = 0
        else:
            prefix = list(draft_ids[:keep])

            def _tail():
                ids = list(prefix)
                if ids:
                    cur = torch.tensor([ids], device=device, dtype=torch.long)
                    out = _continue(model, cur, _clone_past(pack["prefill_past"]), cache_len, attn_dtype)
                    logits = out.logits[0, -1].float()
                    past = out.past_key_values
                    filled = cache_len + len(ids)
                else:
                    logits = pack["prefill_logits"]
                    past = _clone_past(pack["prefill_past"])
                    filled = cache_len
                # Same decoder as generate(): repetition_penalty is part of the teacher.
                dec = TeacherDecoder(model)
                prompt_ids = pack.get("prompt_ids") or []
                eos = getattr(tokenizer, "eos_token_id", None)
                eos_set = set(eos) if isinstance(eos, (list, tuple)) else (
                    {eos} if eos is not None else set())
                for _ in range(max_new):
                    tid = dec(logits, list(prompt_ids) + ids)
                    if tid in eos_set:
                        break
                    ids.append(tid)
                    step = torch.tensor([[tid]], device=device, dtype=torch.long)
                    out = _continue(model, step, past, filled, attn_dtype)
                    filled += 1
                    logits = out.logits[0, -1].float()
                    past = out.past_key_values
                    text = tokenizer.decode(ids, skip_special_tokens=False)
                    n_pts = len(extract_opendrive_points(text))
                    if n_pts >= horizon and "]" in text.replace("<traj_end>", ""):
                        break
                    if n_pts >= horizon and text.rstrip().endswith("]"):
                        break
                return ids

            t_tail, committed_ids = _ms(_tail)
            n_serial = max(len(committed_ids) - keep, 0)
            final_text = tokenizer.decode(committed_ids, skip_special_tokens=False)
            spec_od = parse_opendrive_traj(final_text, horizon=horizon)
            spec_ego = od_to_ego_xy(spec_od)

    t_pre, t_draft, t_ver = pack["t_prefill_ms"], pack["t_draft_ms"], pack["t_verify_ms"]
    info = {
        "accept_mode": str(accept_mode),
        "n_accept": int(n_acc),
        "draft_text": draft_text,
        "teacher_verify_text": pack["teacher_text"],
        "final_text": final_text if str(accept_mode) != KERV_LITE else "[kerv_lite]",
        "t_prefill_ms": round(t_pre, 2),
        "t_draft_ms": round(t_draft, 2),
        "t_verify_ms": round(t_ver, 2),
        "t_tail_ms": round(t_tail, 2),
        "t_spec_e2e_ms": round(t_pre + t_draft + t_ver + t_tail, 2),
        "n_draft_tokens": len(draft_ids),
        "n_serial_tail": int(n_serial),
        "spec_decode_steps": 1 if n_acc >= horizon else 1 + max(horizon - n_acc - 1, 0),
        "draft_xy_ego": draft_ego[:horizon].tolist(),
        "teacher_verify_xy_ego": teacher_ego[:horizon].tolist(),
    }
    if torch.is_tensor(spec_ego):
        spec_np = spec_ego.numpy().astype("float32")
    else:
        spec_np = spec_ego
    # Seam repair: blend the accepted draft prefix into the teacher suffix along
    # the draft's own velocity so the splice is continuous (no extra teacher call).
    # When fewer than `min_accept` waypoints were accepted the blend is skipped, so
    # the output is the pure teacher tail (splice_acc == 0).
    if 0 < splice_acc < int(spec_np.shape[0]):
        from trajspec.sd_baselines import seam_blend

        kk = min(int(spec_np.shape[0]), horizon)
        spec_np = seam_blend(draft_ego[:kk], torch.as_tensor(spec_np[:kk]), splice_acc, horizon=kk).numpy().astype("float32")
    return spec_np, info


@torch.inference_mode()
def spec_generate_opendrive(
    model,
    tokenizer,
    data,
    draft_head,
    extra: torch.Tensor,
    speed: torch.Tensor,
    accept_cfg,
    horizon: int = 6,
    max_new: int = 96,
    accept_mode: str = "geometric",
):
    pack = spec_propose_opendrive(model, tokenizer, data, draft_head, extra, speed, horizon=horizon)
    return spec_commit_opendrive(pack, accept_cfg, accept_mode=accept_mode, max_new=max_new)
