"""Standalone AutoVLA runtime: Qwen2.5-VL + official action codebook. No navsim."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, "/home/xinhchen2/hhb/AutoVLA")
from models.action_tokenizer import ActionTokenizer  # noqa: E402


def _strip_ckpt_prefix(key: str) -> str:
    for pfx in ("autovla.vlm.", "autovla.", "drivevla.", "vlm."):
        if key.startswith(pfx):
            return key[len(pfx) :]
    return key


def remap_qwen25vl_v5(key: str) -> str:
    """Map AutoVLA (transformers 4.x Qwen2.5-VL) keys onto 5.x module names."""
    k = _strip_ckpt_prefix(key)
    if k.startswith("visual."):
        return "model." + k
    if k.startswith("model.layers.") or k.startswith("model.embed_tokens") or k.startswith("model.norm"):
        return "model.language_model." + k[len("model.") :]
    return k


class AutoVLARuntime:
    horizon = 10

    def __init__(
        self,
        qwen_path: str = "/data2/hf_overlap_models/Qwen2.5-VL-3B-Instruct",
        ckpt_path: str = "/data2/hf_overlap_models/AutoVLA/AutoVLA_PDMS_89.ckpt",
        codebook_path: str = "/home/xinhchen2/hhb/AutoVLA/codebook_cache/agent_vocab.pkl",
        device: str = "cuda",
        use_cot: bool = False,
        temperature: float | None = None,
        horizon: int | None = None,
    ):
        self.device = device
        self.use_cot = use_cot
        self.temperature = temperature
        if horizon is not None:
            self.horizon = int(horizon)
        self.video_conf = {"min_pixels": 109760, "max_pixels": 109760}
        self.action_start_id = 151665
        print(f"[AutoVLA] load VLM {qwen_path}", flush=True)
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(qwen_path)
        model_config = {
            "tokens": {"action_start_id": self.action_start_id},
            "codebook_cache_path": codebook_path,
        }
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, model_config=model_config)
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer))
        print(f"[AutoVLA] load ckpt {ckpt_path}", flush=True)
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        remapped = {remap_qwen25vl_v5(k): v for k, v in sd.items()}
        missing, unexpected = self.vlm.load_state_dict(remapped, strict=False)
        hid = getattr(self.vlm.config, "hidden_size", None)
        if hid is None:
            hid = self.vlm.config.text_config.hidden_size
        self.hidden_size = int(hid)
        print(
            f"[AutoVLA] loaded  missing={len(missing)} unused={len(unexpected)} "
            f"vocab={len(self.processor.tokenizer)} hidden={self.hidden_size}",
            flush=True,
        )
        if missing:
            print("[AutoVLA] missing sample:", missing[:8], flush=True)
        self.vlm.eval()
        self._n_logged = 0


    def get_prompt(self, input_features: dict):
        images = input_features["images"]
        min_pixels = self.video_conf["min_pixels"]
        max_pixels = self.video_conf["max_pixels"]
        cams = {}
        for name in ("front_camera", "front_left_camera", "front_right_camera"):
            cams[name] = []
            for i in range(4):
                img = images[name][i]
                if input_features.get("sensor_data_path"):
                    img = os.path.join(input_features["sensor_data_path"], img)
                cams[name].append(img)
        velocity = input_features["vehicle_velocity"]
        if isinstance(velocity, (list, np.ndarray)):
            velocity = float(np.sqrt(float(velocity[0]) ** 2 + float(velocity[1]) ** 2))
        acceleration = input_features["vehicle_acceleration"]
        if isinstance(acceleration, (list, np.ndarray)):
            acceleration = float(np.sqrt(float(acceleration[0]) ** 2 + float(acceleration[1]) ** 2))
        instruction = str(input_features["driving_command"]).lower()
        fc, fl, fr = cams["front_camera"], cams["front_left_camera"], cams["front_right_camera"]
        user_content = [
            {"type": "text", "text": "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."},
            {"type": "text", "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": min_pixels, "max_pixels": max_pixels, "video": [f"file://{p}" for p in fc]},
            {"type": "text", "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": min_pixels, "max_pixels": max_pixels, "video": [f"file://{p}" for p in fl]},
            {"type": "text", "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": min_pixels, "max_pixels": max_pixels, "video": [f"file://{p}" for p in fr]},
            {"type": "text", "text": (
                f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s². "
                f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."
            )},
        ]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": (
                "You are an Advanced Driver Assistance and Full Self-Driving System. "
                "You will be provided with video observations from the ego vehicle’s surrounding cameras, along with the vehicle’s current dynamic states. "
                "Your task is to predict the most appropriate driving action for the next five seconds."
            )}]},
            {"role": "user", "content": user_content},
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True, enable_thinking=False
            )
        except TypeError:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    def _to_dev(self, inputs):
        out = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    @torch.inference_mode()
    def prefill_hidden(self, features: dict) -> torch.Tensor:
        inputs = self._to_dev(self.get_prompt(features))
        out = self.vlm(**inputs, use_cache=False, output_hidden_states=True)
        return out.hidden_states[-1][:, -1].float().cpu()[0]

    def _decode_actions(self, token_ids: torch.Tensor) -> np.ndarray:
        act = token_ids[token_ids >= self.action_start_id]
        if len(act) == 0:
            return np.zeros((6, 2), dtype=np.float32)
        act = act[: self.horizon]
        traj = self.action_tokenizer.decode_token_ids_to_trajectory(act.cpu())
        if traj is None or (hasattr(traj, "__len__") and len(traj) == 0):
            return np.zeros((6, 2), dtype=np.float32)
        arr = torch.as_tensor(traj)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape[0] > 1:
            arr = arr[1:]
        xy = arr[:, :2].float().numpy().astype(np.float32)
        if xy.shape[0] < self.horizon:
            last = xy[-1:] if xy.shape[0] else np.zeros((1, 2), dtype=np.float32)
            xy = np.concatenate([xy, np.repeat(last, self.horizon - xy.shape[0], axis=0)], axis=0)
        return xy[: self.horizon]

    @torch.inference_mode()
    def predict(self, features: dict) -> np.ndarray:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _StopAfterActions(StoppingCriteria):
            def __init__(self, start_id: int, n: int, prompt_len: int):
                super().__init__()
                self.start_id = start_id
                self.n = n
                self.prompt_len = prompt_len

            def __call__(self, input_ids, scores, **kwargs):
                gen = input_ids[0, self.prompt_len :]
                return bool((gen >= self.start_id).sum().item() >= self.n)

        inputs = self.get_prompt(features)
        model_inputs = self._to_dev(inputs)
        prompt_len = int(inputs["input_ids"].shape[1])
        gen_kwargs = dict(
            max_new_tokens=192,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList(
                [_StopAfterActions(self.action_start_id, self.horizon, prompt_len)]
            ),
        )
        if self.temperature is not None and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(self.temperature), top_p=1.0)
        else:
            gen_kwargs.update(do_sample=False)
        outputs = self.vlm.generate(**model_inputs, **gen_kwargs)
        trimmed = outputs[0, prompt_len:]
        if trimmed.numel() and trimmed[-1].item() == self.processor.tokenizer.eos_token_id:
            trimmed = trimmed[:-1]
        n_act = int((trimmed >= self.action_start_id).sum().item()) if trimmed.numel() else 0
        if self._n_logged < 3:
            text = self.processor.tokenizer.decode(trimmed, skip_special_tokens=False)
            print(
                f"[AutoVLA] gen tokens={int(trimmed.numel())} action_tokens={n_act} text={text[:280]!r}",
                flush=True,
            )
            self._n_logged += 1
        return self._decode_actions(trimmed)

    @property
    def n_action_bins(self) -> int:
        return int(self.action_tokenizer.code_book.shape[0])

    @staticmethod
    def _cache_seq_len(past) -> int:
        if past is None:
            return 0
        if hasattr(past, "get_seq_length"):
            n = past.get_seq_length()
            if n:
                return int(n)
        if hasattr(past, "key_cache") and past.key_cache:
            return int(past.key_cache[0].shape[-2])
        if hasattr(past, "layers") and past.layers:
            layer = past.layers[0]
            keys = getattr(layer, "keys", None)
            if keys is None:
                keys = getattr(layer, "key_cache", None)
            if keys is not None:
                return int(keys.shape[-2])
        return int(past[0][0].shape[-2])

    def _clone_past(self, past):
        if past is None:
            return None
        import copy

        return copy.deepcopy(past)

    def _greedy_action(self, logits: torch.Tensor) -> int:
        from trajspec.live_spec import greedy_action_id

        return greedy_action_id(logits.float().view(-1), self.action_start_id, self.n_action_bins)

    def _forward_tokens(self, token_ids: torch.Tensor, past):
        """Continue from prefill KV. Do not pass a full-length mask: Qwen2.5-VL mRoPE
        would then build position_ids of length cache+T while embeds are only T.
        """
        return self.vlm(
            input_ids=token_ids,
            past_key_values=past,
            use_cache=True,
        )

    def _think_until_action(self, logits: torch.Tensor, past, max_new: int = 160):
        """Greedy full-vocab decode until the next token would be an action (not consumed)."""
        n_think = 0
        eos = self.processor.tokenizer.eos_token_id
        for _ in range(max_new):
            tid = int(logits.float().view(-1).argmax().item())
            if tid >= self.action_start_id:
                break
            if eos is not None and tid == eos:
                break
            step = torch.tensor([[tid]], device=self.device)
            out = self._forward_tokens(step, past)
            logits = out.logits[0, -1].float()
            past = out.past_key_values
            n_think += 1
        return logits, past, n_think

    ANSWER_PREFIX = "<answer>\nThe final output action is: "

    def _answer_prefix_ids(self) -> torch.Tensor:
        ids = self.processor.tokenizer.encode(self.ANSWER_PREFIX, add_special_tokens=False)
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    def _force_answer_prefix(self, logits: torch.Tensor, past):
        """Teacher-force the official use_cot=false assistant prefix, then action tokens."""
        prefix = self._answer_prefix_ids()
        out = self._forward_tokens(prefix, past)
        return out.logits[0, -1].float(), out.past_key_values, int(prefix.shape[1])

    @torch.inference_mode()
    def spec_propose(
        self,
        features: dict,
        draft_head,
        extra: torch.Tensor,
        speed: torch.Tensor,
        horizon: int | None = None,
        skip_think: bool = False,
    ) -> dict:
        """Prefill → draft MLP → one teacher-force verify. No accept / tail yet."""
        import time

        from trajspec.action_quant import action_ids_to_xy, xy_to_action_ids
        from trajspec.live_spec import verify_ids_from_logits

        horizon = int(horizon or self.horizon)

        def _ms(fn):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0, out

        cb = self.action_tokenizer.code_book
        inputs = self._to_dev(self.get_prompt(features))
        prompt_len = int(inputs["input_ids"].shape[1])

        def _prefill():
            return self.vlm(**inputs, use_cache=True, output_hidden_states=True)

        t_pre, pre = _ms(_prefill)
        hidden = pre.hidden_states[-1][:, -1].float()
        prefill_logits = pre.logits[0, -1].float()
        prefill_past = pre.past_key_values
        cache_len = self._cache_seq_len(prefill_past)
        if cache_len <= 0:
            cache_len = prompt_len

        extra_b = extra.to(device=self.device, dtype=torch.float32).view(1, -1)
        speed_b = speed.to(device=self.device, dtype=torch.float32).view(-1)

        def _draft():
            return draft_head(hidden, extra=extra_b, speed=speed_b)[0, :horizon]

        t_draft, draft = _ms(_draft)
        draft_xy = draft.float().cpu()
        draft_ids = xy_to_action_ids(draft_xy, cb)
        tok = torch.tensor(
            [[self.action_start_id + i for i in draft_ids]],
            device=self.device,
            dtype=torch.long,
        )

        def _think():
            if skip_think:
                return self._force_answer_prefix(prefill_logits, prefill_past)
            return self._think_until_action(prefill_logits, prefill_past)

        t_think, (think_logits, think_past, n_think) = _ms(_think)

        def _verify():
            return self._forward_tokens(tok, self._clone_past(think_past))

        t_ver, ver = _ms(_verify)
        force_logits = ver.logits[0].float()
        teacher_ids = verify_ids_from_logits(
            think_logits, force_logits, draft_ids, self.action_start_id, self.n_action_bins
        )
        teacher_xy = action_ids_to_xy(teacher_ids, cb)
        if teacher_xy.shape[0] < horizon:
            pad_n = horizon - int(teacher_xy.shape[0])
            last = teacher_xy[-1:] if teacher_xy.shape[0] else torch.zeros(1, 2)
            teacher_xy = torch.cat([teacher_xy, last.repeat(pad_n, 1)], dim=0)
        return {
            "horizon": horizon,
            "cb": cb,
            "draft_xy": draft_xy,
            "draft_ids": draft_ids,
            "teacher_ids": teacher_ids,
            "teacher_xy": teacher_xy,
            "think_past": think_past,
            "speed_ms": float(speed_b.view(-1)[0].item()),
            "t_prefill_ms": t_pre,
            "t_think_ms": t_think,
            "t_draft_ms": t_draft,
            "t_verify_ms": t_ver,
            "n_think": int(n_think),
            "skip_think": bool(skip_think),
        }

    @torch.inference_mode()
    def spec_commit(self, pack: dict, accept_cfg, accept_mode: str = "geometric", id_radius: int = 16):
        """Accept + optional serial tail (or KERV bicycle fill) from a shared propose pack."""
        import time

        from trajspec.action_quant import action_ids_to_xy
        from trajspec.sd_baselines import KERV_LITE, commit_spec, n_accept_autovla

        horizon = int(pack["horizon"])
        draft_xy = pack["draft_xy"]
        draft_ids = pack["draft_ids"]
        teacher_ids = pack["teacher_ids"]
        teacher_xy = pack["teacher_xy"]
        n_acc = n_accept_autovla(
            accept_mode,
            draft_xy[:horizon],
            teacher_xy[:horizon],
            draft_ids,
            teacher_ids,
            accept_cfg,
            id_radius,
        )
        k = len(draft_ids)

        def _ms(fn):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0, out

        t_tail = 0.0
        n_serial = 0
        if str(accept_mode) == KERV_LITE:
            spec, n_acc, n_serial = commit_spec(
                draft_xy, teacher_xy, n_acc, KERV_LITE, speed_ms=pack["speed_ms"], horizon=horizon
            )
            xy = spec
            committed = list(draft_ids[:n_acc])
        else:
            committed = list(draft_ids[:n_acc])
            if n_acc < k:
                committed.append(teacher_ids[n_acc])
            if len(committed) < horizon:
                need = horizon - len(committed)

                def _tail():
                    ids = list(committed)
                    cur = torch.tensor(
                        [[self.action_start_id + i for i in ids]],
                        device=self.device,
                        dtype=torch.long,
                    )
                    out = self._forward_tokens(cur, self._clone_past(pack["think_past"]))
                    logits = out.logits[0, -1].float()
                    past = out.past_key_values
                    for _ in range(need):
                        aid = self._greedy_action(logits)
                        ids.append(aid)
                        step = torch.tensor([[self.action_start_id + aid]], device=self.device)
                        out = self._forward_tokens(step, past)
                        logits = out.logits[0, -1].float()
                        past = out.past_key_values
                    return ids

                t_tail, committed = _ms(_tail)
                n_serial = need
            xy = action_ids_to_xy(committed[:horizon], pack["cb"])
            # Seam repair: the committed action ids decode to a path that jumps
            # at the reject index. Blend into the decoded suffix along the
            # *decoded* prefix's own velocity (never the unquantized draft_xy,
            # which would overwrite already-quantized accepted points).
            if 0 < n_acc < min(horizon, xy.shape[0]):
                from trajspec.sd_baselines import seam_blend

                xy = seam_blend(xy, xy, n_acc, horizon=min(horizon, xy.shape[0]))
            xy = torch.as_tensor(xy, dtype=torch.float32).cpu()
        if xy.shape[0] < horizon:
            pad = xy[-1:].repeat(horizon - xy.shape[0], 1) if xy.shape[0] else torch.zeros(horizon, 2)
            xy = torch.cat([xy, pad], dim=0)[:horizon]
        t_pre, t_think, t_draft, t_ver = pack["t_prefill_ms"], pack["t_think_ms"], pack["t_draft_ms"], pack["t_verify_ms"]
        info = {
            "accept_mode": str(accept_mode),
            "n_accept": int(n_acc),
            "draft_ids": draft_ids,
            "teacher_verify_ids": teacher_ids,
            "committed_ids": list(committed)[:horizon],
            "t_prefill_ms": round(t_pre, 2),
            "t_think_ms": round(t_think, 2),
            "t_draft_ms": round(t_draft, 2),
            "t_verify_ms": round(t_ver, 2),
            "t_tail_ms": round(t_tail, 2),
            "t_spec_e2e_ms": round(t_pre + t_think + t_draft + t_ver + t_tail, 2),
            "n_think_tokens": 0 if pack["skip_think"] else int(pack["n_think"]),
            "n_prefix_tokens": int(pack["n_think"]) if pack["skip_think"] else 0,
            "skip_think": bool(pack["skip_think"]),
            "n_serial_tail": int(n_serial),
            "spec_decode_steps": 1 + int(n_serial),
            "draft_xy": draft_xy[:horizon].tolist(),
            "teacher_verify_xy": teacher_xy[:horizon].tolist() if teacher_xy.numel() else [],
        }
        info = {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in info.items()}
        return xy.numpy().astype("float32") if hasattr(xy, "numpy") else xy.cpu().numpy().astype("float32"), info

    @torch.inference_mode()
    def spec_generate(
        self,
        features: dict,
        draft_head,
        extra: torch.Tensor,
        speed: torch.Tensor,
        accept_cfg,
        horizon: int | None = None,
        skip_think: bool = False,
        accept_mode: str = "geometric",
        id_radius: int = 16,
    ):
        """Prefill → draft MLP → one teacher-force verify → serial action tail.

        skip_think=True is official trajectory-only (use_cot=false): teacher-force
        the <answer> prefix and never decode CoT think tokens.

        Returns (xy [H,2], info dict with real CUDA-sync timings).
        """
        pack = self.spec_propose(features, draft_head, extra, speed, horizon=horizon, skip_think=skip_think)
        return self.spec_commit(pack, accept_cfg, accept_mode=accept_mode, id_radius=id_radius)

    @torch.inference_mode()
    def ar_action_generate(self, features: dict, horizon: int | None = None, skip_think: bool = False):
        """Serial greedy over the action vocab.

        skip_think=False: decode CoT think tokens first (keep-think).
        skip_think=True: official trajectory-only — force <answer> prefix, no CoT.
        """
        import time

        from trajspec.action_quant import action_ids_to_xy

        horizon = int(horizon or self.horizon)

        def _ms(fn):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0, out

        cb = self.action_tokenizer.code_book
        inputs = self._to_dev(self.get_prompt(features))

        def _prefill():
            return self.vlm(**inputs, use_cache=True, output_hidden_states=True)

        t_pre, pre = _ms(_prefill)
        hidden = pre.hidden_states[-1][:, -1].float().cpu()[0]
        logits = pre.logits[0, -1].float()
        past = pre.past_key_values

        def _think():
            if skip_think:
                return self._force_answer_prefix(logits, past)
            return self._think_until_action(logits, past)

        t_think, (logits, past, n_think) = _ms(_think)
        ids: list[int] = []
        t_dec = 0.0
        for _ in range(horizon):
            def _step(logits=logits, past=past):
                aid = self._greedy_action(logits)
                step = torch.tensor([[self.action_start_id + aid]], device=self.device)
                out = self._forward_tokens(step, past)
                return aid, out

            dt, (aid, out) = _ms(_step)
            t_dec += dt
            ids.append(aid)
            logits = out.logits[0, -1].float()
            past = out.past_key_values
        xy = action_ids_to_xy(ids, cb)
        if xy.shape[0] < horizon:
            pad = xy[-1:].repeat(horizon - xy.shape[0], 1) if xy.shape[0] else torch.zeros(horizon, 2)
            xy = torch.cat([xy, pad], dim=0)[:horizon]
        info = {
            "t_prefill_ms": round(t_pre, 2),
            "t_think_ms": 0.0 if skip_think else round(t_think, 2),
            "t_prefix_ms": round(t_think, 2) if skip_think else 0.0,
            "t_decode_ms": round((t_think if skip_think else 0.0) + t_dec, 2),
            "t_action_ms": round(t_dec, 2),
            "t_e2e_ms": round(t_pre + t_think + t_dec, 2),
            "n_think_tokens": 0 if skip_think else int(n_think),
            "n_prefix_tokens": int(n_think) if skip_think else 0,
            "skip_think": bool(skip_think),
            "action_ids": ids,
            "hidden": hidden,
        }
        return xy.numpy().astype("float32"), info


class AutoVLATeacher:
    """Compatibility stub for the toy two-VLA script."""

    horizon = 10
    hidden_size = 2048

    def __init__(self, checkpoint: str | None = None, **kwargs):
        self.checkpoint = checkpoint
        self.runtime = None

    def available(self) -> bool:
        return bool(self.checkpoint) and Path(self.checkpoint).exists()

    def load(self) -> None:
        self.runtime = AutoVLARuntime(ckpt_path=self.checkpoint)

    def eval(self):
        return self
