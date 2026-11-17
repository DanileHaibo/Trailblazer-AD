"""Stanford ASL nuScenes trajectory-only VLA (Qwen3-VL-4B).

Trained to emit Agent-Driver / UniAD waypoints:
  X right (perpendicular), Y forward (along heading), 6 points / 3 s.
We convert to ego (+X forward, +Y left) so the Orion draft bicycle prior matches.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SYSTEM = """**Autonomous Driving Planner**
Role: You're an autonomous vehicle's brain. Given an image of the current scenario, plan a 3-second safe trajectory to avoid obstacles.

Context:
- Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0). Units: meters.
- Goal: Plan a 3-second route using 6 waypoints (0.5s intervals).

Task:
- Based on inputs, plan a safe, feasible 3-second trajectory of 6 waypoints.

Output:
Planned Trajectory:
[(x1,y1), (x2,y2), ... , (x6,y6)]
"""

TRAFFIC = """
*****Traffic Rules:*****
- Avoid collision with other objects.
- Always drive on drivable regions.
- Avoid driving on occupied regions.
- Pay attention to your ego-states and historical trajectory when planning.
- Maintain a safe distance from the objects in front of you.
"""

CKPT_DEFAULT = "/data2/hf_overlap_models/nuscenes-waypoints-model"


def ego_to_od(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    return np.stack([-xy[:, 1], xy[:, 0]], axis=-1)


def od_to_ego(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    return np.stack([xy[:, 1], -xy[:, 0]], axis=-1)


def parse_planned_traj(text: str, horizon: int = 6) -> np.ndarray:
    if "Planned Trajectory:" in text:
        text = text.split("Planned Trajectory:", 1)[1]
    pairs = re.findall(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)", text)
    if len(pairs) < 2:
        pairs = re.findall(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", text)
    coords = [(float(x), float(y)) for x, y in pairs]
    if not coords:
        return np.zeros((horizon, 2), dtype=np.float32)
    if len(coords) < horizon:
        coords = coords + [coords[-1]] * (horizon - len(coords))
    return np.asarray(coords[:horizon], dtype=np.float32)


def ego_prompt(sample: dict) -> str:
    """Rebuild Agent-Driver ego-state text in UniAD (x right, y forward)."""
    hist = np.asarray(sample.get("hist", np.zeros((4, 2))), dtype=np.float32).reshape(-1, 2)
    hist_od = ego_to_od(hist[-4:] if len(hist) else np.zeros((4, 2)))
    while hist_od.shape[0] < 4:
        hist_od = np.vstack([hist_od[:1], hist_od]) if len(hist_od) else np.zeros((4, 2), np.float32)
    hist_od = hist_od[-4:]
    lcf = np.asarray(sample.get("lcf", np.zeros(9)), dtype=np.float32).reshape(-1)

    def g(i, default=0.0):
        return float(lcf[i]) if lcf.size > i else default

    vx, vy = g(0) * 0.5, g(1) * 0.5
    v_yaw, ax, ay = g(4), g(2), g(3)
    cx, cy = g(2), g(3)
    vhead, steer = g(7) * 0.5, g(8)
    if hist_od.shape[0] >= 2:
        d1 = hist_od[-1] - hist_od[-2]
        d0 = hist_od[-2] - hist_od[-3] if hist_od.shape[0] >= 3 else d1
        ax, ay = float(d1[0] - d0[0]), float(d1[1] - d0[1])
    pts = ", ".join(f"({x:.2f},{y:.2f})" for x, y in hist_od)
    return (
        "*****Ego States:*****\n"
        "Current State:\n"
        f" - Velocity (vx,vy): ({vx:.2f},{vy:.2f})\n"
        f" - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})\n"
        f" - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})\n"
        f" - Can Bus: ({cx:.2f},{cy:.2f})\n"
        f" - Heading Speed: ({vhead:.2f})\n"
        f" - Steering: ({steer:.2f})\n"
        f"Historical Trajectory (last 2 seconds): [{pts}]\n"
    )


class RnbWaypointsRuntime:
    horizon = 6

    def __init__(self, ckpt: str = CKPT_DEFAULT, device: str = "cuda"):
        self.ckpt = ckpt
        self.device = device
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None:
            if hasattr(ip, "max_pixels"):
                ip.max_pixels = 50176
            if hasattr(ip, "min_pixels"):
                ip.min_pixels = 784
        self.model = AutoModelForImageTextToText.from_pretrained(
            ckpt,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        cfg = self.model.config
        text_cfg = getattr(cfg, "text_config", None) or cfg
        self.hidden_size = int(getattr(text_cfg, "hidden_size", 2560))
        self._n_logged = 0
        self._n_spec_logged = 0
        self._last_layer = None

    def _image_path(self, sample: dict) -> str:
        imgs = sample.get("images") or {}
        front = imgs.get("front_camera") or []
        if front:
            return str(front[-1])
        raise FileNotFoundError(f"no front camera for {sample.get('token')}")

    def _user_text(self, sample: dict) -> str:
        return SYSTEM + TRAFFIC + "\n" + ego_prompt(sample)

    def _messages(self, sample: dict):
        path = self._image_path(sample)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": path},
                    {"type": "text", "text": self._user_text(sample)},
                ],
            }
        ]

    def _inputs(self, sample: dict):
        from qwen_vl_utils import process_vision_info

        messages = self._messages(sample)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + "Planned Trajectory:\n"
        image_inputs, _video = process_vision_info(messages)
        inp = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inp.items()}

    def _lm_last_layer(self):
        if self._last_layer is not None:
            return self._last_layer
        candidates = []
        for name, mod in self.model.named_modules():
            if not name.endswith("layers"):
                continue
            try:
                n = len(mod)
            except TypeError:
                continue
            if n >= 8:
                candidates.append((name, n, mod[-1]))
        if not candidates:
            raise AttributeError("cannot find last LM layer")
        chosen = None
        for name, n, last in candidates:
            if "language" in name.lower() or "text_model" in name.lower():
                chosen = (name, n, last)
                break
        if chosen is None:
            chosen = candidates[-1]
        name, n, last = chosen
        print(f"[RnB] hook layer={name} n={n}", flush=True)
        self._last_layer = last
        return last

    @torch.inference_mode()
    def predict_and_hidden(self, sample: dict) -> tuple[np.ndarray, torch.Tensor]:
        inp = self._inputs(sample)
        prompt_len = int(inp["input_ids"].shape[1])
        captured: dict[str, torch.Tensor] = {}

        def hook(_m, _inp, out):
            if "h" in captured:
                return
            hs = out[0] if isinstance(out, tuple) else out
            captured["h"] = hs[:, -1].detach().float().cpu()[0]

        handle = self._lm_last_layer().register_forward_hook(hook)
        try:
            out = self.model.generate(
                **inp,
                max_new_tokens=96,
                do_sample=False,
                use_cache=True,
            )
        finally:
            handle.remove()
        text = self.processor.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        if self._n_logged < 3:
            print(f"[RnB] gen={text[:240]!r}", flush=True)
            self._n_logged += 1
        od = parse_planned_traj("Planned Trajectory:\n" + text, horizon=self.horizon)
        hidden = captured.get("h")
        if hidden is None:
            hidden = self.prefill_hidden(sample)
        return od_to_ego(od), hidden

    @torch.inference_mode()
    def prefill_hidden(self, sample: dict) -> torch.Tensor:
        inp = self._inputs(sample)
        out = self.model(**inp, output_hidden_states=True, use_cache=False)
        return out.hidden_states[-1][:, -1].float().cpu()[0]

    @torch.inference_mode()
    def predict(self, sample: dict) -> np.ndarray:
        xy, _ = self.predict_and_hidden(sample)
        return xy

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
            keys = getattr(layer, "keys", None) or getattr(layer, "key_cache", None)
            if keys is not None:
                return int(keys.shape[-2])
        return int(past[0][0].shape[-2])

    def _clone_past(self, past):
        if past is None:
            return None
        if hasattr(past, "copy") and callable(getattr(past, "copy")):
            try:
                return past.copy()
            except Exception:
                pass
        import copy

        return copy.deepcopy(past)

    def _forward_tokens(self, token_ids: torch.Tensor, past):
        """Continue from prefill KV with Qwen3-VL mRoPE cache_position.

        Without cache_position the inner model sets rope delta=0 and positions
        restart at 0, so teacher-force / tail emit garbage coordinates.
        """
        past_len = self._cache_seq_len(past)
        n = int(token_ids.shape[1])
        cache_position = torch.arange(
            past_len, past_len + n, device=token_ids.device, dtype=torch.long
        )
        return self.model(
            input_ids=token_ids,
            past_key_values=past,
            cache_position=cache_position,
            use_cache=True,
            pixel_values=None,
            pixel_values_videos=None,
            logits_to_keep=1 if n == 1 else 0,
        )

    @torch.inference_mode()
    def spec_generate(
        self,
        sample: dict,
        draft_head,
        extra: torch.Tensor,
        speed: torch.Tensor,
        accept_cfg,
        horizon: int = 6,
        max_new: int = 96,
    ):
        """Prefill → draft MLP → one teacher-force of draft [(x,y),...] → serial tail."""
        import time

        from trajspec.accept import longest_prefix_accept

        def _ms(fn):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0, out

        inp = self._inputs(sample)
        prompt_len = int(inp["input_ids"].shape[1])
        tokzr = self.processor.tokenizer

        def _prefill():
            return self.model(**inp, use_cache=True, output_hidden_states=True)

        t_pre, pre = _ms(_prefill)
        hidden = pre.hidden_states[-1][:, -1].float()
        prefill_logits = pre.logits[0, -1].float()
        prefill_past = pre.past_key_values
        extra_b = extra.to(device=self.device, dtype=torch.float32).view(1, -1)
        speed_b = speed.to(device=self.device, dtype=torch.float32).view(-1)

        def _draft():
            return draft_head(hidden, extra=extra_b, speed=speed_b)[0, :horizon]

        t_draft, draft_ego = _ms(_draft)
        draft_ego = draft_ego.float().cpu()
        draft_od = ego_to_od(draft_ego.numpy())
        inner = ", ".join(f"({x:.2f},{y:.2f})" for x, y in draft_od)
        draft_text = f"[{inner}]"
        draft_ids = tokzr(draft_text, add_special_tokens=False)["input_ids"]
        if draft_ids and isinstance(draft_ids[0], list):
            draft_ids = draft_ids[0]
        tok = torch.tensor([draft_ids], device=self.device, dtype=torch.long)

        def _verify():
            return self._forward_tokens(tok, self._clone_past(prefill_past))

        t_ver, ver = _ms(_verify)
        force_logits = ver.logits[0].float()
        teacher_tok = [int(prefill_logits.view(-1).argmax().item())]
        for i in range(min(len(draft_ids), force_logits.shape[0]) - 1):
            teacher_tok.append(int(force_logits[i].view(-1).argmax().item()))
        teacher_text = tokzr.decode(teacher_tok, skip_special_tokens=True)
        teacher_od = parse_planned_traj(teacher_text, horizon=horizon)
        teacher_ego = torch.as_tensor(od_to_ego(teacher_od), dtype=torch.float32)
        n_acc = longest_prefix_accept(draft_ego[:horizon], teacher_ego[:horizon], accept_cfg)

        committed_ids = list(draft_ids)
        n_serial = 0
        t_tail = 0.0
        final_text = draft_text
        if n_acc < horizon:
            if n_acc <= 0:
                prefix_text = ""
            else:
                keep_inner = ", ".join(f"({x:.2f},{y:.2f})" for x, y in draft_od[:n_acc])
                prefix_text = f"[{keep_inner}"
            prefix = tokzr(prefix_text, add_special_tokens=False)["input_ids"] if prefix_text else []
            if prefix and isinstance(prefix[0], list):
                prefix = prefix[0]

            def _tail():
                ids = list(prefix)
                if ids:
                    cur = torch.tensor([ids], device=self.device, dtype=torch.long)
                    out = self._forward_tokens(cur, self._clone_past(prefill_past))
                    logits = out.logits[0, -1].float()
                    past = out.past_key_values
                else:
                    logits = prefill_logits
                    past = self._clone_past(prefill_past)
                eos = getattr(tokzr, "eos_token_id", None)
                for _ in range(max_new):
                    tid = int(logits.view(-1).argmax().item())
                    if eos is not None and tid == eos:
                        break
                    ids.append(tid)
                    step = torch.tensor([[tid]], device=self.device, dtype=torch.long)
                    out = self._forward_tokens(step, past)
                    logits = out.logits[0, -1].float()
                    past = out.past_key_values
                    text = tokzr.decode(ids, skip_special_tokens=True)
                    n_pts = len(parse_planned_traj(text, horizon=horizon * 2))
                    if "]" in text and n_pts >= horizon:
                        break
                return ids

            t_tail, committed_ids = _ms(_tail)
            n_serial = max(len(committed_ids) - len(prefix), 0)
            final_text = tokzr.decode(committed_ids, skip_special_tokens=True)

        if self._n_spec_logged < 3:
            print(
                f"[RnB-spec] acc={int(n_acc)} draft={draft_text[:120]!r} "
                f"teacher={teacher_text[:120]!r} final={final_text[:120]!r}",
                flush=True,
            )
            self._n_spec_logged += 1
        spec_od = parse_planned_traj(final_text, horizon=horizon)
        spec_ego = od_to_ego(spec_od)
        info = {
            "n_accept": int(n_acc),
            "draft_text": draft_text,
            "teacher_verify_text": teacher_text,
            "final_text": final_text,
            "t_prefill_ms": round(t_pre, 2),
            "t_draft_ms": round(t_draft, 2),
            "t_verify_ms": round(t_ver, 2),
            "t_tail_ms": round(t_tail, 2),
            "t_spec_e2e_ms": round(t_pre + t_draft + t_ver + t_tail, 2),
            "n_draft_tokens": len(draft_ids),
            "n_serial_tail": int(n_serial),
            "prompt_len": prompt_len,
        }
        return spec_ego.astype("float32"), info
