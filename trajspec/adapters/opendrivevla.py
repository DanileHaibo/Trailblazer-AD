"""Adapter around official OpenDriveVLA (Qwen2.5 text waypoint AR)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trajspec.serialize import parse_opendrive_traj


class OpenDriveVLATeacher:
    """Wraps OpenDriveVLA generate() + retrieve_traj.

    The official 0.5B checkpoint on HuggingFace is gated. When it is not
    present, `available()` is False and the toy OpenDrive-style teacher is used.
    """

    horizon = 6
    hidden_size = 896  # Qwen2.5-0.5B

    def __init__(
        self,
        repo_root: str | Path = os.environ.get("TRAJSPEC_OPENDRIVE_REPO", "/home/xinhchen2/hhb/OpenDriveVLA"),
        checkpoint: str | None = None,
        device: str = "cuda",
    ):
        self.repo_root = Path(repo_root)
        self.checkpoint = checkpoint
        self.device = device
        self.model = None
        self.tokenizer = None

    def available(self) -> bool:
        return bool(self.checkpoint) and Path(self.checkpoint).exists()

    def load(self) -> None:
        if not self.available():
            raise FileNotFoundError(
                "OpenDriveVLA checkpoint not found. Accept the license at "
                "https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B and pass --opendrive-ckpt"
            )
        import sys

        sys.path.insert(0, str(self.repo_root))
        from drivevla.inference_drivevla import load_pretrained_model  # type: ignore

        self.tokenizer, self.model, _image_processor, _ctx = load_pretrained_model(
            self.checkpoint
        )

    def prefill(self, scene: dict) -> tuple[torch.Tensor, Any]:
        if self.model is None:
            raise RuntimeError("call load() first")
        captured: dict[str, torch.Tensor] = {}

        def hook(_m, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["h"] = hs[:, -1].detach()

        layers = self._lm_layers()
        handle = layers[-1].register_forward_hook(hook)
        input_ids = scene["input_ids"].to(self.device)
        with torch.inference_mode():
            self.model(input_ids=input_ids, use_cache=True)
        handle.remove()
        return captured["h"][0].float().cpu(), {"input_ids": input_ids}

    def teacher_waypoints(self, scene: dict, kv: Any, hidden: torch.Tensor) -> torch.Tensor:
        from drivevla.utils.trajectory_utils import retrieve_traj  # type: ignore

        input_ids = scene["input_ids"].to(self.device)
        with torch.inference_mode():
            cont = self.model.generate(
                input_ids,
                uniad_data=scene.get("uniad_data"),
                uniad_pth=scene.get("uniad_pth"),
                qa_instance_ind=scene.get("qa_instance_ind"),
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
                num_beams=1,
            )
        text = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        coords = retrieve_traj(text)
        return torch.tensor(coords, dtype=torch.float32)

    def continue_decode(self, scene: dict, kv: Any, accepted: torch.Tensor, remaining: int) -> torch.Tensor:
        full = self.teacher_waypoints(scene, kv, torch.zeros(1))
        start = int(accepted.shape[0])
        return full[start : start + remaining]

    def _lm_layers(self):
        m = self.model
        for attr in ("model", "llm", "language_model"):
            if hasattr(m, attr):
                m = getattr(m, attr)
        if hasattr(m, "layers"):
            return m.layers
        raise AttributeError("cannot find transformer layers on OpenDriveVLA")


class OpenDriveLMRuntime:
    """OpenDriveVLA-0.5B language planner without UniAD vision.

    Official inference injects UniAD scene/track/map tokens. Bench2Drive does
    not provide those features, so we load only the Qwen2 LM weights and run
    AR decode in the official [(x,y),...] format from ego/history/command.
    """

    horizon = 6

    def __init__(
        self,
        ckpt: str = os.environ.get("TRAJSPEC_OPENDRIVE_CKPT", "/data2/hf_overlap_models/OpenDriveVLA-0.5B"),
        device: str = "cuda",
    ):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

        print("[OpenDriveVLA] load LM from", ckpt, flush=True)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        cfg = Qwen2Config(
            vocab_size=151682,
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            max_position_embeddings=32768,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
            tie_word_embeddings=True,
            hidden_act="silu",
            bos_token_id=151643,
            eos_token_id=151645,
        )
        self.model = Qwen2ForCausalLM(cfg)
        raw = load_file(str(Path(ckpt) / "model.safetensors"))
        keep = {}
        for k, v in raw.items():
            if k.startswith("model.layers.") or k in (
                "model.embed_tokens.weight",
                "model.norm.weight",
                "lm_head.weight",
            ):
                keep[k] = v
        missing, unexpected = self.model.load_state_dict(keep, strict=False)
        print(
            f"[OpenDriveVLA] LM loaded missing={len(missing)} unused={len(unexpected)}",
            flush=True,
        )
        self.model.to(device=device, dtype=torch.float16).eval()
        self.hidden_size = 896
        self._n_logged = 0

    def _prompt(self, s: dict) -> str:
        lcf = s.get("lcf")
        if lcf is not None:
            lcf = np.asarray(lcf, dtype=np.float32).reshape(-1)
            vx, vy = float(lcf[0]) * 0.5, float(lcf[1]) * 0.5
            v_yaw = float(lcf[4]) if lcf.size > 4 else 0.0
            ax = ay = 0.0
            hist = s.get("hist")
            if hist is not None and len(hist) >= 2:
                d = np.diff(np.vstack([hist[0], hist]), axis=0)
                if len(d) >= 2:
                    ax, ay = float(d[-1, 0] - d[-2, 0]), float(d[-1, 1] - d[-2, 1])
            cx, cy = float(lcf[2]), float(lcf[3]) if lcf.size > 3 else (0.0, 0.0)
            vhead = float(lcf[7]) * 0.5 if lcf.size > 7 else 0.0
            steering = float(lcf[8]) if lcf.size > 8 else 0.0
            ego_message = (
                f"- Velocity (vx,vy): ({vx:.2f},{vy:.2f})"
                f" - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})"
                f" - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})"
                f" - Can Bus: ({cx:.2f},{cy:.2f})"
                f" - Heading Speed: ({vhead:.2f})"
                f" - Steering: ({steering:.2f})"
            )
        else:
            vx, vy = s["vehicle_velocity"]
            ax, ay = s.get("vehicle_acceleration", [0.0, 0.0])
            ego_message = (
                f"- Velocity (vx,vy): ({vx:.2f},{vy:.2f})"
                f" - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})"
            )
        hist = s.get("hist")
        if hist is None:
            his_msg = "[(0.00,0.00),(0.00,0.00),(0.00,0.00),(0.00,0.00)]"
        else:
            pts = [(float(p[0]), float(p[1])) for p in np.asarray(hist)[-4:]]
            while len(pts) < 4:
                pts = [(0.0, 0.0)] + pts
            his_msg = "[" + ",".join(f"({x:.2f},{y:.2f})" for x, y in pts) + "]"
        cmd = str(s.get("mission_goal") or s.get("driving_command") or "keep forward")
        if "left" in cmd.lower():
            mission = "turn left"
        elif "right" in cmd.lower():
            mission = "turn right"
        else:
            mission = "keep forward"
        return (
            f"Ego states: {ego_message}\n"
            f"Historical trajectory (last 2 seconds): {his_msg}\n"
            f"Mission goal: {mission}\n"
            "Planning trajectory: <trajectory>"
        )

    def _tokenize(self, s: dict):
        user = self._prompt(s)
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a planning-oriented autonomous driving model."},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            text = text + "<traj_start>"
        else:
            text = user + "<traj_start>"
        return self.tokenizer(text, return_tensors="pt")

    @torch.inference_mode()
    def prefill_hidden(self, s: dict) -> torch.Tensor:
        inp = {k: v.to(self.device) for k, v in self._tokenize(s).items()}
        out = self.model(**inp, output_hidden_states=True, use_cache=False)
        return out.hidden_states[-1][:, -1].float().cpu()[0]

    @torch.inference_mode()
    def predict(self, s: dict) -> np.ndarray:
        inp = {k: v.to(self.device) for k, v in self._tokenize(s).items()}
        out = self.model.generate(
            **inp,
            max_new_tokens=96,
            do_sample=False,
            use_cache=True,
        )
        text = self.tokenizer.decode(out[0, inp["input_ids"].shape[1] :], skip_special_tokens=False)
        if self._n_logged < 2:
            print(f"[OpenDriveVLA] gen={text[:300]!r}", flush=True)
            self._n_logged += 1
        return parse_opendrive_traj("<traj_start>" + text, horizon=6).numpy()

