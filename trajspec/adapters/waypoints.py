"""Adapter for stanfordasl/nuscenes-waypoints-model (Qwen3-VL-4B, trajectory-only).

Chosen as the third teacher because it matches OpenDriveVLA on the properties
that make speculative decoding pay off:
  * short horizon       6 waypoints / 3 s
  * textual output      "Planned Trajectory:\\n[(x,y), ...]"  (token-by-token)
  * trajectory-only     no chain-of-thought to sit through

Prompt and output format are taken verbatim from the RnB-EnCoRe data generator
(`Agent-Driver/agentdriver/execution/gen_drive_vla_data.py`) so the model sees
the distribution it was fine-tuned on.

Output format: "Planned Trajectory:\\n[(x1,y1), (x2,y2), ... , (x6,y6)]"
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = os.environ.get(
    "TRAJSPEC_RNB_CKPT", "/data2/hf_overlap_models/nuscenes-waypoints-model"
)

SYSTEM_MESSAGE = """**Autonomous Driving Planner**
Role: You're an autonomous vehicle's brain. Given an image of the current scenario, plan a 3-second safe trajectory to avoid obstacles.

Context:
- Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0). Units: meters.
- Goal: Plan a 3-second route using 6 waypoints (0.5s intervals).

Task:
- Based on inputs, plan a safe, feasible 3-second trajectory of 6 waypoints.

Output:
Planned Trajectory:\\n[(x1,y1), (x2,y2), ... , (x6,y6)]
"""


def ego_state_text(lcf, hist) -> str:
    """Ego block, transcribed from Agent-Driver's
    `agentdriver/functional_tools/ego_state.py::extract_ego_inputs`.

    Field mapping over the VAD lcf vector (which mirrors Agent-Driver's
    `ego_states`):
      vx = lcf[0]*0.5, vy = lcf[1]*0.5      (velocity, halved like upstream)
      v_yaw = lcf[4]
      ax/ay = diff of the last two *velocity* samples (ego_hist_traj_diff),
              not differences of positions
      can bus = lcf[2], lcf[3]
      heading speed = lcf[7]*0.5, steering = lcf[8]
    Header string is "*****Ego States:*****" (space, no hyphen) exactly as
    upstream emits it.
    """
    arr = None if lcf is None else np.asarray(lcf, dtype=np.float64).reshape(-1)

    def g(i: int, d: float = 0.0) -> float:
        return float(arr[i]) if arr is not None and arr.size > i else d

    vx = g(0) * 0.5
    vy = g(1) * 0.5
    v_yaw = g(4)
    cx = g(2)
    cy = g(3)
    vhead = g(7) * 0.5
    steer = g(8)

    # Acceleration from velocity differences, mirroring ego_hist_traj_diff.
    ax = ay = 0.0
    if hist is not None and len(hist) >= 3:
        h = np.asarray(hist, dtype=np.float64)
        d = np.diff(h[:, :2], axis=0)
        if len(d) >= 2:
            ax = float(d[-1, 0] - d[-2, 0])
            ay = float(d[-1, 1] - d[-2, 1])

    if hist is not None and len(hist) >= 4:
        pts = np.asarray(hist, dtype=np.float64)[:4]
    elif hist is not None and len(hist):
        pts = np.asarray(hist, dtype=np.float64)
        pts = np.vstack([pts[0]] * (4 - len(pts)) + [pts])[-4:]
    else:
        pts = np.zeros((4, 2))

    his = ", ".join(f"({float(p[0]):.2f},{float(p[1]):.2f})" for p in pts)
    return (
        "\n*****Ego States:*****\n"
        f"Current State:\n"
        f" - Velocity (vx,vy): ({vx:.2f},{vy:.2f})\n"
        f" - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})\n"
        f" - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})\n"
        f" - Can Bus: ({cx:.2f},{cy:.2f})\n"
        f" - Heading Speed: ({vhead:.2f})\n"
        f" - Steering: ({steer:.2f})\n"
        f"Historical Trajectory (last 2 seconds): [{his}]"
    )


def mission_goal(command: str) -> str:
    c = str(command).lower()
    if "left" in c:
        return "LEFT"
    if "right" in c:
        return "RIGHT"
    return "FORWARD"


TRAJ_RE = re.compile(r"\[(.*?)\]", re.S)


def parse_waypoints(text: str, horizon: int = 6) -> torch.Tensor:
    """Parse 'Planned Trajectory: [(x,y), ...]' into [K,2] ego-frame (forward, left).

    The prompt tells the model "X-axis is perpendicular, and Y-axis is parallel
    to the direction you're facing", i.e. its output is (lateral, longitudinal).
    The rest of this codebase uses (forward, left), so swap on parse.
    """
    m = TRAJ_RE.search(text)
    if m is None:
        return torch.zeros(horizon, 2)
    pairs = re.findall(r"\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)", m.group(1))
    # Model emits (lateral, longitudinal) with lateral pointing right; the rest
    # of this codebase uses (forward, left) — swap both axes.
    pts = [(float(y), -float(x)) for x, y in pairs]
    if not pts:
        return torch.zeros(horizon, 2)
    if len(pts) < horizon:
        pts = pts + [pts[-1]] * (horizon - len(pts))
    return torch.tensor(pts[:horizon], dtype=torch.float32)


class WaypointsRuntime:
    """Teacher runtime exposing prefill / generate / hidden states."""

    horizon = 6

    def __init__(self, model_dir: str = MODEL_DIR, device: str = "cuda"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        print(f"[Waypoints] load {model_dir}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        cfg = self.model.config
        self.hidden_size = int(getattr(cfg, "text_config", cfg).hidden_size)
        print(f"[Waypoints] ready hidden={self.hidden_size}", flush=True)

    def _messages(self, sample: dict) -> list[dict]:
        hist = sample.get("hist")
        img = sample["images"]["front_camera"][0]
        path = img if img.startswith("/") else str(Path(sample.get("sensor_data_path", "")) / img)
        user = (
            "<image>\n"
            + SYSTEM_MESSAGE
            + ego_state_text(sample.get("lcf"), hist)
        )
        return [
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{path}"},
                {"type": "text", "text": user},
            ]},
        ]

    def _inputs(self, sample: dict):
        from qwen_vl_utils import process_vision_info

        msgs = self._messages(sample)
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        return self.processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt")

    @torch.inference_mode()
    def prefill_hidden(self, sample: dict) -> torch.Tensor:
        inp = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in self._inputs(sample).items()}
        out = self.model(**inp, use_cache=False, output_hidden_states=True)
        return out.hidden_states[-1][:, -1].float().cpu()[0]

    @torch.inference_mode()
    def predict(self, sample: dict) -> np.ndarray:
        inp = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in self._inputs(sample).items()}
        out = self.model.generate(**inp, max_new_tokens=96, do_sample=False, use_cache=True)
        text = self.processor.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_waypoints(text, self.horizon).numpy()
