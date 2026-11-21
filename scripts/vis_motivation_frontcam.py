#!/usr/bin/env python3
"""Overlay VLA / ego-kinematics / ordinary MLP trajectories on CAM_FRONT.

Uses the same 1000-frame protocol as the motivation L2 plots:
  OpenDrive = rows [1500, 1500+N) of cache_opendrive_nusc_val.pt
  R&B       = first N rows of the RnB val cache
Ordinary MLP is a 2-layer head on prefill hidden + 16-d extras, trained on a
disjoint split vs GT (not the TrajSpecDec residual draft).
"""
from __future__ import annotations

import argparse
import json
import os
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.draft_head import bicycle_waypoints  # noqa: E402
from trajspec.nusc_data import (  # noqa: E402
    VAL_PKL,
    _cam_abs,
    extras_for_tokens,
    load_infos,
    token_index,
)

HORIZON = 6
DT = 0.5
CACHE = ROOT / "results"
# The RnB teacher cache lives in TrajSpecDec; override with TRAJSPEC_RNB_CACHE.
RNB_CACHE = Path(
    os.environ.get("TRAJSPEC_RNB_CACHE", "/home/xinhchen2/hhb/TrajSpecDec/results/cache_rnb_nusc_val.pt")
)
OUT_ROOT = CACHE / "motivation_frontcam"
FONT_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_LEGEND = 40
FONT_CAPTION = 32
FONT_BEV = 24

COLORS = {
    "gt": (255, 255, 255),
    "vla": (49, 130, 206),
    "kin": (237, 137, 54),
    "mlp": (56, 178, 107),
}
LABELS = {
    "gt": "GT",
    "vla": "VLA",
    "kin": "仅自车运动学",
    "mlp": "普通 MLP",
}


class PrefillKineMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        d = hidden_size + 16
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Linear(128, HORIZON * 2),
        )

    def forward(self, hidden: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden.float(), extra.float()], dim=-1)
        return self.net(x).view(-1, HORIZON, 2)


def kinematics_only(extra: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
    v = speed.float()
    v_fwd = extra[:, 4] * 15.0
    use = torch.where(v.abs() > 1e-3, v, v_fwd)
    accel = extra[:, 6] * 3.0
    omega = extra[:, 8]
    return bicycle_waypoints(use, accel, omega, HORIZON, dt=DT)


def train_simple(hs, extra, y, device, steps=400, bs=64, seed=17):
    torch.manual_seed(seed)
    model = PrefillKineMLP(hs.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    n = hs.shape[0]
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),))
        pred = model(hs[idx].to(device), extra[idx].to(device))
        loss = (pred - y[idx].to(device)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_mlp(model, hs, extra, device):
    out = []
    for i in range(0, hs.shape[0], 64):
        out.append(model(hs[i : i + 64].to(device), extra[i : i + 64].to(device)).cpu())
    return torch.cat(out, dim=0)


def rebuild_extras(tokens: list[str]):
    found = extras_for_tokens(tokens)
    extra, speed, gt = [], [], []
    hit = 0
    for t in tokens:
        if t in found:
            e, s, g = found[t]
            extra.append(torch.nn.functional.pad(e.float().view(-1), (0, 16 - e.numel()))[:16])
            speed.append(s.view(-1)[:1].float())
            gt.append(g[:6].float())
            hit += 1
        else:
            extra.append(torch.zeros(16))
            speed.append(torch.zeros(1))
            gt.append(torch.zeros(6, 2))
    print(f"[extra] rebuilt {hit}/{len(tokens)}", flush=True)
    return torch.stack(extra).float(), torch.stack(speed).float().view(-1), torch.stack(gt).float()


def load_rows(path: Path, indices: list[int]):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    rows = blob["rows"]
    tokens = [rows[i]["token"] for i in indices]
    hs = torch.stack([rows[i]["hidden"] for i in indices]).float()
    vla = torch.stack([rows[i]["teacher"][:6] for i in indices]).float()
    extra, speed, gt = rebuild_extras(tokens)
    cmds = [str(rows[i].get("cmd", "")) for i in indices]
    for i, row_i in enumerate(indices):
        if gt[i].abs().sum() < 0.05 and "gt" in rows[row_i]:
            g = torch.as_tensor(rows[row_i]["gt"])[:6].float()
            if g.abs().sum() > 0.05:
                gt[i] = g
    return tokens, hs, extra, speed, vla, gt, cmds


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def densify_xy(xy: np.ndarray, n_per: int = 12) -> np.ndarray:
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    out = [pts[0]]
    prev = pts[0]
    for p in pts[1:]:
        for t in np.linspace(0.0, 1.0, n_per, endpoint=False)[1:]:
            out.append((1.0 - t) * prev + t * p)
        out.append(p)
        prev = p
    return np.stack(out, axis=0)


def project_ego_xy(xy: np.ndarray, cam: dict, z_ego: float) -> tuple[np.ndarray, np.ndarray]:
    R = quat_wxyz_to_R(cam["sensor2ego_rotation"])
    t = np.asarray(cam["sensor2ego_translation"], dtype=np.float64)
    K = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    pts = np.stack([xy[:, 0], xy[:, 1], np.full(len(xy), z_ego)], axis=1)
    pc = (pts - t) @ R
    z = pc[:, 2]
    u = K[0, 0] * pc[:, 0] / np.clip(z, 1e-6, None) + K[0, 2]
    v = K[1, 1] * pc[:, 1] / np.clip(z, 1e-6, None) + K[1, 2]
    return np.stack([u, v], axis=1), z > 0.15


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_TTC, size=size, index=2)
    except OSError:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_polyline(draw: ImageDraw.ImageDraw, uv, valid, wh, color, width, dashed=False):
    w, h = wh
    pts = []

    def flush():
        nonlocal pts
        if len(pts) >= 2:
            if dashed:
                _stroke(draw, pts, (0, 0, 0), width + 3, True)
                _stroke(draw, pts, color, width, True)
            else:
                draw.line(pts, fill=(0, 0, 0), width=width + 4, joint="curve")
                draw.line(pts, fill=(255, 255, 255), width=width + 2, joint="curve")
                draw.line(pts, fill=color, width=width, joint="curve")
        pts = []

    for (u, v), ok in zip(uv, valid):
        if (not ok) or abs(u) > 8000 or abs(v) > 8000:
            flush()
            continue
        # keep points a full frame below the bumper so the polyline enters from the bottom
        if -w <= u < 2 * w and -h <= v < 2 * h:
            pts.append((float(u), float(v)))
        else:
            flush()
    flush()


def _stroke(draw, pts, color, width, dashed):
    if not dashed:
        draw.line(pts, fill=color, width=width, joint="curve")
        return
    # approximate dashes along the polyline
    acc = 0.0
    on = True
    dash, gap = 14.0, 10.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        if seg < 1e-3:
            continue
        ux, uy = dx / seg, dy / seg
        left = seg
        cx, cy = x0, y0
        while left > 1e-3:
            span = dash if on else gap
            take = min(left, span - acc)
            nx, ny = cx + ux * take, cy + uy * take
            if on:
                draw.line([(cx, cy), (nx, ny)], fill=color, width=width)
            acc += take
            left -= take
            cx, cy = nx, ny
            if acc >= span - 1e-6:
                acc = 0.0
                on = not on


def draw_waypoints(draw, uv, valid, wh, color, r=6):
    w, h = wh
    for (u, v), ok in zip(uv, valid):
        if not ok:
            continue
        if 0 <= u < w and 0 <= v < h:
            draw.ellipse((u - r, v - r, u + r, v + r), fill=color, outline=(0, 0, 0))


def draw_bev(xy_dict: dict[str, np.ndarray], size: int = 400) -> Image.Image:
    extent = 8.0
    for xy in xy_dict.values():
        if xy is None or len(xy) == 0:
            continue
        extent = max(extent, float(np.linalg.norm(xy, axis=-1).max()) * 1.15)
    extent = min(max(extent, 12.0), 45.0)
    img = Image.new("RGBA", (size, size), (8, 10, 14, 210))
    d = ImageDraw.Draw(img)
    pad = 28
    cx, cy = size // 2, size - pad

    def to_px(xy):
        x, y = float(xy[0]), float(xy[1])
        u = cx - y / extent * (size - 2 * pad)
        v = cy - x / extent * (size - 2 * pad)
        return u, v

    # rings at 10/20/30 m
    font = load_font(FONT_BEV)
    for m in (10, 20, 30):
        if m > extent:
            continue
        r = m / extent * (size - 2 * pad)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 255, 255, 50), width=1)
        d.text((cx + 6, cy - r - 22), f"{m}m", fill=(230, 230, 230, 220), font=font)
    d.line([(pad, cy), (size - pad, cy)], fill=(255, 255, 255, 40), width=1)
    d.polygon([(cx, cy - 10), (cx - 6, cy + 4), (cx + 6, cy + 4)], fill=(230, 230, 230, 220))
    order = ("gt", "mlp", "kin", "vla")
    widths = {"gt": 2, "mlp": 4, "kin": 4, "vla": 4}
    for key in order:
        xy = xy_dict.get(key)
        if xy is None:
            continue
        pts = [(cx, cy)] + [to_px(p) for p in xy]
        d.line(pts, fill=COLORS[key] + ((255,) if key != "gt" else (200,)), width=widths[key])
        u, v = to_px(xy[-1])
        d.ellipse((u - 3, v - 3, u + 3, v + 3), fill=COLORS[key] + (255,))
    d.rectangle((0, 0, size - 1, size - 1), outline=(255, 255, 255, 70), width=1)
    d.text((10, 8), "BEV  +X↑ +Y←", fill=(240, 240, 240, 230), font=font)
    return img


def draw_legend(draw, ade: dict, font, font_s, origin=(18, 18)):
    x, y = origin
    rows = [("vla", ade.get("vla")), ("kin", ade.get("kin")), ("mlp", ade.get("mlp")), ("gt", None)]
    labels = []
    max_w = 0
    for key, val in rows:
        label = LABELS[key]
        if val is not None:
            label = f"{label}   ADE {val:.2f} m"
        labels.append((key, label))
        max_w = max(max_w, _text_size(draw, label, font)[0])
    row_h = max(48, _text_size(draw, "Ag", font)[1] + 18)
    swatch_w, swatch_h = 44, 22
    pad_x, pad_y = 18, 14
    box_w = pad_x + swatch_w + 16 + max_w + 22
    box_h = pad_y + row_h * len(rows) + 10
    draw.rectangle((x, y, x + box_w, y + box_h), fill=(12, 14, 18), outline=(80, 80, 80), width=2)
    yy = y + pad_y
    for key, label in labels:
        cy = yy + row_h // 2
        draw.rectangle((x + pad_x, cy - swatch_h // 2, x + pad_x + swatch_w, cy + swatch_h // 2), fill=COLORS[key])
        draw.text((x + pad_x + swatch_w + 14, yy + 4), label, fill=(245, 245, 245), font=font, stroke_width=1, stroke_fill=(12, 14, 18))
        yy += row_h


def render_one(
    img_path: str,
    cam: dict,
    trajs: dict[str, np.ndarray],
    title: str,
    ade: dict,
    z_ego: float,
) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    composed = img.convert("RGB")
    d = ImageDraw.Draw(composed)
    font = load_font(FONT_LEGEND)
    font_s = load_font(FONT_CAPTION)
    wh = img.size
    order = (("gt", 4, True), ("mlp", 6, False), ("kin", 6, False), ("vla", 7, False))
    for key, width, dashed in order:
        xy = trajs[key]
        dense = densify_xy(xy, n_per=14)
        uv, valid = project_ego_xy(dense, cam, z_ego)
        draw_polyline(d, uv, valid, wh, COLORS[key], width, dashed=dashed)
        uv_w, valid_w = project_ego_xy(xy, cam, z_ego)
        draw_waypoints(d, uv_w, valid_w, wh, COLORS[key], r=6 if key != "gt" else 5)
    d2 = d
    # legend on opaque bar so Chinese glyphs stay readable
    draw_legend(d2, ade, font, font_s)
    cap_h = _text_size(d2, title, font_s)[1]
    d2.text(
        (18, composed.height - cap_h - 18),
        title,
        fill=(255, 255, 255),
        font=font_s,
        stroke_width=3,
        stroke_fill=(0, 0, 0),
    )
    bev = draw_bev(trajs, size=400)
    composed.paste(bev, (composed.width - bev.width - 12, 12), bev.convert("RGBA"))
    return composed


def l2_ade(pred: np.ndarray, gt: np.ndarray) -> float:
    d = np.linalg.norm(pred[:HORIZON] - gt[:HORIZON], axis=-1)
    return float(d.mean())


def write_index(out_dir: Path, items: list[dict], title: str):
    cards = []
    for it in items:
        rel = it["file"]
        cap = f'{it["i"]:04d}  {it["token"][:8]}  {it["cmd"]}'
        cards.append(
            f'<a class="c" href="{rel}"><img loading="lazy" src="{rel}" alt="">'
            f"<div>{cap}<br>VLA {it['ade_vla']:.2f} · kin {it['ade_kin']:.2f} · MLP {it['ade_mlp']:.2f}</div></a>"
        )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
body{{margin:0;background:#111;color:#eee;font:14px/1.4 sans-serif}}
h1{{margin:16px}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:10px;padding:12px}}
.c{{color:#ddd;text-decoration:none}}
.c img{{width:100%;height:auto;background:#000}}
.c div{{padding:6px 2px}}
</style>
<h1>{title} · {len(items)} frames</h1>
<div class="g">
{"".join(cards)}
</div>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def run_model(name: str, n: int, z_ego: float, device: torch.device, out_dir: Path, start: int = 0):
    if name == "opendrive":
        cache = CACHE / "cache_opendrive_nusc_val.pt"
        n_all = len(torch.load(cache, map_location="cpu", weights_only=False)["rows"])
        ev_idx = list(range(1500 + start, min(1500 + start + n, n_all)))
        tr_idx = list(range(2500, min(5500, n_all)))
        print(f"[OpenDrive] load eval {len(ev_idx)} + train {len(tr_idx)}", flush=True)
        tokens, hs, extra, speed, vla, gt, cmds = load_rows(cache, ev_idx)
        _, hs_tr, extra_tr, _, _, gt_tr, _ = load_rows(cache, tr_idx)
        title_prefix = "OpenDriveVLA-0.5B"
        steps = 800
        seed = 0
    elif name == "rnb":
        cache = RNB_CACHE
        if not cache.exists():
            raise SystemExit(f"RnB cache missing: {cache}")
        n_all = len(torch.load(cache, map_location="cpu", weights_only=False)["rows"])
        ev_idx = list(range(start, min(start + n, n_all)))
        tr_idx = list(range(3000, min(5000, n_all)))
        print(f"[R&B] load eval {len(ev_idx)} + train {len(tr_idx)}", flush=True)
        tokens, hs, extra, speed, vla, gt, cmds = load_rows(cache, ev_idx)
        _, hs_tr, extra_tr, _, _, gt_tr, _ = load_rows(cache, tr_idx)
        title_prefix = "R&B-EnCoRe-4B"
        steps = 800
        seed = 0
    else:
        raise ValueError(name)

    print(f"[{name}] train ordinary MLP on {hs_tr.shape[0]} frames, steps={steps}", flush=True)
    mlp = train_simple(hs_tr, extra_tr, gt_tr, device, steps=steps, seed=seed)
    kin = kinematics_only(extra, speed).cpu()
    simple = predict_mlp(mlp, hs, extra, device)
    vla_np = vla.numpy()
    kin_np = kin.numpy()
    mlp_np = simple.numpy()
    gt_np = gt.numpy()

    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    skipped = 0
    for i, tok in enumerate(tokens):
        if tok not in tok2i:
            skipped += 1
            continue
        info = infos[tok2i[tok]]
        cam = (info.get("cams") or {}).get("CAM_FRONT") or {}
        img_path = _cam_abs(cam.get("data_path", ""))
        if not Path(img_path).exists() or "cam_intrinsic" not in cam:
            skipped += 1
            continue
        trajs = {
            "gt": gt_np[i],
            "vla": vla_np[i],
            "kin": kin_np[i],
            "mlp": mlp_np[i],
        }
        ade = {
            "vla": l2_ade(vla_np[i], gt_np[i]),
            "kin": l2_ade(kin_np[i], gt_np[i]),
            "mlp": l2_ade(mlp_np[i], gt_np[i]),
        }
        cmd = cmds[i] or str(info.get("gt_ego_fut_cmd", ""))
        title = f"{title_prefix}  {tok[:12]}  {cmd}"
        vis = render_one(img_path, cam, trajs, title, ade, z_ego)
        fname = f"{i:04d}_{tok[:8]}.jpg"
        vis.save(out_dir / fname, quality=88, optimize=True)
        items.append(
            {
                "i": i,
                "token": tok,
                "file": fname,
                "cmd": cmd,
                "ade_vla": ade["vla"],
                "ade_kin": ade["kin"],
                "ade_mlp": ade["mlp"],
                "spread": float(
                    np.linalg.norm(vla_np[i][-1] - kin_np[i][-1])
                    + np.linalg.norm(vla_np[i][-1] - mlp_np[i][-1])
                ),
            }
        )
        if (i + 1) % 50 == 0 or i + 1 == len(tokens):
            print(f"[{name}] wrote {len(items)}/{len(tokens)} (skip {skipped})", flush=True)

    torch.save(
        {
            "tokens": tokens,
            "vla": vla,
            "kin": kin,
            "mlp": simple,
            "gt": gt,
        },
        out_dir / "trajs.pt",
    )
    write_index(out_dir, items, title_prefix)
    (out_dir / "manifest.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    hi = out_dir / "highlights"
    if hi.exists():
        shutil.rmtree(hi)
    hi.mkdir()
    top = sorted(items, key=lambda x: -x["spread"])[:24]
    for rank, it in enumerate(top):
        src = out_dir / it["file"]
        if src.exists():
            shutil.copy2(src, hi / f"{rank:02d}_{it['file']}")
    write_index(hi, [{**it, "file": f"{rank:02d}_{it['file']}"} for rank, it in enumerate(top)], f"{title_prefix} highlights")
    print(f"[{name}] done {len(items)} images -> {out_dir}", flush=True)
    return len(items)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--start", type=int, default=0, help="start index inside the eval slice")
    p.add_argument("--model", choices=["opendrive", "rnb", "both"], default="both")
    p.add_argument("--z", type=float, default=0.25, help="ego-frame height (m) for ground waypoints")
    p.add_argument("--out", default=str(OUT_ROOT))
    args = p.parse_args()
    device = torch.device("cpu")
    out_root = Path(args.out)
    models = ["opendrive", "rnb"] if args.model == "both" else [args.model]
    for name in models:
        run_model(name, args.n, args.z, device, out_root / name, start=args.start)


if __name__ == "__main__":
    main()
