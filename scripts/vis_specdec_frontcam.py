#!/usr/bin/env python3
"""Overlay VLA / draft / spec / GT on CAM_FRONT for the same 1000 frames.

Coincident waypoints (same timestep, nearby pixels) are merged into one marker
with split / pie colors so overlapping methods stay readable.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajspec.accept import AcceptConfig  # noqa: E402
from trajspec.draft_head import DraftHead  # noqa: E402
from trajspec.eval_loop import spec_from_draft  # noqa: E402
from trajspec.nusc_data import VAL_PKL, _cam_abs, extras_for_tokens, load_infos, token_index  # noqa: E402

HORIZON = 6
CACHE = ROOT / "results"
TOKEN_LIST = Path("/home/xinhchen2/hhb/TrajSpecDec/results/motivation_frontcam/frame_tokens.json")
OUT_ROOT = CACHE / "motivation_frontcam" / "specdec"
# TrajSpecDec still holds the RnB teacher cache used for these figures.
TSD = Path("/home/xinhchen2/hhb/TrajSpecDec/results")
MODELS = {
    "opendrive": {
        "cache": CACHE / "cache_opendrive_nusc_val.pt",
        "ckpt": CACHE / "specdec_opendrive_heldout.pt",
        "title": "OpenDriveVLA-0.5B",
        "tokens": "slice1500",
    },
    "rnb": {
        "cache": TSD / "cache_rnb_nusc_val.pt",
        "ckpt": ROOT / "weights" / "draft_waypoints_nusc.pt",
        "title": "R&B-EnCoRe-4B",
        "tokens": "list",
    },
}
FONT_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_LEGEND = 40
FONT_CAPTION = 32
FONT_BEV = 24

# Distinct, high-contrast on road / sky / shadow.
COLORS = {
    "vla": (51, 154, 240),      # blue    only VLA
    "draft": (230, 73, 73),     # red     only draft
    "spec": (34, 197, 94),      # green   our method
    "gt": (250, 250, 250),      # white   GT
}
LABELS = {
    "vla": "VLA",
    "draft": "Draft",
    "spec": "Ours",
    "gt": "GT",
}
METHOD_ORDER = ("gt", "vla", "draft", "spec")
POINT_R = 13
MERGE_PX = 18


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


def draw_polyline(draw, uv, valid, wh, color, width, dashed=False):
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
        if -w <= u < 2 * w and -h <= v < 2 * h:
            pts.append((float(u), float(v)))
        else:
            flush()
    flush()


def _stroke(draw, pts, color, width, dashed):
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


def draw_split_marker(draw, cx, cy, colors, r=POINT_R):
    """One waypoint. 1 color = disk; 2 = left/right; 3–4 = pie slices."""
    cx, cy, r = float(cx), float(cy), float(r)
    bbox = (cx - r, cy - r, cx + r, cy + r)
    draw.ellipse((cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2), fill=(0, 0, 0))
    n = len(colors)
    if n == 1:
        draw.ellipse(bbox, fill=colors[0])
    elif n == 2:
        # left | right, matching “一个点左右两个颜色”
        draw.pieslice(bbox, 90, 270, fill=colors[0])
        draw.pieslice(bbox, -90, 90, fill=colors[1])
    else:
        step = 360.0 / n
        start = 90.0
        for i, col in enumerate(colors):
            draw.pieslice(bbox, start + i * step, start + (i + 1) * step, fill=col)
    draw.ellipse(bbox, outline=(255, 255, 255), width=2)


def cluster_markers(uvs: dict[str, np.ndarray], valids: dict[str, np.ndarray], wh, merge_px=MERGE_PX):
    """Per timestep, merge methods whose projected points sit on top of each other."""
    w, h = wh
    clusters = []
    for k in range(HORIZON):
        items = []
        for m in METHOD_ORDER:
            uv, ok = uvs[m], valids[m]
            if k >= len(uv) or not bool(ok[k]):
                continue
            u, v = float(uv[k, 0]), float(uv[k, 1])
            if abs(u) > 8000 or abs(v) > 8000:
                continue
            if not (0 <= u < w and 0 <= v < h):
                continue
            items.append((m, np.array([u, v], dtype=np.float64)))
        used = [False] * len(items)
        for i, (mi, pi) in enumerate(items):
            if used[i]:
                continue
            group = [(mi, pi)]
            used[i] = True
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                mj, pj = items[j]
                if math.hypot(*(pj - pi)) <= merge_px:
                    group.append((mj, pj))
                    used[j] = True
            colors = [COLORS[m] for m, _ in sorted(group, key=lambda x: METHOD_ORDER.index(x[0]))]
            cxy = np.mean([p for _, p in group], axis=0)
            clusters.append((cxy[0], cxy[1], colors))
    return clusters


def draw_legend(draw, ade, n_acc, font):
    x, y = 18, 18
    rows = [
        ("vla", ade.get("vla")),
        ("draft", ade.get("draft")),
        ("spec", ade.get("spec")),
        ("gt", None),
    ]
    labels = []
    max_w = 0
    for key, val in rows:
        label = LABELS[key]
        if val is not None:
            label = f"{label}   ADE {val:.2f} m"
        labels.append((key, label))
        max_w = max(max_w, _text_size(draw, label, font)[0])
    extra = f"accepted prefix  n_acc = {n_acc} / 6"
    max_w = max(max_w, _text_size(draw, extra, font)[0])
    row_h = max(48, _text_size(draw, "Ag", font)[1] + 18)
    swatch = 32
    pad_x, pad_y = 18, 14
    box_w = pad_x + swatch + 16 + max_w + 22
    box_h = pad_y + row_h * (len(rows) + 1) + 10
    draw.rectangle((x, y, x + box_w, y + box_h), fill=(12, 14, 18), outline=(80, 80, 80), width=2)
    yy = y + pad_y
    for key, label in labels:
        cy = yy + row_h // 2
        draw.ellipse((x + pad_x, cy - swatch // 2, x + pad_x + swatch, cy + swatch // 2), fill=COLORS[key], outline=(255, 255, 255), width=2)
        draw.text((x + pad_x + swatch + 14, yy + 4), label, fill=(245, 245, 245), font=font, stroke_width=1, stroke_fill=(12, 14, 18))
        yy += row_h
    draw.text((x + pad_x + swatch + 14, yy + 4), extra, fill=(200, 200, 200), font=font, stroke_width=1, stroke_fill=(12, 14, 18))


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

    font = load_font(FONT_BEV)
    for m in (10, 20, 30):
        if m > extent:
            continue
        r = m / extent * (size - 2 * pad)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 255, 255, 50), width=1)
        d.text((cx + 6, cy - r - 22), f"{m}m", fill=(230, 230, 230, 220), font=font)
    d.line([(pad, cy), (size - pad, cy)], fill=(255, 255, 255, 40), width=1)
    d.polygon([(cx, cy - 10), (cx - 6, cy + 4), (cx + 6, cy + 4)], fill=(230, 230, 230, 220))
    widths = {"gt": 2, "vla": 4, "draft": 4, "spec": 5}
    for key in METHOD_ORDER:
        xy = xy_dict.get(key)
        if xy is None:
            continue
        pts = [(cx, cy)] + [to_px(p) for p in xy]
        col = COLORS[key] + (255,)
        d.line(pts, fill=col, width=widths[key])
    # merged pies at each waypoint in BEV
    uvs = {m: np.array([to_px(p) for p in xy_dict[m]]) for m in METHOD_ORDER}
    valids = {m: np.ones(HORIZON, dtype=bool) for m in METHOD_ORDER}
    for x, y, colors in cluster_markers(uvs, valids, (size, size), merge_px=10):
        draw_split_marker(d, x, y, [c + (255,) if len(c) == 3 else c for c in colors], r=8)
    d.rectangle((0, 0, size - 1, size - 1), outline=(255, 255, 255, 70), width=1)
    d.text((10, 8), "BEV  +X↑ +Y←", fill=(240, 240, 240, 230), font=font)
    return img


def l2_ade(pred, gt) -> float:
    d = np.linalg.norm(pred[:HORIZON] - gt[:HORIZON], axis=-1)
    return float(d.mean())


def render_one(img_path, cam, trajs, title, ade, n_acc, z_ego):
    img = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(img)
    font = load_font(FONT_LEGEND)
    font_s = load_font(FONT_CAPTION)
    wh = img.size
    line_style = (("gt", 4, True), ("vla", 6, False), ("draft", 6, False), ("spec", 7, False))
    uvs, valids = {}, {}
    for key, width, dashed in line_style:
        xy = trajs[key]
        dense = densify_xy(xy, n_per=14)
        uv, valid = project_ego_xy(dense, cam, z_ego)
        draw_polyline(d, uv, valid, wh, COLORS[key], width, dashed=dashed)
        uvs[key], valids[key] = project_ego_xy(xy, cam, z_ego)
    for x, y, colors in cluster_markers(uvs, valids, wh):
        draw_split_marker(d, x, y, colors, r=POINT_R)
    draw_legend(d, ade, n_acc, font)
    cap_h = _text_size(d, title, font_s)[1]
    d.text(
        (18, img.height - cap_h - 18),
        title,
        fill=(255, 255, 255),
        font=font_s,
        stroke_width=3,
        stroke_fill=(0, 0, 0),
    )
    bev = draw_bev(trajs, size=400)
    img.paste(bev, (img.width - bev.width - 12, 12), bev)
    return img


def write_index(out_dir: Path, items: list[dict], title: str):
    cards = []
    for it in items:
        rel = it["file"]
        cap = f'{it["i"]:04d}  {it["token"][:8]}  {it["cmd"]}  n_acc={it["n_acc"]}'
        cards.append(
            f'<a class="c" href="{rel}"><img loading="lazy" src="{rel}" alt="">'
            f"<div>{cap}<br>VLA {it['ade_vla']:.2f} · draft {it['ade_draft']:.2f} · ours {it['ade_spec']:.2f}</div></a>"
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
<div class="g">{"".join(cards)}</div>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def pad16(t: torch.Tensor) -> torch.Tensor:
    t = t.float().view(-1)
    if t.numel() < 16:
        t = torch.nn.functional.pad(t, (0, 16 - t.numel()))
    return t[:16]


@torch.no_grad()
def infer_draft_spec(tokens, row_by_tok, head, cfg, device):
    rebuilt = extras_for_tokens(tokens)
    hs, extra, speed, vla, gt = [], [], [], [], []
    cmds = []
    for tok in tokens:
        r = row_by_tok[tok]
        hs.append(r["hidden"].float().view(-1))
        if tok in rebuilt:
            e, s, g = rebuilt[tok]
            extra.append(pad16(e))
            speed.append(s.view(-1)[:1].float())
            g = g[:6].float()
            if g.abs().sum() < 0.05:
                g = torch.as_tensor(r["gt"])[:6].float()
            gt.append(g)
        else:
            extra.append(pad16(r["extra"]))
            speed.append(r["speed"].view(-1)[:1].float())
            gt.append(torch.as_tensor(r["gt"])[:6].float())
        vla.append(r["teacher"][:6].float())
        cmds.append(str(r.get("cmd", "")))
    hs = torch.stack(hs)
    extra = torch.stack(extra)
    speed = torch.stack(speed).view(-1)
    vla = torch.stack(vla)
    gt = torch.stack(gt)
    drafts = []
    for i in range(0, hs.shape[0], 64):
        pred = head(
            hs[i : i + 64].to(device),
            extra=extra[i : i + 64].to(device),
            speed=speed[i : i + 64].to(device),
        )
        drafts.append(pred.cpu())
    draft = torch.cat(drafts, dim=0)[:, :HORIZON]
    specs, naccs = [], []
    for i in range(draft.shape[0]):
        spec, n, _ = spec_from_draft(draft[i].numpy(), vla[i].numpy(), cfg, horizon=HORIZON)
        specs.append(torch.as_tensor(spec))
        naccs.append(int(n))
    spec = torch.stack(specs).float()
    return vla, draft, spec, gt, cmds, naccs


def load_tokens(cfg: dict, n: int, ids: str) -> list[str]:
    if cfg["tokens"] == "list":
        token_src = None
        for cand in (
            TOKEN_LIST,
            OUT_ROOT / "manifest.json",
        ):
            if cand.exists():
                token_src = cand
                break
        if token_src is None:
            raise SystemExit("no frame token list found (run with --model opendrive first)")
        prev = json.loads(token_src.read_text())
        if isinstance(prev, dict) and "tokens" in prev:
            all_tokens = list(prev["tokens"])
        else:
            all_tokens = [x["token"] for x in prev]
        print(f"[tokens] {token_src.name}", flush=True)
    elif cfg["tokens"] == "slice1500":
        blob = torch.load(cfg["cache"], map_location="cpu", weights_only=False)
        rows = blob["rows"]
        all_tokens = [rows[i]["token"] for i in range(1500, min(1500 + max(n, 1), len(rows)))]
        print(f"[tokens] OpenDrive cache rows [1500, {1500 + len(all_tokens)})", flush=True)
    else:
        raise ValueError(cfg["tokens"])
    tokens = all_tokens[:n]
    if ids.strip():
        pick = [int(x) for x in ids.split(",") if x.strip()]
        tokens = [all_tokens[i] for i in pick]
    return tokens


def write_parent_index(root: Path):
    cards = []
    for key, cfg in MODELS.items():
        d = root / key
        n = len(list(d.glob("*.jpg"))) if d.exists() else 0
        if n == 0:
            continue
        cards.append(
            f'<a class="c" href="{key}/index.html"><h2>{cfg["title"]}</h2>'
            f"<div>{n} frames · VLA / Draft / Ours / GT</div></a>"
        )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Front-camera trajectories</title>
<style>
body{{margin:40px;background:#111;color:#eee;font:16px/1.5 sans-serif}}
a.c{{display:block;margin:12px 0;padding:18px 20px;background:#1a1d22;color:#eee;text-decoration:none;border-radius:8px}}
h2{{margin:0 0 6px}}
</style>
<h1>Front-camera trajectories</h1>
{''.join(cards)}
"""
    (root / "index.html").write_text(html, encoding="utf-8")


def run_model(name: str, n: int, ids: str, z_ego: float, out_root: Path):
    cfg = MODELS[name]
    title_prefix = cfg["title"]
    tokens = load_tokens(cfg, n, ids)
    cache = torch.load(cfg["cache"], map_location="cpu", weights_only=False)
    row_by_tok = {r["token"]: r for r in cache["rows"]}
    missing = [t for t in tokens if t not in row_by_tok]
    if missing:
        raise SystemExit(f"{name}: missing {len(missing)} tokens in cache")

    blob = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    head = DraftHead(int(blob["hidden_size"]), horizon=HORIZON, extra_size=int(blob.get("extra_size", 16)))
    head.load_state_dict(blob["state_dict"])
    head.eval()
    device = torch.device("cpu")
    accept_cfg = AcceptConfig(lon_m=0.50, lat_m=0.30)
    print(f"[{name}] infer draft + spec  n={len(tokens)}", flush=True)
    vla, draft, spec, gt, cmds, naccs = infer_draft_spec(tokens, row_by_tok, head, accept_cfg, device)
    print(
        f"[{name}] mean n_acc={float(np.mean(naccs)):.2f}  "
        f"ade VLA={l2_ade(vla.numpy(), gt.numpy()):.3f} "
        f"draft={l2_ade(draft.numpy(), gt.numpy()):.3f} "
        f"spec={l2_ade(spec.numpy(), gt.numpy()):.3f}",
        flush=True,
    )

    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    out_dir = out_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    items = []
    skipped = 0
    vla_np, draft_np, spec_np, gt_np = vla.numpy(), draft.numpy(), spec.numpy(), gt.numpy()
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
        trajs = {"gt": gt_np[i], "vla": vla_np[i], "draft": draft_np[i], "spec": spec_np[i]}
        ade = {
            "vla": l2_ade(vla_np[i], gt_np[i]),
            "draft": l2_ade(draft_np[i], gt_np[i]),
            "spec": l2_ade(spec_np[i], gt_np[i]),
        }
        cmd = cmds[i]
        title = f"{title_prefix}  {tok[:12]}  {cmd}  n_acc={naccs[i]}"
        vis = render_one(img_path, cam, trajs, title, ade, naccs[i], z_ego)
        fname = f"{i:04d}_{tok[:8]}.jpg"
        vis.save(out_dir / fname, quality=88, optimize=True)
        items.append(
            {
                "i": i,
                "token": tok,
                "file": fname,
                "cmd": cmd,
                "n_acc": naccs[i],
                "ade_vla": ade["vla"],
                "ade_draft": ade["draft"],
                "ade_spec": ade["spec"],
                "spread": float(
                    np.linalg.norm(vla_np[i][-1] - draft_np[i][-1])
                    + np.linalg.norm(spec_np[i][-1] - draft_np[i][-1])
                ),
            }
        )
        if (i + 1) % 50 == 0 or i + 1 == len(tokens):
            print(f"[{name}] wrote {len(items)}/{len(tokens)} skip={skipped}", flush=True)

    torch.save({"tokens": tokens, "vla": vla, "draft": draft, "spec": spec, "gt": gt, "n_acc": naccs}, out_dir / "trajs.pt")
    write_index(out_dir, items, f"{title_prefix}  ·  VLA / Draft / Ours / GT")
    (out_dir / "manifest.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    hi = out_dir / "highlights"
    hi.mkdir()
    top = sorted(items, key=lambda x: -x["spread"])[:24]
    mixed = sorted(items, key=lambda x: (0 if 1 <= x["n_acc"] <= 4 else 9, -x["spread"]))[:12]
    seen = set()
    ranked = []
    for it in top + mixed:
        if it["file"] in seen:
            continue
        seen.add(it["file"])
        ranked.append(it)
    for rank, it in enumerate(ranked[:24]):
        shutil.copy2(out_dir / it["file"], hi / f"{rank:02d}_{it['file']}")
    write_index(
        hi,
        [{**it, "file": f"{rank:02d}_{it['file']}"} for rank, it in enumerate(ranked[:24])],
        f"{title_prefix} highlights",
    )
    print(f"[{name}] done {len(items)} -> {out_dir}", flush=True)


def _refresh_copies(out_dir: Path):
    hi = out_dir / "highlights"
    if hi.exists():
        for p in hi.glob("*.jpg"):
            parts = p.name.split("_", 1)
            src = out_dir / (parts[1] if len(parts) == 2 else p.name)
            if src.exists():
                shutil.copy2(src, p)
    show = out_dir.parent / "show"
    if show.exists():
        for p in show.glob("*.jpg"):
            name = p.name
            if not name.startswith("auto"):
                continue
            stem = name[len("auto") :]
            if "_" in stem:
                continue
            cands = sorted(out_dir.glob(f"{Path(stem).stem}_*.jpg"))
            if cands:
                shutil.copy2(cands[0], p)
            else:
                src = out_dir / name
                if src.exists():
                    shutil.copy2(src, p)


def rerender_model(name: str, z_ego: float, out_root: Path):
    cfg = MODELS[name]
    title_prefix = cfg["title"]
    out_dir = out_root / name
    traj_path = out_dir / "trajs.pt"
    man_path = out_dir / "manifest.json"
    if not traj_path.exists() or not man_path.exists():
        raise SystemExit(f"{name}: missing {traj_path} or {man_path}")
    blob = torch.load(traj_path, map_location="cpu", weights_only=False)
    items_old = json.loads(man_path.read_text())
    by_tok = {it["token"]: it for it in items_old}
    tokens = list(blob["tokens"])
    vla_np = blob["vla"].numpy()
    draft_np = blob["draft"].numpy()
    spec_np = blob["spec"].numpy()
    gt_np = blob["gt"].numpy()
    naccs = [int(x) for x in blob["n_acc"]]
    infos = load_infos(VAL_PKL)
    tok2i = token_index(infos)
    items = []
    skipped = 0
    print(f"[{name}] rerender n={len(tokens)} larger fonts", flush=True)
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
        trajs = {"gt": gt_np[i], "vla": vla_np[i], "draft": draft_np[i], "spec": spec_np[i]}
        prev = by_tok.get(tok, {})
        ade = {
            "vla": float(prev.get("ade_vla", l2_ade(vla_np[i], gt_np[i]))),
            "draft": float(prev.get("ade_draft", l2_ade(draft_np[i], gt_np[i]))),
            "spec": float(prev.get("ade_spec", l2_ade(spec_np[i], gt_np[i]))),
        }
        cmd = str(prev.get("cmd", ""))
        title = f"{title_prefix}  {tok[:12]}  {cmd}  n_acc={naccs[i]}"
        vis = render_one(img_path, cam, trajs, title, ade, naccs[i], z_ego)
        fname = prev.get("file") or f"{i:04d}_{tok[:8]}.jpg"
        vis.save(out_dir / fname, quality=88, optimize=True)
        items.append({**prev, "i": i, "token": tok, "file": fname, "cmd": cmd, "n_acc": naccs[i], **{f"ade_{k}": ade[k] for k in ("vla", "draft", "spec")}})
        if (i + 1) % 50 == 0 or i + 1 == len(tokens):
            print(f"[{name}] wrote {len(items)}/{len(tokens)} skip={skipped}", flush=True)
    write_index(out_dir, items, f"{title_prefix}  ·  VLA / Draft / Ours / GT")
    (out_dir / "manifest.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    _refresh_copies(out_dir)
    print(f"[{name}] rerender done {len(items)} -> {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--ids", default="", help="comma-separated indices into the token list (smoke)")
    p.add_argument("--z", type=float, default=0.25)
    p.add_argument("--model", choices=["opendrive", "rnb", "both"], default="both")
    p.add_argument("--out", default=str(OUT_ROOT))
    p.add_argument("--rerender", action="store_true", help="redraw existing trajs.pt with current fonts")
    args = p.parse_args()
    out_root = Path(args.out)
    names = ["opendrive", "rnb"] if args.model == "both" else [args.model]
    for name in names:
        if args.rerender:
            rerender_model(name, args.z, out_root)
        else:
            run_model(name, args.n, args.ids, args.z, out_root)
    write_parent_index(out_root)


if __name__ == "__main__":
    main()
