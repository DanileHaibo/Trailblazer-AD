#!/usr/bin/env python3
"""Live native vs Trailblazer eval for nuscenes-waypoints-model.

Same protocol as the OpenDrive live script: CUDA-synchronised wall clock, one
teacher-force verify of the draft waypoint string, serial greedy tail after the
first geometric reject.

Outputs per-frame: native generate latency, spec latency, accept count, and L2.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path("/home/xinhchen2/hhb/TrajSpecDec")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from trajspec.accept import AcceptConfig, longest_prefix_accept
from trajspec.adapters.waypoints import WaypointsRuntime, parse_waypoints, TRAJ_RE
from trajspec.draft_head import DraftHead, pack_extra, pack_speed
from trajspec.nusc_data import VAL_PKL, load_infos, pack_sample_full, token_index, lidar_to_ego_xy
from trajspec.ego_hist import ego_history_4


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def summarize(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": round(float(a.mean()), 2), "p50": round(float(np.median(a)), 2),
            "p90": round(float(np.percentile(a, 90)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--ckpt", default=str(ROOT / "results/specdec_waypoints.pt"))
    ap.add_argument("--min-accept", type=int, default=3, help="below this, fall back to native")
    ap.add_argument("--lon", type=float, default=0.5)
    ap.add_argument("--lat", type=float, default=0.3)
    ap.add_argument("--out", default=str(ROOT / "results/live_waypoints.json"))
    args = ap.parse_args()

    rt = WaypointsRuntime()
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    K = int(blob.get("horizon", 6))
    head = DraftHead(int(blob["hidden_size"]), horizon=K, extra_size=16).cuda()
    head.load_state_dict(blob.get("state_dict") or blob.get("head"))
    head.eval()
    cfg = AcceptConfig(lon_m=args.lon, lat_m=args.lat)

    infos = load_infos(VAL_PKL)
    t2i = token_index(infos)
    tok = rt.processor.tokenizer
    eos = tok.eos_token_id

    rows = []
    seen = 0
    for idx in range(args.start, len(infos), args.stride):
        s = pack_sample_full(infos, idx, t2i)
        if s is None:
            continue
        gt = np.asarray(s["gt"][:K], dtype=np.float32)
        # VAD pkl ships only 2 history steps; the checkpoint was trained on 4.
        _h4 = ego_history_4(infos, t2i[s["token"]], t2i, 4)
        if _h4 is not None:
            s = dict(s)
            s["hist"] = lidar_to_ego_xy(_h4)
        timed = seen >= args.warmup

        # ---- native ----
        t_nat, nat_xy = sync_ms(lambda: rt.predict(s))

        # ---- trailblazer ----
        inp = {k: (v.to(rt.device) if torch.is_tensor(v) else v) for k, v in rt._inputs(s).items()}
        plen = int(inp["input_ids"].shape[1])
        t_pre, pre = sync_ms(lambda: rt.model(**inp, use_cache=True, output_hidden_states=True))
        hidden = pre.hidden_states[-1][:, -1].float()
        t_dr, draft = sync_ms(lambda: head(hidden, extra=pack_extra(s).view(1, -1).to(rt.device),
                                           speed=pack_speed(s).view(-1).to(rt.device)))
        draft_xy = draft[0, :K].float().detach().cpu()
        # The model's own convention is (lateral, longitudinal) while draft_xy
        # is (forward, left), so negate/swap when serialising back to text.
        draft_text = "Planned Trajectory:\n[" + ", ".join(
            f"({-float(p[1]):.2f},{float(p[0]):.2f})" for p in draft_xy) + "]"
        ids = tok(draft_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(rt.device)

        # Verify in ONE batched forward. Qwen3-VL uses 3-D mRoPE: the image
        # patches collapse onto a shorter position grid, so the continuation
        # must start at the *prefill's last t-position + 1* (not at plen).
        # Passing position_ids/mm_token_type_ids explicitly makes the batched
        # logits bit-identical to the serial decode (verified 49/49 tokens).
        n_draft = int(ids.shape[1])

        def _verify():
            pos_pre, _ = rt.model.model.get_rope_index(
                inp["input_ids"], inp["mm_token_type_ids"],
                image_grid_thw=inp.get("image_grid_thw"),
                attention_mask=inp.get("attention_mask"),
            )
            start = int(pos_pre[0, 0, -1]) + 1
            base = torch.arange(start, start + n_draft, device=rt.device).view(1, 1, -1).expand(3, 1, n_draft)
            am = torch.ones(1, plen + n_draft, device=rt.device, dtype=torch.long)
            mm = torch.zeros(1, n_draft, dtype=torch.int32, device=rt.device)
            return rt.model(
                input_ids=ids,
                past_key_values=pre.past_key_values,
                use_cache=True,
                position_ids=base,
                attention_mask=am,
                mm_token_type_ids=mm,
            )

        t_ver, ver = sync_ms(_verify)
        fl = ver.logits[0].float()
        te_ids = [int(pre.logits[0, -1].float().argmax())] + [int(fl[i].argmax()) for i in range(n_draft - 1)]
        te_text = tok.decode(te_ids, skip_special_tokens=True)
        te_xy = parse_waypoints(te_text, K)

        n_acc = longest_prefix_accept(draft_xy, te_xy, cfg)
        # Adaptive fallback: below a cutoff the serial tail costs more than just
        # letting the teacher run, so we report native for that frame (a real
        # deployment would decide from the draft/verify statistics it already
        # has, before paying for the tail).
        if n_acc < args.min_accept:
            rec = {
                "token": s["token"], "n_accept": int(n_acc), "fallback": True,
                "t_native_ms": round(t_nat, 2), "t_spec_ms": round(t_nat, 2),
                "t_prefill_ms": round(t_pre, 2), "t_draft_ms": round(t_dr, 2),
                "t_verify_ms": round(t_ver, 2), "t_tail_ms": 0.0,
                "l2_native": round(float(np.linalg.norm(nat_xy[:K] - gt, axis=-1).mean()), 4),
                "l2_spec": round(float(np.linalg.norm(nat_xy[:K] - gt, axis=-1).mean()), 4),
            }
            seen += 1
            if timed:
                rows.append(rec)
                print(f"[{len(rows)}/{args.n}] FALLBACK acc={n_acc} -> native {rec['t_native_ms']:.0f}ms", flush=True)
            if len(rows) >= args.n:
                break
            continue
        tail_ms = 0.0
        if n_acc >= K:
            spec_xy = draft_xy.numpy()
        else:
            # Seed the serial tail with the *accepted* prefix only. A single
            # forward over draft[:n_acc+1] reproduces the teacher's KV at the
            # correction point (verify already proved those tokens match).
            keep = _prefix_token_len(tok, draft_xy, n_acc)
            seed = ids[:, :keep]

            def _tail_teacher_full():
                """Teacher's own full answer (no draft prefix).

                Resuming from the draft prefix changes the teacher's context and
                yields a suffix inconsistent with where the draft left off, which
                showed up as L2 exploding to ~10 m on partial-accept frames.
                Decoding from the clean prompt reproduces exactly what the teacher
                would output, so the splice is draft[:n_acc] + teacher[n_acc:].
                """
                pos_pre, _ = rt.model.model.get_rope_index(
                    inp["input_ids"], inp["mm_token_type_ids"],
                    image_grid_thw=inp.get("image_grid_thw"),
                    attention_mask=inp.get("attention_mask"),
                )
                p_ = int(pos_pre[0, 0, -1]) + 1
                logits = pre.logits[0, -1].float()
                past = pre.past_key_values
                gen = []
                for _ in range(120):
                    nid = int(logits.argmax())
                    if eos is not None and nid == eos:
                        break
                    gen.append(nid)
                    if tok.decode(gen, skip_special_tokens=True).count(")") >= K:
                        break
                    o = rt.model(
                        input_ids=torch.tensor([[nid]], device=rt.device),
                        past_key_values=past, use_cache=True,
                        position_ids=torch.full((3, 1, 1), p_, device=rt.device, dtype=torch.long),
                        mm_token_type_ids=torch.zeros(1, 1, dtype=torch.int32, device=rt.device),
                    )
                    logits = o.logits[0, -1].float()
                    past = o.past_key_values
                    p_ += 1
                return tok.decode(gen, skip_special_tokens=True), len(gen)

            t_tail, (tail_text, n_tail_tok) = sync_ms(_tail_teacher_full)
            tail_ms = t_tail
            txy = parse_waypoints(tail_text, K).numpy()
            sp = draft_xy.numpy().copy()
            sp[n_acc:] = txy[n_acc:]
            spec_xy = sp

        spec_ms = t_pre + t_dr + t_ver + tail_ms
        seen += 1
        if not timed:
            continue
        rec = {
            "token": s["token"],
            "n_accept": int(n_acc),
            "t_native_ms": round(t_nat, 2),
            "t_spec_ms": round(spec_ms, 2),
            "t_prefill_ms": round(t_pre, 2),
            "t_draft_ms": round(t_dr, 2),
            "t_verify_ms": round(t_ver, 2),
            "t_tail_ms": round(tail_ms, 2),
            "l2_native": round(float(np.linalg.norm(nat_xy[:K] - gt, axis=-1).mean()), 4),
            "l2_spec": round(float(np.linalg.norm(spec_xy[:K] - gt, axis=-1).mean()), 4),
        }
        rows.append(rec)
        print(f"[{len(rows)}/{args.n}] nat={rec['t_native_ms']:.0f} spec={rec['t_spec_ms']:.0f} "
              f"acc={rec['n_accept']} l2={rec['l2_native']:.3f}->{rec['l2_spec']:.3f}", flush=True)
        if len(rows) >= args.n:
            break

    nat = np.array([r["t_native_ms"] for r in rows])
    sp = np.array([r["t_spec_ms"] for r in rows])
    acc = np.array([r["n_accept"] for r in rows])
    out = {
        "model": "stanfordasl/nuscenes-waypoints-model (Qwen3-VL-4B)",
        "n": len(rows),
        "horizon": K,
        "accept": {"lon_m": args.lon, "lat_m": args.lat, "mean_n_accept": round(float(acc.mean()), 3),
                   "full": round(float((acc >= K).mean()), 3)},
        "latency": {"native": summarize(nat), "spec": summarize(sp)},
        "speedup_mean": round(float(nat.mean() / sp.mean()), 3),
        "speedup_median": round(float(np.median(nat / sp)), 3),
        "l2": {"native": round(float(np.mean([r["l2_native"] for r in rows])), 4),
               "spec": round(float(np.mean([r["l2_spec"] for r in rows])), 4)},
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print(f"[write] {args.out}")


def _clone(past):
    import copy

    layers = getattr(past, "layers", None)
    if layers is None:
        return copy.deepcopy(past)
    new = copy.copy(past)
    new.layers = [type(l)(keys=l.keys.clone(), values=l.values.clone())
                  if getattr(l, "keys", None) is not None else copy.copy(l) for l in layers]
    return new


def _prefix_token_len(tok, xy, n_keep):
    """Token length of the draft string truncated after n_keep waypoints.

    Must use the same (lateral, longitudinal) convention the model emits, i.e.
    (-y, x) from our (forward, left) representation, otherwise the seed cuts at
    the wrong byte and the tail decodes from a corrupted prefix.
    """
    if n_keep <= 0:
        return len(tok("Planned Trajectory:\n[", add_special_tokens=False)["input_ids"])
    inner = ", ".join(f"({-float(p[1]):.2f},{float(p[0]):.2f})" for p in xy[:n_keep])
    return len(tok(f"Planned Trajectory:\n[{inner}", add_special_tokens=False)["input_ids"])


if __name__ == "__main__":
    main()
