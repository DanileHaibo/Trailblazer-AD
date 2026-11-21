# Trailblazer-AD

Waypoint-level speculative decoding for autoregressive driving VLAs.

Paper implementation: **TrajSpecDec**. This repo holds the source, the
reproduction scripts, and the result JSONs for both AR teachers.

## Two teachers

| Teacher | Weights | Decode unit | Horizon |
|---|---|---|---|
| **OpenDriveVLA-0.5B + UniAD** | `OpenDriveVLA/OpenDriveVLA-0.5B` (gated) | textual waypoints `[(x,y),...]` | 6 pts / 3 s |
| **R&B-EnCoRe (Qwen3-VL-4B)** | `stanfordasl/nuscenes-waypoints-model` | textual waypoints `[(x,y),...]` | 6 pts / 3 s |

Both are AR textual-waypoint teachers: each waypoint costs ~11 BPE tokens, so
there is a long serial decode to skip. Only AR teachers benefit from speculative
decoding — DiT / flow-matching planners emit a whole trajectory at once.

## Method

```
prefill (shared, frozen teacher)
  -> last hidden + 16-d ego kinematics
  -> DraftHead: bicycle prior (speed/accel/yaw-rate, command + kappa bias) + residual MLP
  -> K absolute waypoints in one shot
  -> one teacher-force forward verifies the prefix
  -> accept longest prefix with |dx| <= 0.50 m and |dy| <= 0.30 m
  -> first reject: take teacher waypoint, teacher AR-decodes the suffix
```

Acceptance is geometric, not BPE/JSON exact match: a 2-decimal rounding
difference or one comma must not kill a geometrically valid prefix. The draft is
supervised on **teacher** trajectories, not ground truth.

## Results

Live wall clock on nuScenes val, CUDA-synchronized, vs the official `generate()`:

| Metric | OpenDriveVLA-0.5B | R&B-EnCoRe-4B |
|---|---|---|
| frames | 12 (start 1500) | 6015 |
| mean accepted waypoints / 6 | 3.33 | **4.14** |
| all 6 accepted | 50% | **53.2%** |
| official `generate()` | 1640 ms | 2349 ms |
| TrajSpecDec E2E | 996 ms | 862 ms |
| **speedup** | **1.65x** | **2.73x** |
| draft latency | 1.0 ms | 1.1 ms |
| verify latency | 23 ms | 43 ms |

Draft heads (offline, 15% hash holdout of nuScenes val 6019):

| Metric | OpenDriveVLA-0.5B | R&B-EnCoRe-4B |
|---|---|---|
| mean accepted waypoints / 6 | 4.11 | 3.81 |
| >= 2 accepted | 85.2% | 93.2% |
| all 6 accepted | 38.8% | 26.6% |
| decode-step speedup | 3.90x | 3.47x |

`spec` open-loop L2 stays level with the teacher, i.e. the accepted draft prefix
does not damage planning (see `results/full_val_6019.json`).

## Reproducing

### Environment

```bash
OD=/path/to/env/bin/python     # torch 2.5.1, transformers 4.57.1, llava, mmengine  (OpenDriveVLA)
RNB=/path/to/env/bin/python    # torch 2.6.0, transformers 5.x                       (Qwen3-VL / R&B)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

- **R&B needs transformers >= 5** for Qwen3-VL (`AutoModelForImageTextToText`).
  The continue path must pass `cache_position`, otherwise mRoPE restarts at 0
  and the teacher-force tail emits garbage coordinates.
- **OpenDriveVLA needs a shapely>=2 patch**: `mask_for_polygons` in
  `projects/mmdet3d_plugin/datasets/eval_utils/map_api.py` must flatten
  `MultiPolygon` via `list(geom.geoms)`, otherwise it raises
  `'MultiPolygon' object is not iterable`.

Teacher paths are env-overridable:

```bash
export TRAJSPEC_OPENDRIVE_CKPT=/path/to/OpenDriveVLA-0.5B
export TRAJSPEC_OPENDRIVE_REPO=/path/to/OpenDriveVLA
export TRAJSPEC_RNB_CKPT=/path/to/nuscenes-waypoints-model
export TRAJSPEC_NUSC_ROOT=/path/to/nuscenes
```

### 1) Offline draft head (minutes, no UniAD / 4B rerun)

Uses cached teacher hidden states:

```bash
# OpenDriveVLA: official val 6019, hash 15% holdout
$RNB scripts/run_full_specdec.py --name opendrive_nusc \
  --train-cache results/cache_opendrive_nusc_val.pt \
  --target-accept 4.009 --out results/repro2_specdec_opendrive.json

# R&B-EnCoRe: same split, same recipe
$RNB scripts/run_full_specdec.py --name rnb_waypoints_nusc \
  --train-cache /path/to/cache_rnb_nusc_val.pt \
  --target-accept 3.5 --out results/specdec_waypoints_rnb_retrain.json
```

### 2) Live wall clock: OpenDriveVLA + UniAD

```bash
$OD scripts/run_live_specdec_opendrive.py \
  --n 12 --warmup 2 --start 1500 --out results/repro_live_opendrive.json
```

`--start 1500` indexes the OpenDrive cache order (skips early mixed-coordinate
samples).

### 3) Live wall clock: R&B-EnCoRe, full val

```bash
# two shards, then merge
CUDA_VISIBLE_DEVICES=0 $RNB scripts/run_live_specdec_rnb.py \
  --n 3010 --warmup 2 --stride 1 --shard 0 --num-shards 2 --out results/shard0.json &
CUDA_VISIBLE_DEVICES=1 $RNB scripts/run_live_specdec_rnb.py \
  --n 3010 --warmup 2 --stride 1 --shard 1 --num-shards 2 --out results/shard1.json &
wait
$RNB scripts/merge_live_rnb_shards.py results/shard0.json results/shard1.json \
  --out results/live_waypoints_rnb_full_6019.json
```

## Layout

```
trajspec/            core library
  accept.py          geometric longest-prefix accept
  draft_head.py      bicycle prior + residual MLP
  spec_loop.py       prefill -> draft -> verify -> commit
  opendrive_live.py  OpenDriveVLA live specdec (UniAD reset, TeacherDecoder)
  metrics.py         ST-P3 L2 at 1/2/3 s
  serialize.py       textual waypoint format/parse
  adapters/          OpenDriveVLA and R&B-EnCoRe runtimes
  nusc_data.py       nuScenes VAD temporal pkl packing
scripts/             caching, draft training, live eval, baselines
tests/               accept + baseline unit tests
results/             paper and reproduction JSONs (teacher caches excluded)
weights/             trained draft heads (3-7 MB each) + loading notes
survey/              literature survey of AR driving-VLA acceleration
```

See `TRANSFER_OD_RNB.md` for moving this setup to another machine.

## Limitations

- Reported `decode_speedup` (3.9x / 3.5x) is **decode-step accounting**
  (1 parallel verify + remaining serial steps), not wall clock. End-to-end
  wall-clock speedup is 1.65x / 2.73x; the gap is the shared prefill.
- Both draft heads were trained on 85% of nuScenes val and evaluated on the
  same split's holdout. Full-val accept is therefore ~4% optimistic relative to
  a fully held-out 903-frame number.
- Verify is currently one teacher-force forward over the draft tokens; a
  token-level parallel verify is not implemented.
- The R&B tail path does not yet apply the `min_accept` fallback that
  OpenDriveVLA uses, so frames with 1 accepted waypoint can degrade ADE.

## Acceleration baselines and where the time actually goes

`survey/ACCEL_SURVEY.md` is the literature survey of acceleration methods for
AR driving VLAs (Spec-VLA, KERV, PD-VLA, FastDriveVLA, Prune2Drive, FastV,
DeeAD, Orion-Lite, ...), graded by publication status and by which pipeline
stage each one attacks.

`survey/AD_NATIVE_ACCEL.md` narrows this to **autonomous-driving-native** work
only (visual-token pruning, CoT compression, non-autoregressive decoding, KV
reuse, distillation, serving).

`survey/SKIPLAYER_AND_PRUNING.md` is the deep dive on the two reduce-compute
families: dynamic depth (early exit vs layer skipping vs MoD vs Router-Tuning vs
static layer pruning) and token pruning. Both are falsified on this teacher -- see
the "why everything went to zero" section, which ties back to decode being
weight-bound rather than KV-length-bound.

Profile of OpenDriveVLA-0.5B + UniAD (`scripts/profile_opendrive_stages.py`,
L20, n=10, CUDA-synchronized):

| stage | ms | share of E2E |
|---|---|---|
| UniAD vision tower | 411.7 | 24.9% |
| feature projection | 0.5 | 0.03% |
| prompt assembly | 1.6 | 0.1% |
| LM prefill (620 tok) | 30.6 | 1.8% |
| AR decode (66 tok @ 18.6 ms) | 1211.8 | **73.2%** |

Decode dominates; prefill is 1.8%. But the intuitive follow-up -- "so pruning
visual tokens caps out at 1.8%" -- is **wrong twice over**: the visual tokens are
also in the KV cache (so they would also tax every decode step), and yet measurement
shows the real ceiling is **zero**. Decode is weight-bound, not KV-length-bound:
truncating the prompt from 625 to 200 tokens leaves a decode step unchanged
(23.39 -> 23.48 ms), and an oracle that deletes *every* visual token still gives
1.002x E2E (`results/kvlen_scaling_opendrive.json`,
`results/oracle_prune_opendrive.json`).

### The main comparison: 4 families, same frames

`scripts/run_unified_accel_table.py` -> `results/unified_accel_table.json`,
rendered to `results/tab_unified_accel.tex`.

n=40 (34 GT-valid), one L20, CUDA-synchronized. **The UniAD tower runs once per
frame and its features are shared by every row**, so only the decode strategy
differs; E2E includes that shared tower cost.

| method | attacks | E2E ms | speedup | ADE | L2@1s | L2@2s | L2@3s |
|---|---|---|---|---|---|---|---|
| AR greedy | --- | 1639 | 1.00x | 0.691 | 0.262 | 0.433 | 0.691 |
| Spec-VLA (exact) | accept rule | 1700 | **0.96x** | 0.691 | 0.262 | 0.433 | 0.691 |
| Spec-VLA (pose-relax) | accept rule | 908 | 1.81x | 0.750 | 0.291 | 0.478 | 0.750 |
| PD-VLA (parallel) | serial dependency | 1289 | 1.27x | **29.235** | 1.887 | 20.637 | 29.235 |
| Pruning (oracle 25%) | visual tokens | 1639 | **1.00x** | 0.659 | 0.254 | 0.414 | 0.659 |
| **TrajSpecDec (ours)** | accept + seam | 957 | **1.71x** | **0.692** | 0.274 | 0.441 | 0.692 |

Spec-VLA (exact) reproduces AR greedy's ADE *exactly* (0.691) because every frame is
a full reject and falls back to the official `generate()`. That is also the check that
the fallback path is bit-exact: `n_accept == 0` frames measure
`ADE(ours, AR) = 0.000000`.

Token-exact SD gives no speedup at all (1.00x). PD-VLA is unusable (29 m ADE,
0% convergence). Pruning gives exactly 1.00x -- its lower ADE is the output
drifting, not an improvement, since its latency is identical to AR.

### Two bugs that invalidated every earlier AR-vs-spec number

Both are fixed; both must stay documented because they silently corrupt results.

**1. The SD verify path was not the same decoder as `generate()`.**
`generation_config.json` sets `repetition_penalty = 1.1`, which `generate()`
applies even at `do_sample=False`; the SD path used a bare argmax. Token agreement
with `generate()` was 0.13 (all 10 frames diverged within 20 tokens), and frames
with `n_accept == 0` -- which should reproduce the teacher verbatim -- still
differed by 0.15 m ADE. Fix: `TeacherDecoder` in `trajspec/opendrive_live.py`.
(This also surfaced that `eos_token_id` is a list, so the old `tid == eos` test was
never true.)

**2. UniAD's vision tower carries persistent temporal state.**
`prev_frame_info` (`prev_bev` / `prev_pos` / `scene_token`) and
`track_base.max_obj_id` survive across forwards. The same frame prefilled twice
around a `generate()` call differs by 0.65 in logit space; and even with state
reset, three consecutive tower calls on one frame yield track counts 30/31/30.
Sampling by index is therefore not reproducible. Fix: run the tower once per frame
and pass the result to every method via `uniad_pth`, plus `reset_uniad_temporal()`
as a safety net.

### A third issue, exposed only after the first two were fixed

Splicing from a **single** accepted waypoint is what cost us accuracy: 8 frames with
`n_accept == 1` carried +0.985 m ADE, because `seam_blend` extrapolates from `d[0]`
*and* the tail is decoded conditioned on that same wrong prefix. With
`min_accept=2` (below it we fall back to a full reject, accepting the latency cost)
overall ADE went **0.929 -> 0.707** against AR's 0.691.

Per-accept-length, after the fix:

| accept /6 | frames | AR ADE | ours ADE | ours E2E ms |
|---|---|---|---|---|
| 0 | 2 | 0.604 | 0.694 | 1675 |
| 1 | 8 | 0.668 | 0.710 | 1680 |
| 2 | 6 | 0.776 | 0.774 | 1301 |
| 4 | 1 | 0.762 | 0.737 | 904 |
| 5 | 1 | 0.306 | 0.422 | 698 |
| 6 | 16 | 0.701 | 0.698 | **477** |

Latency is bimodal: full-accept frames run at 477 ms, reject-heavy frames fall
back to AR latency. The 1.75x mean hides that.
