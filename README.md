# Trailblazer-AD

Waypoint-level speculative decoding for autoregressive driving VLAs.

Paper implementation: **TrajSpecDec**. This repo holds the source, the
reproduction scripts, and the result JSONs for both AR teachers.

## Two teachers

| Teacher | Weights | Decode unit | Horizon |
|---|---|---|---|
| **AutoVLA** (Qwen2.5-VL-3B + PDMS) | `Zewei-Zhou/AutoVLA` | physical `<action_i>` codebook tokens | 10 pts / 5 s (first 6 used) |
| **OpenDriveVLA-0.5B + UniAD** | `OpenDriveVLA/OpenDriveVLA-0.5B` (gated) | textual waypoints `[(x,y),...]` | 6 pts / 3 s |

Only AR teachers benefit from speculative decoding: DiT / flow-matching planners
emit a whole trajectory at once and have no serial decode steps to skip.

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

Offline draft head, held-out accept (lon 0.50 m / lat 0.30 m):

| Metric | AutoVLA (paper -> repro) | OpenDriveVLA (paper -> repro) |
|---|---|---|
| n | 2000 -> 2000 | 903 -> 903 |
| mean accepted waypoints / 6 | 3.757 -> **3.777** | 4.009 -> **4.110** |
| >= 2 accepted | 97.6% -> 97.9% | 85.2% -> 87.9% |
| all 6 accepted | 20.5% -> 17.5% | 38.8% -> 38.2% |
| decode-step speedup | 3.26x -> **3.30x** | 3.90x -> **4.01x** |
| spec ST-P3 L2 @1/2/3 s | 0.359/0.745/1.274 -> 0.363/0.746/1.273 | 0.697/1.232/1.881 -> 0.699/1.231/1.880 |

Live wall clock (n=12, CUDA-synchronized):

| Metric | AutoVLA (paper -> repro) | OpenDriveVLA (paper -> repro) |
|---|---|---|
| mean accepted waypoints | 4.833 -> **4.833** (per-frame identical) | 3.333 -> **3.333** (per-frame identical) |
| speedup vs official `generate()` | 1.256x -> **1.261x** | 1.637x -> **1.649x** |
| draft latency | 1.25 ms -> 1.21 ms | 0.99 ms -> 1.02 ms |
| verify latency | 39.75 ms -> 38.77 ms | 22.70 ms -> 22.91 ms |

Per-frame accept sequences match the paper exactly
(AutoVLA `[3,6,5,4,5,3,5,6,5,6,4,6]`, OpenDrive `[6,1,6,6,6,6,1,1,6,1,0,0]`).

`spec` open-loop L2 stays level with the teacher, i.e. the accepted draft prefix
does not damage planning.

## Reproducing

### Environment

```bash
UNI=/path/to/env/bin/python    # torch 2.5.1, transformers 4.57.1, llava, mmengine  (OpenDriveVLA)
QWEN=/path/to/env/bin/python   # torch 2.6.0, transformers 5.16.1                  (AutoVLA)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

- **AutoVLA requires transformers >= 5.** On 4.x, manual KV-cache incremental
  decode does not advance Qwen2.5-VL mRoPE positions, so the think segment
  repeats one token (observed `'<thinkinkinkink...'`), think length grows from
  30 to 160 tokens and accept collapses from 4.83 to 0.33. The
  `remap_qwen25vl_v5` key remapping in `trajspec/adapters/autovla.py` targets 5.x.
- **OpenDriveVLA needs a shapely>=2 patch**: `mask_for_polygons` in
  `projects/mmdet3d_plugin/datasets/eval_utils/map_api.py` must flatten
  `MultiPolygon` via `list(geom.geoms)`, otherwise it raises
  `'MultiPolygon' object is not iterable`.

### 1) Offline draft head (minutes, no 3B / UniAD rerun)

Uses cached teacher hidden states:

```bash
# AutoVLA: train 2500 / val 2000
$QWEN scripts/run_full_specdec.py --name autovla_nusc \
  --train-cache results/cache_autovla_nusc_train.pt \
  --val-cache   results/cache_autovla_nusc_val.pt \
  --target-accept 3.757 --out results/repro2_specdec_autovla.json

# OpenDriveVLA: official val 6019, hash 15% holdout
$QWEN scripts/run_full_specdec.py --name opendrive_nusc \
  --train-cache results/cache_opendrive_nusc_val.pt \
  --target-accept 4.009 --out results/repro2_specdec_opendrive.json
```

### 2) Live wall clock: AutoVLA

```bash
$QWEN scripts/run_live_specdec_autovla.py \
  --n 12 --warmup 2 --out results/repro_live_autovla.json
```

Defaults to `--ckpt results/specdec_autovla_heldout.pt`, `--stride 16 --start 0`
over the VAD val pkl. Using `specdec_autovla_full.pt` yields accept 3.92 because
it is not the checkpoint used in the paper.

### 3) Live wall clock: OpenDriveVLA + UniAD

```bash
$UNI scripts/run_live_specdec_opendrive.py \
  --n 12 --warmup 2 --start 1500 --out results/repro_live_opendrive.json
```

`--start 1500` indexes the OpenDrive cache order (skips early mixed-coordinate
samples); it is a different token list from AutoVLA's stride-16 val order.

### 4) Paper vs repro table

```bash
$QWEN repro/compare_to_paper.py
```

## Layout

```
trajspec/            core library
  accept.py          geometric longest-prefix accept
  draft_head.py      bicycle prior + residual MLP
  spec_loop.py       prefill -> draft -> verify -> commit
  action_quant.py    AutoVLA codebook <-> xy
  serialize.py       OpenDriveVLA text waypoint format/parse
  adapters/          AutoVLA (Qwen2.5-VL) and OpenDriveVLA runtimes
  nusc_data.py       nuScenes VAD temporal pkl packing
scripts/             caching, draft training, live eval, baselines
tests/               accept + baseline unit tests
results/             paper and reproduction JSONs (teacher caches excluded)
weights/             trained draft heads (3-7 MB each) + loading notes
repro/               README + paper-vs-repro comparison
survey/              literature survey of AR driving-VLA acceleration
```

## Draft head diagnostics

`diag/REPORT.md` documents why the current draft is close to its information
limit, and which improvement routes were falsified by experiment:

| idea | measured outcome |
|---|---|
| bigger head | 19M params: train accept 5.997 / **val 3.679** (base 3.885) |
| more data | 12146 out-of-domain frames: val **2.403** |
| hidden instead of kinematics | kin_only 3.778 / hidden_only 3.791 / both 3.782 |
| post-hoc bias removal | 4.070 -> 4.099, noise std 1.57 m unremovable |
| tapered thresholds | equivalent to a scaled fixed box (<0.001 ADE apart) |

The single effective knob is the accept box itself.
`results/diag_threshold_sweep.json` holds the trade-off curve for both teachers,
e.g. AutoVLA 3.778 -> 4.854 accept as the box widens 0.50/0.30 -> 1.00/0.60 m.

## Limitations

- Reported `decode_speedup` (3.3x / 4.0x) is **decode-step accounting**
  (1 parallel verify + remaining serial steps), not wall clock. End-to-end
  wall-clock speedup is 1.26x / 1.65x; the gap is prefill and, for AutoVLA,
  the CoT segment.
- AutoVLA's `think` block (~890 ms, ~60% of its E2E) is not skipped by the
  draft. The skip-think `--traj-only` path collapses ADE to ~13 m and is not
  usable for accuracy tables.
- Verify is currently one teacher-force forward over the draft tokens; a
  token-level parallel verify is not implemented.

## Acceleration baselines and where the time actually goes

`survey/ACCEL_SURVEY.md` is the literature survey of acceleration methods for
AR driving VLAs (Spec-VLA, KERV, PD-VLA, FastDriveVLA, Prune2Drive, FastV,
DeeAD, Orion-Lite, ...), graded by publication status and by which pipeline
stage each one attacks. It also holds the single-GPU head-to-head numbers.

`survey/AD_NATIVE_ACCEL.md` narrows this to **autonomous-driving-native** work
only (six routes: visual-token pruning, CoT compression, non-autoregressive
decoding, KV reuse, distillation, serving), excluding robotics VLA methods
(Spec-VLA, KERV, PD-VLA, Zora, SwiftVLA, SQAP-VLA, TRIAGE) which appear there
only as cross-domain references.

`survey/SKIPLAYER_AND_PRUNING.md` is the deep dive on the two reduce-compute
families: dynamic depth (early exit vs layer skipping vs MoD vs Router-Tuning vs
static layer pruning) and token pruning (importance-based vs redundancy-based, and
why they criticise each other). Both are falsified on this teacher -- see the
"why everything went to zero" section, which ties back to decode being
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
`results/oracle_prune_opendrive.json`). So FastDriveVLA / Prune2Drive / FastV are
not merely orthogonal here -- they have no headroom at all on this teacher. Their
reported 3.7x-6.4x prefill wins come from 3249-token multi-camera ViTs, not from
148 UniAD query tokens feeding a 0.5B LM.

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
the fallback path is bit-exact: `n_accept == 0` frames now measure
`ADE(ours, AR) = 0.000000`.

Token-exact SD gives no speedup at all (1.00x). PD-VLA is unusable (29 m ADE,
0% convergence, 17.5% unparseable). Pruning gives exactly 1.00x -- its lower ADE
is the output drifting, not an improvement, since its latency is identical to AR.

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
