# Draft head weights

All three heads are the small residual-MLP draft used by Trailblazer. They are
supervised on **teacher** trajectories (never ground truth) and were trained by
us with `scripts/train_draft_navsim.py` / `scripts/run_full_specdec.py`.

| File | Teacher | Dataset | Horizon | Size | Val accept |
|---|---|---|---|---|---|
| `draft_opendrive_nusc.pt` | OpenDriveVLA-0.5B | nuScenes val (6019) | 6 pts / 3 s | 3.0 MB | 4.18 / 6 |
| `draft_autovla_navsim.pt` | AutoVLA-3B | NAVSIM navtest (12146) | 10 pts / 5 s | 5.9 MB | 3.63 / 10 (offline) |
| `draft_waypoints_nusc.pt` | Qwen3-VL-4B waypoints | nuScenes val (600) | 6 pts / 3 s | 7.1 MB | 2.69 / 6 |

## Load

```python
import torch
from trajspec.draft_head import DraftHead

blob = torch.load("weights/draft_autovla_navsim.pt", map_location="cpu", weights_only=False)
sd = blob.get("state_dict") or blob.get("head")
head = DraftHead(
    blob["hidden_size"],
    horizon=blob.get("horizon", 6),      # only the AutoVLA head stores this
    extra_size=blob.get("extra_size", 16),
).cuda()
head.load_state_dict(sd)
head.eval()
```

`blob` carries `hidden_size`; `horizon`/`extra_size` are stored explicitly only
in the AutoVLA head, so fall back to 6 / 16 for the other two (6-waypoint
nuScenes horizon, 16-d ego extras). The state dict lives under `state_dict` for
the two nuScenes heads and under `head` for the NAVSIM / waypoints ones.

## Notes

- `draft_opendrive_nusc.pt` is the checkpoint the 2.29x end-to-end acceleration
  number was measured with.
- `draft_autovla_navsim.pt` is the checkpoint behind the 1.15x AutoVLA number.
  An earlier head (`specdec_autovla_navsim.pt`, accept 2.92) is superseded.
- `draft_waypoints_nusc.pt` was trained on only 600 cached frames when the
  third teacher was still being evaluated; it is included for completeness and
  should be retrained on a larger cache before citing numbers.
