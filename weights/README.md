# Draft head weights

Both heads are the small residual-MLP draft used by Trailblazer. They are
supervised on **teacher** trajectories (never ground truth) and were trained by
us with `scripts/run_full_specdec.py`.

| File | Teacher | Dataset | Horizon | Size | Val accept |
|---|---|---|---|---|---|
| `draft_opendrive_nusc.pt` | OpenDriveVLA-0.5B | nuScenes val (6019) | 6 pts / 3 s | 3.0 MB | 4.33 / 6 (4.11 on holdout) |
| `draft_waypoints_nusc.pt` | Qwen3-VL-4B waypoints (R&B-EnCoRe) | nuScenes val (6019) | 6 pts / 3 s | 7.1 MB | 4.59 / 6 (3.81 on holdout) |

## Load

```python
import torch
from trajspec.draft_head import DraftHead

blob = torch.load("weights/draft_waypoints_nusc.pt", map_location="cpu", weights_only=False)
head = DraftHead(
    blob["hidden_size"],
    horizon=blob.get("horizon", 6),
    extra_size=blob.get("extra_size", 16),
).cuda()
head.load_state_dict(blob["state_dict"])
head.eval()
```

`blob` carries `hidden_size` (896 for Qwen2.5-0.5B, 2560 for Qwen3-VL-4B) and
`extra_size` (16-d ego kinematics). Both state dicts live under `state_dict`.

## Notes

- `draft_opendrive_nusc.pt` is the checkpoint behind the OpenDrive live numbers
  (1.65x E2E, n=12) and the 4.33/6 full-val accept.
- `draft_waypoints_nusc.pt` is the R&B head behind the full-val live run
  (2.73x E2E, mean accept 4.14, n=6015).
- Teacher HF weights are **not** in this folder; use
  `scripts/download_teacher_weights.sh` (see `TRANSFER_OD_RNB.md`).
