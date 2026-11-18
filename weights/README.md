# Draft head weights

All three heads are the small residual-MLP draft used by Trailblazer. They are
supervised on **teacher** trajectories (never ground truth) and were trained by
us with `scripts/train_draft_navsim.py` / `scripts/run_full_specdec.py`.

| File | Teacher | Dataset | Horizon | Size | Val accept |
|---|---|---|---|---|---|
| `draft_opendrive_nusc.pt` | OpenDriveVLA-0.5B | nuScenes val (6019) | 6 pts / 3 s | 3.0 MB | 4.18 / 6 |
| `draft_autovla_navsim.pt` | AutoVLA-3B | NAVSIM navtest (12146) | 10 pts / 5 s | 5.9 MB | 3.63 / 10 (offline) |
| `draft_waypoints_nusc.pt` | Qwen3-VL-4B waypoints (RnB) | nuScenes val (6019) | 6 pts / 3 s | 7.1 MB | ~3.8 / 6 |

## Load

```python
import torch
from trajspec.draft_head import DraftHead

blob = torch.load("weights/draft_waypoints_nusc.pt", map_location="cpu", weights_only=False)
sd = blob.get("state_dict") or blob.get("head")
head = DraftHead(
    blob["hidden_size"],
    horizon=blob.get("horizon", 6),
    extra_size=blob.get("extra_size", 16),
).cuda()
head.load_state_dict(sd)
head.eval()
```

`blob` carries `hidden_size`; `horizon`/`extra_size` default to 6 / 16 when absent.
State dict is under `state_dict` for the nuScenes heads and under `head` for the
older AutoVLA NAVSIM checkpoint.

## Notes

- `draft_opendrive_nusc.pt` is the checkpoint behind the ~2.3× OpenDrive E2E number.
- `draft_waypoints_nusc.pt` is the RnB head used for the full-val live run (~2.7×).
- Teacher HF weights are **not** in this folder; use `scripts/download_teacher_weights.sh`
  (see `TRANSFER_OD_RNB.md`).
