n=40, GT-valid n=34, OpenDriveVLA-0.5B + UniAD, one NVIDIA L20

| method | E2E ms | speedup | ADE | L2 1s | L2 2s | L2 3s |
|---|---|---|---|---|---|---|
| AR greedy | 1639 | 1.00x | 0.691 | 0.262 | 0.433 | 0.691 |
| Spec-VLA (exact) | 1700 | 0.96x | 0.691 | 0.262 | 0.433 | 0.691 |
| Spec-VLA (pose-relax) | 908 | 1.81x | 0.750 | 0.291 | 0.478 | 0.750 |
| PD-VLA (parallel) | 1289 | 1.27x | 29.235 | 1.887 | 20.637 | 29.235 |
| Pruning (oracle 25%) | 1639 | 1.00x | 0.659 | 0.254 | 0.414 | 0.659 |
| TrajSpecDec (ours) | 957 | 1.71x | 0.692 | 0.274 | 0.441 | 0.692 |
