n=24 (GT-valid 20), OpenDriveVLA-0.5B+UniAD, one NVIDIA L20

| method | stage attacked | E2E ms | speedup | ADE | L2@1s | L2@2s | L2@3s |
|---|---|---|---|---|---|---|---|
| AR greedy | - | 1635 | 1.00x | 0.727 | 0.334 | 0.490 | 0.727 |
| FastV (K=2,R=50%) | visual tokens | 1692 | 0.97x | 0.647 | 0.312 | 0.433 | 0.647 |
| DeeAD (depth 22/24) | layer depth | 1834 | 0.89x | 3.253 | 1.655 | 2.401 | 3.253 |
| Scaffold accept | accept rule | 1585 | 1.03x | 2.381 | 0.897 | 2.217 | 2.381 |
| Pruning oracle (25%) | visual tokens | 1625 | 1.01x | 0.627 | 0.312 | 0.426 | 0.627 |
| TrajSpecDec (ours) | accept+seam | 1080 | 1.51x | 0.815 | 0.346 | 0.515 | 0.815 |
