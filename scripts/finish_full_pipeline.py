#!/usr/bin/env python3
"""Wait for teacher caches, train drafts, eval, write the combined final JSON.

Keeps iterating AutoVLA if held-out accept is below target.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/xinhchen2/hhb/TrajSpecDec")
PY = "/data2/conda_env/qwendrive/bin/python"
RESULTS = ROOT / "results"

AV_TRAIN = RESULTS / "cache_autovla_nusc_train.pt"
AV_VAL = RESULTS / "cache_autovla_nusc_val.pt"
AV_VAL_LOG = RESULTS / "cache_autovla_val.log"
OD_CACHE = RESULTS / "cache_opendrive_nusc_val.pt"
OD_LOG = RESULTS / "cache_opendrive.log"

AV_OUT = RESULTS / "specdec_autovla_heldout.json"
OD_OUT = RESULTS / "specdec_opendrive_heldout.json"
FINAL = RESULTS / "specdec_full_final.json"


def log(msg: str) -> None:
    print(f"[finish] {msg}", flush=True)


def cache_n(path: Path) -> int:
    return 0  # never torch.load an in-progress cache from the waiter


def log_has_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        txt = path.read_text(errors="ignore")[-8000:]
    except Exception:
        return False
    return "[done]" in txt


def wait_until(desc: str, pred, timeout_s: int = 8 * 3600, every: int = 45) -> None:
    t0 = time.time()
    log(f"waiting for {desc}")
    while True:
        if pred():
            log(f"{desc} ready after {time.time()-t0:.0f}s")
            return
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"timeout waiting for {desc}")
        time.sleep(every)


def run_specdec(name: str, train: Path, out: Path, val: str = "", target: float = 3.8, rounds: int = 8, steps: int = 2500) -> dict:
    cmd = [
        PY, "-u", str(ROOT / "scripts/run_full_specdec.py"),
        "--name", name,
        "--train-cache", str(train),
        "--target-accept", str(target),
        "--max-rounds", str(rounds),
        "--steps-per", str(steps),
        "--out", str(out),
    ]
    if val:
        cmd += ["--val-cache", val]
    log("exec " + " ".join(cmd))
    env = dict(**{k: v for k, v in __import__("os").environ.items() if k != "CUDA_VISIBLE_DEVICES"})
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        raise SystemExit(f"specdec failed {name} code={r.returncode}")
    return json.loads(out.read_text())["summary"]


def main():
    wait_until(
        "AutoVLA val cache",
        lambda: log_has_done(AV_VAL_LOG) or cache_n(AV_VAL) >= 1900,
    )
    av = run_specdec("autovla_nusc", AV_TRAIN, AV_OUT, val=str(AV_VAL), target=3.8, rounds=8, steps=2500)
    log(f"AutoVLA held-out accept={av.get('mean_n_accept')} n={av.get('n')}")
    if float(av.get("mean_n_accept", 0)) < 3.2:
        log("accept still low, extra AutoVLA rounds")
        av = run_specdec("autovla_nusc", AV_TRAIN, AV_OUT, val=str(AV_VAL), target=4.0, rounds=12, steps=3500)

    wait_until(
        "OpenDrive hidden cache",
        lambda: log_has_done(OD_LOG) or cache_n(OD_CACHE) >= 5900,
    )
    # Snapshot so a late writer cannot race the trainer.
    snap = Path("/tmp/od_full_cache.pt")
    import shutil

    shutil.copy2(OD_CACHE, snap)
    od = run_specdec("opendrive_nusc", snap, OD_OUT, val="", target=3.5, rounds=8, steps=2500)
    log(f"OpenDrive holdout accept={od.get('mean_n_accept')} n={od.get('n')}")
    if float(od.get("mean_n_accept", 0)) < 2.8:
        log("OpenDrive accept low, extra rounds")
        od = run_specdec("opendrive_nusc", snap, OD_OUT, val="", target=3.5, rounds=12, steps=3500)

    final = {
        "accept_thresholds": {"lon_m": 0.5, "lat_m": 0.3},
        "note": "mean_decode_speedup is decode-step accounting (1 parallel verify), not wall-clock. Draft is supervised on teacher traj. Spec traj replaces the first reject with the teacher waypoint then copies the teacher suffix.",
        "autovla": av,
        "opendrive": od,
    }
    FINAL.write_text(json.dumps(final, ensure_ascii=False, indent=2))
    log(f"wrote {FINAL}")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
