#!/usr/bin/env python3
"""Merge holdout quality + live latency into one LaTeX table.

Absolute milliseconds are anchored to the empty-L20 TrajSpecDec live runs so
the numbers match the main latency table. Other methods are rescaled by the
decode-time ratio measured in the shared-propose baseline run.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFF = json.loads((ROOT / "results/sd_baselines_offline.json").read_text())
AV_LIVE = json.loads((ROOT / "results/live_sd_baselines_autovla.json").read_text())
OD_LIVE = json.loads((ROOT / "results/live_sd_baselines_opendrive.json").read_text())
AV_CAN = json.loads((ROOT / "results/live_specdec_autovla_nusc_trajonly.json").read_text())
OD_CAN = json.loads((ROOT / "results/live_specdec_opendrive.json").read_text())


def q(block, name):
    d = block[name]
    ade_g = d["spec_stp3"].get("ade")
    return d["mean_n_accept"], d["ade_vs_teacher"], ade_g


def scale_lat(live, can_prefill, can_ar_dec, can_ar_e2e, can_ours_dec):
    ours = live["methods"]["TrajSpecDec (ours)"]["decode_ms"]
    ratio = can_ours_dec / max(ours, 1e-6)
    out = {
        "AR greedy": {
            "decode_ms": can_ar_dec,
            "e2e_ms": can_ar_e2e,
            "spd": 1.0,
            "spd_dec": 1.0,
        }
    }
    for name, m in live["methods"].items():
        dec = m["decode_ms"] * ratio
        e2e = can_prefill + dec
        out[name] = {
            "decode_ms": dec,
            "e2e_ms": e2e,
            "spd": can_ar_e2e / max(e2e, 1e-6),
            "spd_dec": can_ar_dec / max(dec, 1e-6),
        }
    return out


AV_LAT = scale_lat(
    AV_LIVE,
    AV_CAN["latency"]["spec"]["prefill_ms"],
    AV_CAN["latency"]["teacher"]["decode_ms"],
    AV_CAN["latency"]["teacher"]["e2e_ms"],
    AV_CAN["latency"]["spec"]["decode_ms"],
)
OD_LAT = scale_lat(
    OD_LIVE,
    OD_CAN["measured_ms"]["prefill"]["mean"],
    OD_CAN["measured_ms"]["official_generate"]["mean"] - OD_CAN["measured_ms"]["prefill"]["mean"],
    OD_CAN["measured_ms"]["official_generate"]["mean"],
    OD_CAN["measured_ms"]["spec_e2e"]["mean"] - OD_CAN["measured_ms"]["prefill"]["mean"],
)


def fms(x):
    return f"{x:.0f}"


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def row(teacher, method, lat, hold, dash_acc=False, bold=False):
    acc, ade_t, ade_g = q(hold, method)
    L = lat[method]
    acc_s = "---" if dash_acc else f2(acc)
    name = method
    if bold:
        name = r"\textbf{" + method + "}"
        acc_s = r"\textbf{" + acc_s + "}"
        dec_s = r"\textbf{" + fms(L["decode_ms"]) + "}"
        e2e_s = r"\textbf{" + fms(L["e2e_ms"]) + "}"
        spd_s = r"\textbf{" + f2(L["spd"]) + r"$\times$}"
        at_s = r"\textbf{" + f3(ade_t) + "}"
        ag_s = r"\textbf{" + f3(ade_g) + "}"
    else:
        dec_s, e2e_s = fms(L["decode_ms"]), fms(L["e2e_ms"])
        spd_s = f2(L["spd"]) + r"$\times$"
        at_s, ag_s = f3(ade_t), f3(ade_g)
    return f"{teacher} & {name} & {acc_s} & {dec_s} & {e2e_s} & {spd_s} & {at_s} & {ag_s} \\\\"


av_rows = [
    ("AR greedy", True, False),
    ("Spec-VLA (exact)", False, False),
    ("Spec-VLA (id-relax)", False, False),
    ("Spec-VLA (pose-relax)", False, False),
    ("KERV-lite", False, False),
    ("TrajSpecDec (ours)", False, True),
]
od_rows = [
    ("AR greedy", True, False),
    ("Spec-VLA (exact)", False, False),
    ("Scaffold-SD", False, False),
    ("Spec-VLA (pose-relax)", False, False),
    ("KERV-lite", False, False),
    ("TrajSpecDec (ours)", False, True),
]

lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Speculative-decoding baselines on the same frozen teachers. "
    r"All SD variants share one one-shot draft and one teacher-force verify; only the accept rule and reject recovery change. "
    r"Acc.\ / ADE are on the 15\% hash holdout ($n{=}903$, seed 17). "
    r"Latency is CUDA-sync wall-clock on NVIDIA L20 ($n{=}12$, start index 1500), "
    r"with AR / TrajSpecDec milliseconds taken from the empty-GPU live protocol and other methods rescaled by the measured decode-time ratio. "
    r"Spd.\ is end-to-end vs AR greedy. "
    r"Spec-VLA (exact) is greedy token match; (id-relax) uses $|\Delta\mathrm{id}|\le 16$ on the AutoVLA codebook; "
    r"(pose-relax) is a metre-space adaptation with Euclidean radius $0.5$\,m. "
    r"Scaffold-SD auto-accepts non-numeric BPE tokens. "
    r"KERV-lite uses geometric accept then a bicycle tail \emph{without} AR re-decode. "
    r"ADE$_T$ / ADE$_G$: average L2 (m) vs teacher / GT.}",
    r"\label{tab:sd-baselines}",
    r"\small",
    r"\setlength{\tabcolsep}{3.0pt}",
    r"\begin{tabular}{llcccccc}",
    r"\toprule",
    r"Teacher & Method & Acc. & Dec.\ (ms) & E2E (ms) & Spd. & ADE$_T$ & ADE$_G$ \\",
    r"\midrule",
]
for i, (m, dash, bold) in enumerate(av_rows):
    t = "AutoVLA-3B" if i == 0 else ""
    lines.append(row(t, m, AV_LAT, OFF["autovla"], dash_acc=dash, bold=bold))
lines.append(r"\midrule")
for i, (m, dash, bold) in enumerate(od_rows):
    t = "OpenDriveVLA-0.5B" if i == 0 else ""
    lines.append(row(t, m, OD_LAT, OFF["opendrive"], dash_acc=dash, bold=bold))
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table*}",
    "",
]
tex = "\n".join(lines)
out = ROOT / "results/tab_sd_baselines.tex"
out.write_text(tex)
print(tex)
print("[write]", out)

summary = {"autovla_lat": AV_LAT, "opendrive_lat": OD_LAT, "holdout": OFF}
(ROOT / "results/sd_baselines_table_source.json").write_text(json.dumps(summary, indent=2, default=float))
