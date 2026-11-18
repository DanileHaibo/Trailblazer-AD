#!/usr/bin/env python3
"""Render results/unified_accel_table.json as a LaTeX table + a markdown twin."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/unified_accel_table.json"
ORDER = [
    "AR greedy",
    "Spec-VLA (exact)",
    "Spec-VLA (pose-relax)",
    "PD-VLA (parallel)",
    "Pruning (oracle 25%)",
    "TrajSpecDec (ours)",
]
SHORT = {
    "AR greedy": "AR greedy (official \\texttt{generate()})",
    "Spec-VLA (exact)": "Spec-VLA (exact)",
    "Spec-VLA (pose-relax)": "Spec-VLA (pose-relax)",
    "PD-VLA (parallel)": "PD-VLA (parallel decoding)",
    "Pruning (oracle 25%)": "Pruning (oracle, 25\\% kept)",
    "TrajSpecDec (ours)": "\\textbf{TrajSpecDec (ours)}",
}


def main():
    d = json.loads(SRC.read_text())
    m = d["methods"]
    n = d["n"]
    ng = m["AR greedy"]["n_with_gt"]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Four acceleration families vs TrajSpecDec on \textbf{the same frames} "
        rf"($n{{=}}{n}$, GT-valid $n{{=}}{ng}$), OpenDriveVLA-0.5B + UniAD, one NVIDIA L20. "
        r"The UniAD tower runs \emph{once per frame} and its features are shared by every row, "
        r"so only the decode strategy differs; E2E includes that shared tower cost. "
        r"Accuracy is vs VAD ground truth in metres (lower is better). "
        r"Each family attacks a different stage: Spec-VLA/KERV the accept rule, PD-VLA the "
        r"serial dependency, pruning the visual tokens.}",
        r"\label{tab:unified-accel}",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & E2E (ms) $\downarrow$ & Speedup $\uparrow$ & ADE $\downarrow$ & "
        r"L2 1s $\downarrow$ & L2 2s $\downarrow$ & L2 3s $\downarrow$ \\",
        r"\midrule",
    ]
    for key in ORDER:
        v = m[key]
        name = SHORT[key]
        lines.append(
            f"{name} & {v['e2e_ms']:.0f} & {v['speedup_e2e']:.2f}$\\times$ & "
            f"{v.get('ade_gt', 0):.3f} & {v.get('l2_1s', 0):.3f} & "
            f"{v.get('l2_2s', 0):.3f} & {v.get('l2_3s', 0):.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    tex = "\n".join(lines)
    (ROOT / "results" / "tab_unified_accel.tex").write_text(tex)

    # markdown twin
    md = [
        f"n={n}, GT-valid n={ng}, OpenDriveVLA-0.5B + UniAD, one NVIDIA L20",
        "",
        "| method | E2E ms | speedup | ADE | L2 1s | L2 2s | L2 3s |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in ORDER:
        v = m[key]
        md.append(
            f"| {key} | {v['e2e_ms']:.0f} | {v['speedup_e2e']:.2f}x | "
            f"{v.get('ade_gt',0):.3f} | {v.get('l2_1s',0):.3f} | "
            f"{v.get('l2_2s',0):.3f} | {v.get('l2_3s',0):.3f} |"
        )
    (ROOT / "results" / "tab_unified_accel.md").write_text("\n".join(md) + "\n")
    print(tex)
    print("[write] results/tab_unified_accel.tex + .md")


if __name__ == "__main__":
    main()
