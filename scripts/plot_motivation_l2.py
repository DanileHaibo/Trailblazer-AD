#!/usr/bin/env python3
"""Motivation Fig.1: mean L2 vs GT for VLA / ego kinematics / ordinary MLP."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "motivation_figs"

T = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
RNB = {
    "VLA": [0.1644, 0.3448, 0.5942, 0.9407, 1.3924, 1.9737],
    "Ego kinematics only": [0.1544, 0.412, 0.837, 1.462, 2.297, 3.3475],
    "Ordinary MLP": [0.4661, 0.8285, 1.1692, 1.5605, 2.0428, 2.5977],
}
OD = {
    "VLA": [0.4666, 0.9467, 1.4765, 2.0616, 2.7102, 3.4258],
    "Ego kinematics only": [0.496, 1.0883, 1.822, 2.7179, 3.7806, 5.0305],
    "Ordinary MLP": [0.773, 1.4204, 2.1352, 2.8825, 3.6398, 4.4277],
}
COLORS = {"VLA": "#2B6CB0", "Ego kinematics only": "#4A5568", "Ordinary MLP": "#2F855A"}
MARKERS = {"VLA": "o", "Ego kinematics only": "s", "Ordinary MLP": "D"}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "legend.fontsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw(ax, series, title, ymax):
    for name, ys in series.items():
        ax.plot(
            T,
            ys,
            color=COLORS[name],
            marker=MARKERS[name],
            markersize=9,
            linewidth=2.8,
            label=name,
        )
    ax.set_title(title, pad=10, fontweight="semibold")
    ax.set_xlabel("Waypoint time (s)")
    ax.set_ylabel("Mean L2 (m)")
    ax.set_xticks(T)
    ax.set_xlim(0.35, 3.15)
    ax.set_ylim(0, ymax)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8, borderaxespad=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.1, length=5)


def main():
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("motivation_l2_opendrive", OD, "OpenDriveVLA-0.5B, 1000 val frames, mean L2 vs GT", 5.6),
        ("motivation_l2_rnb", RNB, "R&B-EnCoRe-4B, 1000 val frames, mean L2 vs GT", 3.6),
    ]
    for stem, series, title, ymax in jobs:
        fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=150)
        draw(ax, series, title, ymax)
        fig.tight_layout()
        fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
        fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        print(OUT / f"{stem}.png")
        print(OUT / f"{stem}.pdf")


if __name__ == "__main__":
    main()
