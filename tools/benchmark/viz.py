"""Front-view plot of ground truth vs predicted joints for benchmark runs.

    python tools/benchmark/viz.py --data D:/bench_data --runs kay_knight__rest:geometric,kay_knight__rest:full --out fig.png
"""
import argparse
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_benchmark import JOINTS, to_norm  # noqa: E402

BONES = [("neck_01", "head"), ("upperarm_l", "lowerarm_l"), ("lowerarm_l", "hand_l"),
         ("upperarm_r", "lowerarm_r"), ("lowerarm_r", "hand_r"), ("thigh_l", "calf_l"),
         ("calf_l", "foot_l"), ("thigh_r", "calf_r"), ("calf_r", "foot_r"),
         ("upperarm_l", "upperarm_r"), ("thigh_l", "thigh_r")]


def panel(ax, data, key, variant, title=None):
    case = os.path.join(data, "cases", key)
    run = os.path.join(data, "runs", variant, key)
    gt = json.load(open(os.path.join(case, "gt.json")))
    norm = json.load(open(os.path.join(run, "norm.json")))
    rep = json.load(open(os.path.join(run, "05_rig", "report.json")))
    P = to_norm(np.load(os.path.join(case, "gt_points.npy")), norm)
    ax.scatter(P[:, 0], P[:, 2], s=0.3, c="#c8c8c8")
    G = {j: to_norm([gt["joints"][j]], norm)[0] for j in JOINTS if gt["joints"].get(j)}
    Q = {j: np.asarray(v) for j, v in rep["joints"].items()}
    for S, col, mk in ((G, "#2a7f62", "o"), (Q, "#d1495b", "x")):
        for a, b in BONES:
            if a in S and b in S:
                ax.plot([S[a][0], S[b][0]], [S[a][2], S[b][2]], color=col, lw=1.4)
        xs = np.array([S[j] for j in JOINTS if j in S])
        ax.scatter(xs[:, 0], xs[:, 2], c=col, s=16, marker=mk, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"{key}\n{variant}", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True, help="comma list of case__pose:variant[:title]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [r.split(":") for r in a.runs.split(",")]
    fig, axes = plt.subplots(1, len(items), figsize=(3.2 * len(items), 4.2))
    for ax, it in zip(np.atleast_1d(axes), items):
        panel(ax, a.data, it[0], it[1], it[2] if len(it) > 2 else None)
    fig.legend(handles=[plt.Line2D([], [], color="#2a7f62", marker="o", label="artist rig (ground truth)"),
                        plt.Line2D([], [], color="#d1495b", marker="x", label="predicted")],
               loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == "__main__":
    main()
