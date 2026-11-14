"""Aggregate <data>/results.jsonl into tables + figures (docs/results/).

    python tools/benchmark/report.py --data D:/bench_data [--out docs/results]
"""
import argparse
import json
import os
import textwrap

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# the six models looked at while developing the arm-anchored estimator; everything else is held out
DEV = {"quat_casual_male", "quat_suit_female", "kay_knight", "kay_mage", "cesium_man", "robot_expressive"}
JOINT_VARIANTS = [("geo_profile", "Width profile (geometry only)"),
                  ("geo_profile+kp2d", "+ 2D keypoints"),
                  ("geo_profile+kp2d+reconcile", "+ proportion reconcile (v1 system)"),
                  ("geo_arm", "Arm-anchored geometry only"),
                  ("full", "Arm-anchored + 2D + arbitration (ours)")]
SKIN_VARIANTS = [("full", "heat, auto fallback + post-processing (ours)"),
                 ("full_proxyheat", "voxel-proxy heat"),
                 ("full_nopost", "no post-processing")]
GROUP = {"stylised-realistic": "realistic", "realistic-lowpoly": "realistic",
         "chibi": "stylised", "cartoon-robot": "stylised"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def load(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return [r for r in rows if r.get("ok")]


def fmt(xs, pct=True, k=1):
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        return "-"
    m = xs.mean() * (100 if pct else 1)
    return f"{m:.{k}f}"


def table(rows, variants, metric_cols, split):
    sel = [r for r in rows if (r["case"] in DEV) == (split == "dev")] if split != "all" else rows
    L = ["| Variant | " + " | ".join(c[0] for c in metric_cols) + " |",
         "|---|" + "---|" * len(metric_cols)]
    for v, name in variants:
        rv = [r for r in sel if r["variant"] == v]
        cells = []
        for _, fn in metric_cols:
            cells.append(fn(rv))
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def by_group(g, key, scale=100, k=1):
    def f(rv):
        xs = [r[key] for r in rv if GROUP[r["style"]] == g] if g else [r[key] for r in rv]
        xs = [x for x in xs if x == x]  # runs that produced no weights are counted separately
        return f"{np.mean(xs) * scale:.{k}f}" if xs else "-"
    return f


def unweighted(rv):
    n = sum(any("without weights" in e for e in r.get("validator_errors", [])) for r in rv)
    return f"{n} / {len(rv)}"


def frac_below(g, thr):
    def f(rv):
        xs = [r["mpjpe"] < thr for r in rv if (GROUP[r["style"]] == g or g is None)]
        return f"{100 * np.mean(xs):.0f}%" if xs else "-"
    return f


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_ablation(rows, out):
    test = [r for r in rows if r["case"] not in DEV]
    groups = [("realistic", "Realistic proportions"), ("stylised", "Stylised / chibi")]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    n = len(JOINT_VARIANTS)
    w = 0.36
    x = np.arange(n)
    for gi, (g, gname) in enumerate(groups):
        vals = []
        for v, _ in JOINT_VARIANTS:
            xs = [r["mpjpe"] * 100 for r in test if r["variant"] == v and GROUP[r["style"]] == g]
            vals.append(np.mean(xs) if xs else 0)
        bars = ax.bar(x + (gi - 0.5) * (w + 0.02), vals, w, color=SERIES[gi], label=gname,
                      edgecolor="white", linewidth=1)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.2, f"{val:.1f}", ha="center", va="bottom",
                    fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(name, 20) for _, name in JOINT_VARIANTS], fontsize=8, color=INK)
    ax.set_ylabel("Mean joint error (% of height)", color=MUTED, fontsize=9)
    ax.set_title("Joint placement ablation - held-out models, rest pose + A-pose", fontsize=10, color=INK,
                 loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_per_joint(rows, out):
    test = [r for r in rows if r["case"] not in DEV]
    joints = ["head", "neck_01", "upperarm", "lowerarm", "hand", "thigh", "calf", "foot"]
    names = ["head", "neck", "shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (g, gname) in zip(axes, [("realistic", "Realistic"), ("stylised", "Stylised / chibi")]):
        for vi, (v, lab) in enumerate([("geo_profile+kp2d+reconcile", "v1 system"), ("full", "ours")]):
            vals = []
            for j in joints:
                xs = []
                for r in test:
                    if r["variant"] != v or GROUP[r["style"]] != g:
                        continue
                    for k, e in r["joint_err"].items():
                        if k == j or k in (f"{j}_l", f"{j}_r"):
                            xs.append(e * 100)
                vals.append(np.mean(xs) if xs else np.nan)
            ax.barh(np.arange(len(joints)) + (vi - 0.5) * 0.38, vals, 0.36, color=SERIES[vi], label=lab,
                    edgecolor="white", linewidth=1)
        ax.set_yticks(np.arange(len(joints)))
        ax.set_yticklabels(names, fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_title(gname, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("error (% of height)", fontsize=9, color=MUTED)
        style_axes(ax)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.yaxis.grid(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_qualitative(rows, data, out, k=4):
    """Held-out runs where v1 and ours differ most (both directions are shown honestly)."""
    import viz
    v1 = {(r["case"], r["pose"]): r for r in rows if r["variant"] == "geo_profile+kp2d+reconcile"}
    ours = {(r["case"], r["pose"]): r for r in rows if r["variant"] == "full"}
    diffs = sorted(((v1[key]["mpjpe"] - ours[key]["mpjpe"], key) for key in v1 if key in ours
                    and key[0] not in DEV), reverse=True)
    picks = [d[1] for d in diffs[: k - 1]] + [diffs[-1][1]]  # three biggest gains + the worst regression
    fig, axes = plt.subplots(2, len(picks), figsize=(3.0 * len(picks), 7.4))
    for j, (case, pose) in enumerate(picks):
        key = f"{case}__{pose}"
        for i, (v, lab) in enumerate((("geo_profile+kp2d+reconcile", "v1"), ("full", "ours"))):
            r = v1 if i == 0 else ours
            title = f"{case} ({pose})\n{lab}: {r[(case, pose)]['mpjpe'] * 100:.1f} %H"
            viz.panel(axes[i, j], data, key, v, title)
    fig.legend(handles=[plt.Line2D([], [], color="#2a7f62", marker="o", label="artist rig (ground truth)"),
                        plt.Line2D([], [], color="#d1495b", marker="x", label="predicted")],
               loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=120)
    plt.close(fig)


def detection(rows, variant="full", thr=0.08):
    rv = [r for r in rows if r["variant"] == variant]
    bad = [r["mpjpe"] > thr for r in rv]
    flag = [bool(r.get("validator_errors")) for r in rv]
    tp = sum(b and f for b, f in zip(bad, flag))
    fp = sum((not b) and f for b, f in zip(bad, flag))
    fn = sum(b and (not f) for b, f in zip(bad, flag))
    return {"n": len(rv), "failures": sum(bad), "flagged": sum(flag), "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "docs", "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(os.path.join(a.data, "results.jsonl"))
    cases = sorted({r["case"] for r in rows})
    jcols = [("Realistic MPJPE %H", by_group("realistic", "mpjpe")),
             ("Stylised MPJPE %H", by_group("stylised", "mpjpe")),
             ("All MPJPE %H", by_group(None, "mpjpe")),
             ("PCK@5%H", by_group(None, "pck05", 100, 0)),
             ("Runs < 8%H", frac_below(None, 0.08))]
    scols = [("Realistic part acc %", by_group("realistic", "part_accuracy")),
             ("Stylised part acc %", by_group("stylised", "part_accuracy")),
             ("All part acc %", by_group(None, "part_accuracy")),
             ("All part mIoU %", by_group(None, "part_miou")),
             ("Runs left with unweighted vertices", unweighted)]
    md = ["# Benchmark results (auto-generated by tools/benchmark/report.py)", "",
          f"{len(cases)} models ({len([c for c in cases if c not in DEV])} held-out), "
          f"{len(rows)} successful runs. MPJPE = mean joint position error over 14 joints "
          "(neck, head, shoulders, elbows, wrists, hips, knees, ankles), in % of character height. "
          "Part accuracy = share of vertices whose dominant bone lies in the same body part "
          "(14 parts) as in the artist's rig.", "",
          "## Joint placement - held-out models", "", table(rows, JOINT_VARIANTS, jcols, "test"), "",
          "## Joint placement - development models", "", table(rows, JOINT_VARIANTS, jcols, "dev"), "",
          "## Skinning (joints from the full method) - held-out models", "",
          table(rows, SKIN_VARIANTS, scols, "test"), "",
          "## Skinning - all models", "", table(rows, SKIN_VARIANTS, scols, "all"), ""]
    det = detection(rows)
    md += ["## Self-check: do the validators catch bad rigs?", "",
           f"Full method, all runs (n={det['n']}). A run counts as a failure when MPJPE > 8 % H.", "",
           "| Failures | Flagged by validators | True positives | False positives | Missed |",
           "|---|---|---|---|---|",
           f"| {det['failures']} | {det['flagged']} | {det['tp']} | {det['fp']} | {det['fn']} |", ""]
    md += ["## Per model (full method)", "", "| Model | Pose | Style | MPJPE %H | Part acc % | 2D keypoints |",
           "|---|---|---|---|---|---|"]
    for r in sorted([r for r in rows if r["variant"] == "full"], key=lambda r: (r["style"], r["case"], r["pose"])):
        md.append(f"| {r['case']}{' (dev)' if r['case'] in DEV else ''} | {r['pose']} | {r['style']} | "
                  f"{r['mpjpe'] * 100:.1f} | {r['part_accuracy'] * 100:.1f} | "
                  f"{'yes' if r['pose2d_ok'] else 'no person found'} |")
    with open(os.path.join(a.out, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    fig_ablation(rows, os.path.join(a.out, "joint_ablation.png"))
    fig_per_joint(rows, os.path.join(a.out, "per_joint.png"))
    fig_qualitative(rows, a.data, os.path.join(a.out, "qualitative.png"))
    json.dump({"detection": det}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
