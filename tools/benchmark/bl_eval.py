"""
Blender: skinning accuracy of a rigged result against the artist rig.

    blender -b --factory-startup -P bl_eval.py -- --rigged x_rigged.blend --case case/ --norm norm.json --out m.json

Every vertex of our mesh gets the body part of its dominant deform bone; the ground-truth
part is the one of the nearest artist-rig vertex (after mapping the ground truth into the
pipeline's normalised space). Reports overall accuracy and per-part IoU.
"""
import argparse
import json
import math
import os
import sys

import bpy
import mathutils
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bl_prepare import PARTS  # noqa: E402


def our_part(bone):
    if bone.startswith(("neck_", "head")):
        return "head"
    for side in ("l", "r"):
        if bone in (f"upperarm_{side}",):
            return f"upperarm_{side}"
        if bone in (f"lowerarm_{side}",):
            return f"lowerarm_{side}"
        if bone == f"hand_{side}" or (bone.endswith(f"_{side}") and bone.split("_")[0] in
                                      ("thumb", "index", "middle", "ring", "pinky")):
            return f"hand_{side}"
        if bone == f"thigh_{side}":
            return f"thigh_{side}"
        if bone == f"calf_{side}":
            return f"calf_{side}"
        if bone in (f"foot_{side}", f"ball_{side}"):
            return f"foot_{side}"
    return "torso"


def to_norm(P, norm):
    P = np.asarray(P, dtype=np.float64)
    e = norm.get("stand_up_euler_deg")
    if e:
        R = mathutils.Euler([math.radians(x) for x in e], "XYZ").to_matrix()
        P = P @ np.array(R).T
    rz = math.radians(norm.get("rot_z_deg") or 0)
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return np.asarray(norm["offset"]) + norm["scale"] * (P @ Rz.T)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--rigged", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--norm", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    norm = json.load(open(a.norm))
    gtP = to_norm(np.load(os.path.join(a.case, "gt_points.npy")), norm)
    gtL = np.load(os.path.join(a.case, "gt_labels.npy"))
    keep = gtL >= 0
    gtP, gtL = gtP[keep], gtL[keep]
    kd = mathutils.kdtree.KDTree(len(gtP))
    for i, q in enumerate(gtP):
        kd.insert(q, i)
    kd.balance()

    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.rigged))
    mesh = next(o for o in bpy.data.objects if o.type == "MESH" and o.vertex_groups)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    deform = {b.name for b in arm.data.bones if b.use_deform}
    gname = {g.index: g.name for g in mesh.vertex_groups}
    pred, truth, dist = [], [], []
    mw = mesh.matrix_world
    for v in mesh.data.vertices:
        best = max(((g.weight, gname[g.group]) for g in v.groups if gname.get(g.group) in deform),
                   default=(0, None))
        if best[1] is None:
            continue
        co = mw @ v.co
        _, idx, d = kd.find(co)
        pred.append(PARTS.index(our_part(best[1])))
        truth.append(int(gtL[idx]))
        dist.append(d)
    pred, truth = np.array(pred), np.array(truth)
    ious = {}
    for k, name in enumerate(PARTS):
        inter = np.sum((pred == k) & (truth == k))
        union = np.sum((pred == k) | (truth == k))
        if union:
            ious[name] = float(inter / union)
    out = {"part_accuracy": float((pred == truth).mean()), "part_miou": float(np.mean(list(ious.values()))),
           "part_iou": ious, "n_vertices": int(len(pred)),
           "median_nn_dist_m": float(np.median(dist)) if dist else None}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("eval", out["part_accuracy"], out["part_miou"])


if __name__ == "__main__":
    main()
