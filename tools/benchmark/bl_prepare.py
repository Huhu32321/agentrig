"""
Blender: turn an artist-rigged humanoid into a benchmark case.

    blender -b --factory-startup -P bl_prepare.py -- --in raw.glb --outdir case/ [--apose 45]

Writes
    mesh.glb        the posed, unrigged, single mesh (what an image-to-3D model would give us)
    gt.json         ground-truth joint positions (Blender world space, metres) + metadata
    gt_points.npy   posed vertex positions (N x 3)
    gt_labels.npy   body-part id per vertex (-1 = unknown), see PARTS
"""
import argparse
import json
import math
import os
import sys

import bpy
import mathutils
import numpy as np

PARTS = ["torso", "head", "upperarm_l", "lowerarm_l", "hand_l", "upperarm_r", "lowerarm_r", "hand_r",
         "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r"]
JOINTS = ["neck_01", "head", "upperarm_l", "lowerarm_l", "hand_l", "upperarm_r", "lowerarm_r", "hand_r",
          "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r"]


def _sided(table, l_tok, r_tok):
    out = {}
    for k, v in table.items():
        if k.endswith("_s"):
            out[k[:-2] + "_l"] = [x.replace("{s}", l_tok) for x in v] if isinstance(v, list) else v.replace("{s}", l_tok)
            out[k[:-2] + "_r"] = [x.replace("{s}", r_tok) for x in v] if isinstance(v, list) else v.replace("{s}", r_tok)
        else:
            out[k] = v
    return out


# joint -> "Bone" (bone head) or "tail:Bone";  part -> list of seed bones
SPECS = {
    "quaternius": {
        "detect": "UpperArm.L",
        "joints": _sided({"neck_01": "Neck", "head": "Head", "upperarm_s": "UpperArm.{s}",
                          "lowerarm_s": "LowerArm.{s}", "hand_s": "Fist.{s}", "thigh_s": "UpperLeg.{s}",
                          "calf_s": "LowerLeg.{s}", "foot_s": "tail:LowerLeg.{s}"}, "L", "R"),
        "parts": _sided({"head": ["Neck"], "upperarm_s": ["UpperArm.{s}"], "lowerarm_s": ["LowerArm.{s}"],
                         "hand_s": ["Fist.{s}", "Palm1.{s}", "Palm2.{s}", "Palm3.{s}", "Thumb.{s}"],
                         "thigh_s": ["UpperLeg.{s}"], "calf_s": ["LowerLeg.{s}"], "foot_s": ["Foot.{s}"]},
                        "L", "R"),
        "upperarm": "UpperArm.{s}",
    },
    "kaykit": {
        "detect": "upperarm.l",
        "joints": _sided({"head": "head", "upperarm_s": "upperarm.{s}", "lowerarm_s": "lowerarm.{s}",
                          "hand_s": "wrist.{s}", "thigh_s": "upperleg.{s}", "calf_s": "lowerleg.{s}",
                          "foot_s": "foot.{s}"}, "l", "r"),
        "parts": _sided({"head": ["head"], "upperarm_s": ["upperarm.{s}"], "lowerarm_s": ["lowerarm.{s}"],
                         "hand_s": ["wrist.{s}"], "thigh_s": ["upperleg.{s}"], "calf_s": ["lowerleg.{s}"],
                         "foot_s": ["foot.{s}"]}, "l", "r"),
        "upperarm": "upperarm.{s}",
    },
    "cesium": {
        "detect": "Skeleton_arm_joint_L__4_",
        "joints": {"neck_01": "Skeleton_neck_joint_1", "head": "Skeleton_neck_joint_2",
                   "upperarm_l": "Skeleton_arm_joint_L__4_", "lowerarm_l": "Skeleton_arm_joint_L__3_",
                   "hand_l": "Skeleton_arm_joint_L__2_", "upperarm_r": "Skeleton_arm_joint_R",
                   "lowerarm_r": "Skeleton_arm_joint_R__2_", "hand_r": "Skeleton_arm_joint_R__3_",
                   "thigh_l": "leg_joint_L_1", "calf_l": "leg_joint_L_2", "foot_l": "leg_joint_L_3",
                   "thigh_r": "leg_joint_R_1", "calf_r": "leg_joint_R_2", "foot_r": "leg_joint_R_3"},
        "parts": {"head": ["Skeleton_neck_joint_1"],
                  "upperarm_l": ["Skeleton_arm_joint_L__4_"], "lowerarm_l": ["Skeleton_arm_joint_L__3_"],
                  "hand_l": ["Skeleton_arm_joint_L__2_"], "upperarm_r": ["Skeleton_arm_joint_R"],
                  "lowerarm_r": ["Skeleton_arm_joint_R__2_"], "hand_r": ["Skeleton_arm_joint_R__3_"],
                  "thigh_l": ["leg_joint_L_1"], "calf_l": ["leg_joint_L_2"], "foot_l": ["leg_joint_L_3"],
                  "thigh_r": ["leg_joint_R_1"], "calf_r": ["leg_joint_R_2"], "foot_r": ["leg_joint_R_3"]},
        "upperarm": None,  # already arms-down
    },
}


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--apose", type=float, default=0.0, help="lower T-pose arms by this many degrees")
    return p.parse_args(argv)


def bone_part(bone, seeds):
    b = bone
    while b is not None:
        if b.name in seeds:
            return seeds[b.name]
        b = b.parent
    return "torso"


def main():
    a = parse()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(a.inp))
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    names = set(arm.data.bones.keys())
    fam = next(k for k, s in SPECS.items() if s["detect"] in names)
    spec = SPECS[fam]
    if fam == "quaternius" and "Fist.L" not in names:
        spec = dict(spec, joints=dict(spec["joints"], hand_l="Palm2.L", hand_r="Palm2.R"))

    # rest pose, no animation
    arm.animation_data_clear()
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
    for o in bpy.data.objects:
        if o.type == "MESH" and o.data.shape_keys:
            for kb in o.data.shape_keys.key_blocks:
                kb.value = 0.0
    bpy.context.view_layer.update()

    posed = False
    if a.apose and spec["upperarm"]:
        for s, sign in (("L" if fam != "kaykit" else "l", 1), ("R" if fam != "kaykit" else "r", -1)):
            pb = arm.pose.bones[spec["upperarm"].format(s=s)]
            d = (arm.matrix_world.to_3x3() @ (pb.bone.tail_local - pb.bone.head_local)).normalized()
            if abs(d.z) > 0.5:  # already pointing down
                continue
            M = arm.matrix_world.to_3x3() @ pb.bone.matrix_local.to_3x3()
            R = mathutils.Matrix.Rotation(math.radians(sign * a.apose), 3, "Y")
            pb.rotation_quaternion = (M.inverted() @ R @ M).to_quaternion()
            posed = True
        bpy.context.view_layer.update()

    def jpos(spec_str):
        if spec_str is None:
            return None
        tail = spec_str.startswith("tail:")
        pb = arm.pose.bones.get(spec_str[5:] if tail else spec_str)
        if pb is None:
            return None
        v = arm.matrix_world @ (pb.tail if tail else pb.head)
        return [round(c, 5) for c in v]

    joints = {j: jpos(spec["joints"].get(j)) for j in JOINTS}

    seeds = {b: part for part, bl in spec["parts"].items() for b in bl}
    part_of_bone = {b.name: bone_part(b, seeds) for b in arm.data.bones}
    dg = bpy.context.evaluated_depsgraph_get()
    pts, labels, objs = [], [], []
    def real_mesh(o):
        if o.type != "MESH":
            return False
        # the importer's hidden bone-shape helper
        if any(c.name.startswith("glTF_not_exported") for c in o.users_collection):
            return False
        # props held in the hands (KayKit swords / shields / staffs) are not part of the character
        return not (o.parent_type == "BONE" and "handslot" in (o.parent_bone or "").lower())

    for o in [o for o in bpy.data.objects if real_mesh(o)]:
        ev = o.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(ev)
        me.transform(o.matrix_world)
        n = len(me.vertices)
        if n == 0:
            continue
        P = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", P)
        pts.append(P.reshape(-1, 3))
        lab = np.full(n, -1, dtype=np.int32)
        gnames = {g.index: g.name for g in o.vertex_groups}
        if o.parent_type == "BONE" and o.parent_bone:
            lab[:] = PARTS.index(part_of_bone.get(o.parent_bone, "torso"))
        elif gnames:
            for v in o.data.vertices:
                best = max(((g.weight, gnames.get(g.group)) for g in v.groups), default=(0, None))
                if best[1] in part_of_bone and best[0] > 0:
                    lab[v.index] = PARTS.index(part_of_bone[best[1]])
        labels.append(lab)
        ob = bpy.data.objects.new(o.name + "_baked", me)
        bpy.context.scene.collection.objects.link(ob)
        objs.append(ob)
    for o in list(bpy.data.objects):
        if o not in objs:
            bpy.data.objects.remove(o)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()

    os.makedirs(a.outdir, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=os.path.join(a.outdir, "mesh.glb"), export_format="GLB",
                              export_animations=False, export_skins=False)
    P = np.concatenate(pts)
    L = np.concatenate(labels)
    np.save(os.path.join(a.outdir, "gt_points.npy"), P)
    np.save(os.path.join(a.outdir, "gt_labels.npy"), L)
    zmin, zmax = float(P[:, 2].min()), float(P[:, 2].max())
    with open(os.path.join(a.outdir, "gt.json"), "w") as f:
        json.dump({"family": fam, "a_posed": posed, "joints": joints, "parts": PARTS,
                   "height": zmax - zmin, "labelled_fraction": float((L >= 0).mean()),
                   "n_points": int(len(P))}, f, indent=1)
    print("prepared", a.inp, fam, "a_posed" if posed else "rest", len(P), "verts")


if __name__ == "__main__":
    main()
