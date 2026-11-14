"""
Bake a rigged humanoid glb into an "AI-generated looking" single unrigged mesh (optionally
posed into an A-pose). Used for regression tests and the joint/skinning benchmark.

  blender -b -P tools/make_test_meshes.py -- in.glb out.glb [--apose 48]
"""
import argparse
import math
import sys

import bpy
import mathutils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--apose", type=float, default=0.0, help="upper-arm drop in degrees")
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=a.inp)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    arm.animation_data_clear()
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
    bpy.context.view_layer.update()
    if a.apose:
        for suffix, deg in (("LeftArm", a.apose), ("RightArm", -a.apose)):
            pb = next(p for p in arm.pose.bones if p.name.endswith(suffix))
            M = arm.matrix_world.to_3x3() @ pb.bone.matrix_local.to_3x3()
            R = mathutils.Matrix.Rotation(math.radians(deg), 3, "Y")
            pb.rotation_quaternion = (M.inverted() @ R @ M).to_quaternion()
        bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    new = []
    for o in [o for o in bpy.data.objects if o.type == "MESH"]:
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
        me.transform(o.matrix_world)
        ob = bpy.data.objects.new(o.name + "_baked", me)
        bpy.context.scene.collection.objects.link(ob)
        new.append(ob)
    for o in list(bpy.data.objects):
        if o not in new:
            bpy.data.objects.remove(o)
    bpy.ops.object.select_all(action="DESELECT")
    for o in new:
        o.vertex_groups.clear()
        o.select_set(True)
    bpy.context.view_layer.objects.active = new[0]
    if len(new) > 1:
        bpy.ops.object.join()
    bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB")


if __name__ == "__main__":
    main()
