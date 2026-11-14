"""
Shared helpers for all Blender-side scripts.

These scripts run either inside Blender (`blender -b -P script.py -- args`)
or with the pip `bpy` module (`python script.py args`). Both are supported so
the pipeline works on an artist's machine (Blender installed) and in CI.
"""
import json
import math
import os
import sys

import bpy
import mathutils


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def script_args():
    """Return the argument list meant for the script (after `--` in Blender)."""
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return argv[1:]


def write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log(*a):
    print("[pipeline]", *a, flush=True)


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=path)
    else:
        raise ValueError(f"Unsupported model format: {ext}")


def select_only(objs, active=None):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or (objs[0] if objs else None)


def mesh_objects():
    return [o for o in bpy.data.objects if o.type == "MESH"]


def get_main_mesh():
    meshes = mesh_objects()
    if not meshes:
        raise RuntimeError("No mesh object in scene")
    return max(meshes, key=lambda o: len(o.data.vertices))


def world_bbox(obj):
    pts = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    mn = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


def vertex_array(obj):
    """World-space vertex coordinates as numpy (N,3)."""
    import numpy as np
    n = len(obj.data.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    m = np.array(obj.matrix_world)
    return co @ m[:3, :3].T + m[:3, 3]


# --------------------------------------------------------------------------
# Rendering (Cycles CPU works headless everywhere; Workbench needs a GPU ctx)
# --------------------------------------------------------------------------
def setup_render(engine="CYCLES", res=768, samples=16, bg=(1, 1, 1), flat=True):
    sc = bpy.context.scene
    sc.render.engine = engine
    sc.render.resolution_x = res
    sc.render.resolution_y = res
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    if engine == "CYCLES":
        sc.cycles.samples = samples
        sc.cycles.device = "CPU"
        try:
            sc.cycles.use_denoising = False
        except Exception:
            pass
    else:
        # flat = looks like the concept art (for pose estimation); studio = shows deformation (QA)
        sc.display.shading.light = "FLAT" if flat else "STUDIO"
        sc.display.shading.color_type = "TEXTURE"
    world = bpy.data.worlds.get("PipelineWorld") or bpy.data.worlds.new("PipelineWorld")
    world.use_nodes = True
    bgnode = world.node_tree.nodes.get("Background")
    if bgnode is None:
        bgnode = world.node_tree.nodes.new("ShaderNodeBackground")
    bgnode.inputs[0].default_value = (bg[0], bg[1], bg[2], 1.0)
    bgnode.inputs[1].default_value = 1.0
    world.color = bg  # Workbench ignores the node tree and uses this colour
    sc.world = world
    if engine != "CYCLES":
        try:
            sc.display.shading.background_type = "WORLD"
        except Exception:
            pass
    try:
        sc.view_settings.view_transform = "Standard"
    except Exception:
        pass


def ortho_camera(name, location, rotation_deg, ortho_scale):
    cam_data = bpy.data.cameras.get(name) or bpy.data.cameras.new(name)
    cam = bpy.data.objects.get(name) or bpy.data.objects.new(name, cam_data)
    if cam.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = ortho_scale
    cam.data.clip_end = 1000
    cam.location = location
    cam.rotation_euler = [math.radians(a) for a in rotation_deg]
    bpy.context.scene.camera = cam
    return cam


def render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def front_camera_for(obj, margin=1.1):
    """Orthographic camera looking at the character front (-Y side, looking +Y)."""
    mn, mx = world_bbox(obj)
    size = max(mx.x - mn.x, mx.z - mn.z) * margin
    cx, cz = (mn.x + mx.x) / 2, (mn.z + mx.z) / 2
    cam = ortho_camera("PipelineFrontCam", (cx, mn.y - 10.0, cz), (90, 0, 0), size)
    return cam, {"center_x": cx, "center_z": cz, "ortho_scale": size}


def side_camera_for(obj, margin=1.1):
    """Orthographic camera looking at the character's left side (+X looking -X)."""
    mn, mx = world_bbox(obj)
    size = max(mx.y - mn.y, mx.z - mn.z) * margin
    cy, cz = (mn.y + mx.y) / 2, (mn.z + mx.z) / 2
    cam = ortho_camera("PipelineSideCam", (mx.x + 10.0, cy, cz), (90, 0, 90), size)
    return cam, {"center_y": cy, "center_z": cz, "ortho_scale": size}
