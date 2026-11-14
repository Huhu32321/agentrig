"""
Step 3 - Mesh cleanup / normalisation (Blender, headless).

Input : raw AI-generated model (glb / obj / fbx)
Output: <out>.blend  - one clean mesh object named "Body", Z-up, facing -Y,
                        feet on Z=0, centred on X/Y, scaled to target height
        textures/*.png - textures extracted per material with role suffixes
        report json   - metrics for QA / the agent

Usage:
  blender -b -P bl_cleanup.py -- --in raw.glb --out clean.blend --report r.json
          [--height 1.75] [--max-faces 50000] [--front -Y] [--texdir dir]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bl_common as C  # noqa: E402

import bmesh  # noqa: E402
import bpy  # noqa: E402

# Map an "AI model says front is X" axis to the Z rotation that turns it to -Y.
FRONT_TO_ROT_Z = {"-Y": 0.0, "+Y": 180.0, "+X": -90.0, "-X": 90.0}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--texdir", default=None)
    p.add_argument("--height", type=float, default=1.75)
    p.add_argument("--max-faces", type=int, default=50000)
    p.add_argument("--merge-dist", type=float, default=1e-5)
    p.add_argument("--front", default="-Y", choices=list(FRONT_TO_ROT_Z))
    return p.parse_args(C.script_args())


def join_meshes():
    # drop everything that is not a mesh (AI exports sometimes carry empties/cameras)
    for o in list(bpy.data.objects):
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    meshes = C.mesh_objects()
    for o in meshes:
        o.parent = None
        for m in list(o.modifiers):
            o.modifiers.remove(m)
    C.select_only(meshes)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    body = bpy.context.view_layer.objects.active
    body.name = "Body"
    body.data.name = "Body"
    return body


def stand_up(body):
    """If the model lies down rotate it upright. A lying body is thinnest along Z;
    an upright T-pose can be wider than tall (long arms), so width alone is not
    evidence (this used to tip wide T-poses onto their side)."""
    mn, mx = C.world_bbox(body)
    ext = mx - mn
    if ext.z > min(ext.x, ext.y):
        return None
    if ext.y > ext.z * 1.15 and ext.y >= ext.x:
        euler = (90.0, 0.0, 0.0)
    elif ext.x > ext.z * 1.15 and ext.x > ext.y:
        euler = (0.0, -90.0, 0.0)
    else:
        return None
    body.rotation_euler = tuple(math.radians(e) for e in euler)
    C.select_only([body])
    bpy.ops.object.transform_apply(rotation=True)
    return euler


def normalise(body, height, front):
    """Rotate (front -> -Y), scale to `height`, put the feet centre at the origin.
    Returns the similarity transform p' = offset + scale * Rz(rot_z_deg) p (recorded in
    the report so external tools, e.g. the benchmark, can map points into this space)."""
    rot = FRONT_TO_ROT_Z[front]
    if rot:
        body.rotation_euler = (0, 0, math.radians(rot))
        C.select_only([body])
        bpy.ops.object.transform_apply(rotation=True)
    mn, mx = C.world_bbox(body)
    s = height / max(mx.z - mn.z, 1e-6)
    body.scale = (s, s, s)
    C.select_only([body])
    bpy.ops.object.transform_apply(scale=True)
    mn, mx = C.world_bbox(body)
    # centre X/Y on the feet, not the bounding box: a big ponytail, cape or
    # sleeves shift the bbox and the character would stand off its root/capsule
    V = C.vertex_array(body)
    low = V[V[:, 2] < mn.z + 0.06 * (mx.z - mn.z)]
    if len(low) >= 20:
        cx = 0.5 * (low[:, 0].min() + low[:, 0].max())
        cy = 0.5 * (low[:, 1].min() + low[:, 1].max())
    else:
        cx, cy = (mn.x + mx.x) / 2, (mn.y + mx.y) / 2
    body.location = (-cx, -cy, -mn.z)
    bpy.ops.object.transform_apply(location=True)
    return {"rot_z_deg": rot, "scale": s, "offset": [-cx, -cy, -mn.z]}


def mesh_stats(body):
    bm = bmesh.new()
    bm.from_mesh(body.data)
    non_manifold_edges = sum(1 for e in bm.edges if not e.is_manifold)
    boundary_edges = sum(1 for e in bm.edges if e.is_boundary)
    # connected islands (by edges)
    bm.verts.ensure_lookup_table()
    seen = set()
    islands = []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack = [v]
        seen.add(v.index)
        count = 0
        while stack:
            cur = stack.pop()
            count += 1
            for e in cur.link_edges:
                o = e.other_vert(cur)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
        islands.append(count)
    bm.free()
    tris = sum(len(p.vertices) - 2 for p in body.data.polygons)
    islands.sort(reverse=True)
    return {
        "verts": len(body.data.vertices),
        "faces": len(body.data.polygons),
        "tris": tris,
        "non_manifold_edges": non_manifold_edges,
        "boundary_edges": boundary_edges,
        "islands": len(islands),
        "largest_islands": islands[:8],
        "uv_layers": len(body.data.uv_layers),
        "materials": [m.name if m else None for m in body.data.materials],
    }


def merge_by_distance(body, dist):
    bm = bmesh.new()
    bm.from_mesh(body.data)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    after = len(bm.verts)
    bm.to_mesh(body.data)
    bm.free()
    return before - after


def decimate(body, max_faces):
    tris = sum(len(p.vertices) - 2 for p in body.data.polygons)
    if tris <= max_faces:
        return 1.0
    ratio = max_faces / tris
    mod = body.modifiers.new("Decimate", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    C.select_only([body])
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return ratio


def socket_image(node_input, depth=0):
    """Follow a shader input upstream until an Image Texture node is found."""
    if depth > 6 or not node_input.is_linked:
        return None
    n = node_input.links[0].from_node
    if n.type == "TEX_IMAGE":
        return n.image
    for i in n.inputs:
        img = socket_image(i, depth + 1)
        if img is not None:
            return img
    return None


def fix_emissive_only(body):
    """Some web image-to-3D exporters bake the colour into the *emissive*
    texture and leave baseColor = black (an 'unlit' look). In a lit engine
    that renders as a black character, so move the texture to Base Color and
    switch emission off. Returns the converted material names."""
    fixed = []
    for mat in body.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        bc = bsdf.inputs.get("Base Color")
        em = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
        if bc is None or em is None or bc.is_linked or not em.is_linked:
            continue
        if max(bc.default_value[:3]) > 0.02 or socket_image(em) is None:
            continue
        src = em.links[0].from_socket
        nt.links.remove(em.links[0])
        nt.links.new(src, bc)
        em.default_value = (0, 0, 0, 1)
        st = bsdf.inputs.get("Emission Strength")
        if st is not None:
            st.default_value = 0.0
        fixed.append(mat.name)
    return fixed


def extract_textures(body, texdir):
    """Save textures per material with UE-friendly role names."""
    os.makedirs(texdir, exist_ok=True)
    out = {}
    for mat in body.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        roles = {
            "BaseColor": bsdf.inputs.get("Base Color"),
            "Normal": bsdf.inputs.get("Normal"),
            # glTF packs metallic (B) + roughness (G) in one texture
            "MetallicRoughness": bsdf.inputs.get("Roughness") or bsdf.inputs.get("Metallic"),
            "Emissive": bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission"),
        }
        safe = "".join(ch if ch.isalnum() else "_" for ch in mat.name)
        entry = {"material": mat.name, "safe_name": safe, "textures": {}}
        for role, sock in roles.items():
            if sock is None:
                continue
            img = socket_image(sock)
            if img is None:
                continue
            path = os.path.join(texdir, f"T_{safe}_{role}.png")
            try:
                if img.packed_file is None and img.filepath and os.path.exists(bpy.path.abspath(img.filepath)):
                    img.reload()
                img.filepath_raw = path
                img.file_format = "PNG"
                img.save()
            except Exception:
                try:
                    img.save_render(path)
                except Exception as e:  # pragma: no cover
                    C.log("texture save failed", role, e)
                    continue
            entry["textures"][role] = path
        # scalar fallbacks so UE can build a material even without textures
        bc = bsdf.inputs.get("Base Color")
        if bc is not None and not bc.is_linked:
            entry["base_color_value"] = list(bc.default_value)
        out[mat.name] = entry
    return out


def main():
    a = parse()
    C.reset_scene()
    C.import_any(a.inp)
    body = join_meshes()
    stood = stand_up(body)
    norm = normalise(body, a.height, a.front)
    norm["stand_up_euler_deg"] = stood
    merged = merge_by_distance(body, a.merge_dist)
    ratio = decimate(body, a.max_faces)
    if len(body.data.materials) == 0:
        body.data.materials.append(bpy.data.materials.new("M_Body"))
    texdir = a.texdir or os.path.join(os.path.dirname(os.path.abspath(a.out)), "textures")
    if os.path.isdir(texdir):  # re-run: drop textures of the previous run
        for fn in os.listdir(texdir):
            if fn.startswith("T_") and fn.endswith(".png"):
                os.remove(os.path.join(texdir, fn))
    emissive_fixed = fix_emissive_only(body)
    if emissive_fixed:
        C.log("emissive-only material -> colour moved to Base Color:", emissive_fixed)
    textures = extract_textures(body, texdir)
    # Shade smooth keeps the AI normals readable after decimation
    for p in body.data.polygons:
        p.use_smooth = True
    mn, mx = C.world_bbox(body)
    stats = mesh_stats(body)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    report = {
        "ok": True,
        "input": a.inp,
        "output_blend": os.path.abspath(a.out),
        "auto_stood_up": bool(stood),
        "normalisation": norm,
        "front_axis_in": a.front,
        "merged_vertices": merged,
        "decimate_ratio": ratio,
        "bbox_min": list(mn),
        "bbox_max": list(mx),
        "height_m": mx.z - mn.z,
        "textures": textures,
        "emissive_only_converted": emissive_fixed,
        **stats,
    }
    C.write_json(a.report, report)
    C.log("cleanup done", stats)


if __name__ == "__main__":
    main()
