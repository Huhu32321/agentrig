"""
Step 8 - Build the character inside Unreal Engine 5.7 (runs in UE's Python).

Reads a manifest JSON written by the pipeline and:
  1. imports textures (correct sRGB / compression / normal-map green flip)
  2. imports the skinned FBX as a Skeletal Mesh (+ Skeleton + Physics Asset)
  3. builds a material + material instance per slot and assigns them
  4. creates the target IK Rig (auto retarget chains + auto FBIK)
  5. finds / imports source animations (UE Mannequin "Manny" or a custom
     FBX mocap set), creates an IK Retargeter, auto maps + aligns chains,
     and batch-retargets Idle / Walk onto the new character
  6. optionally retargets the template Anim Blueprint and creates a
     playable Character Blueprint
  7. builds a preview level (Idle actor, Walk actor, lights, floor)
  8. writes a JSON report for the pipeline QA step

Run (either works):
  UnrealEditor-Cmd.exe Proj.uproject -run=pythonscript -script="ue_build_character.py manifest.json"
  (open editor) Output Log > Python: py "ue_build_character.py" "manifest.json"
"""
import json
import os
import sys
import traceback

import unreal

AT = unreal.AssetToolsHelpers.get_asset_tools()
EAL = unreal.EditorAssetLibrary
MEL = unreal.MaterialEditingLibrary
SRC = unreal.RetargetSourceOrTarget.SOURCE
TGT = unreal.RetargetSourceOrTarget.TARGET

REPORT = {"ok": False, "steps": {}, "warnings": [], "assets": {}}
PREVIEW = {}  # preview-level camera, used for the automatic screenshot


def log(*a):
    unreal.log("[AICharPipeline] " + " ".join(str(x) for x in a))


def warn(msg):
    REPORT["warnings"].append(msg)
    unreal.log_warning("[AICharPipeline] " + msg)


def step(name):
    """Decorator: run a build step, record success / error in the report."""
    def deco(fn):
        def run(*a, **k):
            log("==>", name)
            try:
                res = fn(*a, **k)
                REPORT["steps"][name] = {"ok": True}
                return res
            except Exception as e:  # keep going; the agent reads the report
                REPORT["steps"][name] = {"ok": False, "error": str(e),
                                         "trace": traceback.format_exc()[-2000:]}
                unreal.log_error(f"[AICharPipeline] {name} failed: {e}")
                return None
        return run
    return deco


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def ensure_dir(path):
    if not EAL.does_directory_exist(path):
        EAL.make_directory(path)


def import_file(filename, dest, name=None, options=None):
    t = unreal.AssetImportTask()
    t.set_editor_property("filename", filename)
    t.set_editor_property("destination_path", dest)
    if name:
        t.set_editor_property("destination_name", name)
    t.set_editor_property("automated", True)
    t.set_editor_property("replace_existing", True)
    t.set_editor_property("save", True)
    if options is not None:
        t.set_editor_property("options", options)
    AT.import_asset_tasks([t])
    paths = list(t.get_editor_property("imported_object_paths") or [])
    return [unreal.load_asset(p) for p in paths]


def set_legacy_fbx(enabled):
    """UE 5.5+ routes FBX through Interchange; the FbxImportUI options below are
    for the classic importer, so switch it on while we import."""
    val = "false" if enabled else "true"
    for cvar in ("Interchange.FeatureFlags.Import.FBX", "Interchange.FeatureFlags.Import.FBX.ToLevel"):
        unreal.SystemLibrary.execute_console_command(None, f"{cvar} {val}")


def class_name(ad):
    try:
        return str(ad.asset_class_path.asset_name)
    except Exception:
        return str(ad.asset_class)


_SCANNED = set()


def find_assets(class_names, root="/Game", name_filter=None, rescan=False):
    ar = unreal.AssetRegistryHelpers.get_asset_registry()
    if rescan or root not in _SCANNED:
        # in a commandlet the asset registry is not populated yet -> scan synchronously
        try:
            ar.scan_paths_synchronous([root], True)
        except Exception:
            ar.search_all_assets(True)
        _SCANNED.add(root)
    out = []
    for ad in ar.get_assets_by_path(root, recursive=True):
        if class_name(ad) in class_names:
            n = str(ad.asset_name)
            if name_filter is None or name_filter(n):
                out.append(str(ad.package_name))
    return sorted(out)


def save_all(paths):
    for p in paths:
        try:
            EAL.save_asset(p, only_if_is_dirty=False)
        except Exception:
            pass


# --------------------------------------------------------------------------
# 1. textures
# --------------------------------------------------------------------------
@step("import_textures")
def import_textures(man, base):
    dest = f"{base}/Textures"
    ensure_dir(dest)
    result = {}
    for mat_name, entry in man.get("materials", {}).items():
        texs = {}
        for role, path in entry.get("textures", {}).items():
            if not os.path.exists(path):
                warn(f"texture missing on disk: {path}")
                continue
            assets = import_file(path, dest, os.path.splitext(os.path.basename(path))[0])
            if not assets:
                warn(f"texture import failed: {path}")
                continue
            tex = assets[0]
            if role == "Normal":
                tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
                tex.set_editor_property("srgb", False)
                tex.set_editor_property("flip_green_channel", True)  # glTF/OpenGL -> DirectX
            elif role == "MetallicRoughness":
                tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_MASKS)
                tex.set_editor_property("srgb", False)
            EAL.save_loaded_asset(tex)
            texs[role] = tex
        result[mat_name] = texs
    return result


# --------------------------------------------------------------------------
# 2. skeletal mesh
# --------------------------------------------------------------------------
@step("import_skeletal_mesh")
def import_skeletal_mesh(man, base):
    dest = f"{base}/Mesh"
    ensure_dir(dest)
    ui = unreal.FbxImportUI()
    ui.set_editor_property("import_mesh", True)
    ui.set_editor_property("import_as_skeletal", True)
    ui.set_editor_property("mesh_type_to_import", unreal.FBXImportType.FBXIT_SKELETAL_MESH)
    ui.set_editor_property("import_animations", False)
    ui.set_editor_property("import_materials", False)
    ui.set_editor_property("import_textures", False)
    ui.set_editor_property("create_physics_asset", True)
    skd = ui.get_editor_property("skeletal_mesh_import_data")
    skd.set_editor_property("import_morph_targets", False)
    skd.set_editor_property("convert_scene", True)
    skd.set_editor_property("update_skeleton_reference_pose", False)
    set_legacy_fbx(True)
    try:
        assets = import_file(man["fbx"], dest, f"SK_{man['name']}", ui)
    finally:
        set_legacy_fbx(False)
    sk = next((a for a in assets if isinstance(a, unreal.SkeletalMesh)), None)
    if sk is None:
        found = find_assets({"SkeletalMesh"}, dest)
        sk = unreal.load_asset(found[0]) if found else None
    if sk is None:
        raise RuntimeError("Skeletal mesh import produced no SkeletalMesh")
    skel = sk.get_editor_property("skeleton")
    bounds = sk.get_bounds()
    height = bounds.box_extent.z * 2
    REPORT["mesh"] = {
        "path": sk.get_path_name(), "skeleton": skel.get_path_name() if skel else None,
        "height_cm": height, "bone_count": len(skel.get_editor_property("bone_tree")) if skel else None,
    }
    if not (100 < height < 300):
        warn(f"unexpected character height {height:.1f} cm (scale problem?)")
    return sk


# --------------------------------------------------------------------------
# 3. materials
# --------------------------------------------------------------------------
def tex_node(mat, tex, param, sampler, x, y):
    n = MEL.create_material_expression(mat, unreal.MaterialExpressionTextureSampleParameter2D, x, y)
    n.set_editor_property("parameter_name", param)
    n.set_editor_property("texture", tex)
    n.set_editor_property("sampler_type", sampler)
    return n


def scalar(mat, name, value, x, y):
    n = MEL.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, x, y)
    n.set_editor_property("parameter_name", name)
    n.set_editor_property("default_value", value)
    return n


def multiply(mat, a, a_out, b, b_out, x, y):
    m = MEL.create_material_expression(mat, unreal.MaterialExpressionMultiply, x, y)
    MEL.connect_material_expressions(a, a_out, m, "A")
    MEL.connect_material_expressions(b, b_out, m, "B")
    return m


def build_material(name, dest, texs, base_color_value=None):
    """Parent material wired only with the textures that exist, plus artist
    parameters (Tint, RoughnessScale, MetallicScale, Specular)."""
    mpath = f"{dest}/M_{name}"
    for p in (f"{dest}/MI_{name}", mpath):  # re-runs: instance first (it references the parent)
        if EAL.does_asset_exist(p):
            EAL.delete_asset(p)
    mat = AT.create_asset(f"M_{name}", dest, unreal.Material, unreal.MaterialFactoryNew())
    tint = MEL.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, -600, -200)
    tint.set_editor_property("parameter_name", "Tint")
    tint.set_editor_property("default_value", unreal.LinearColor(1, 1, 1, 1))
    if "BaseColor" in texs:
        bc = tex_node(mat, texs["BaseColor"], "BaseColorTex", unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, -900, -300)
        m = multiply(mat, bc, "RGB", tint, "", -350, -250)
        MEL.connect_material_property(m, "", unreal.MaterialProperty.MP_BASE_COLOR)
    else:
        if base_color_value:
            tint.set_editor_property("default_value", unreal.LinearColor(*base_color_value[:3], 1))
        MEL.connect_material_property(tint, "", unreal.MaterialProperty.MP_BASE_COLOR)
    if "Normal" in texs:
        nm = tex_node(mat, texs["Normal"], "NormalTex", unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL, -900, 100)
        MEL.connect_material_property(nm, "RGB", unreal.MaterialProperty.MP_NORMAL)
    rs = scalar(mat, "RoughnessScale", 1.0 if "MetallicRoughness" in texs else 0.65, -600, 350)
    ms = scalar(mat, "MetallicScale", 1.0 if "MetallicRoughness" in texs else 0.0, -600, 450)
    if "MetallicRoughness" in texs:
        mr = tex_node(mat, texs["MetallicRoughness"], "MetallicRoughnessTex",
                      unreal.MaterialSamplerType.SAMPLERTYPE_MASKS, -900, 400)
        r = multiply(mat, mr, "G", rs, "", -350, 350)   # glTF: G = roughness
        mt = multiply(mat, mr, "B", ms, "", -350, 450)  # glTF: B = metallic
        MEL.connect_material_property(r, "", unreal.MaterialProperty.MP_ROUGHNESS)
        MEL.connect_material_property(mt, "", unreal.MaterialProperty.MP_METALLIC)
    else:
        MEL.connect_material_property(rs, "", unreal.MaterialProperty.MP_ROUGHNESS)
        MEL.connect_material_property(ms, "", unreal.MaterialProperty.MP_METALLIC)
    spec = scalar(mat, "Specular", 0.4, -600, 550)
    MEL.connect_material_property(spec, "", unreal.MaterialProperty.MP_SPECULAR)
    MEL.layout_material_expressions(mat)
    MEL.recompile_material(mat)
    EAL.save_loaded_asset(mat)
    mi = AT.create_asset(f"MI_{name}", dest, unreal.MaterialInstanceConstant,
                         unreal.MaterialInstanceConstantFactoryNew())
    MEL.set_material_instance_parent(mi, mat)
    EAL.save_loaded_asset(mi)
    return mi


def norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


@step("materials")
def materials(man, base, sk, textures):
    dest = f"{base}/Materials"
    ensure_dir(dest)
    textures = textures or {}
    slots = list(sk.get_editor_property("materials"))
    new_slots, made = [], {}
    for i, slot in enumerate(slots):
        sname = str(slot.get_editor_property("material_slot_name"))
        # match slot -> manifest material by normalised name, else by index
        key = next((k for k in man.get("materials", {}) if norm(k) == norm(sname)), None)
        if key is None and i < len(man.get("materials", {})):
            key = list(man["materials"].keys())[i]
        entry = man.get("materials", {}).get(key, {}) if key else {}
        safe = entry.get("safe_name") or f"{man['name']}_{i}"
        mi = build_material(f"{man['name']}_{safe}", dest, textures.get(key, {}),
                            entry.get("base_color_value"))
        slot.set_editor_property("material_interface", mi)
        new_slots.append(slot)
        made[sname] = mi.get_path_name()
    sk.set_editor_property("materials", new_slots)
    EAL.save_loaded_asset(sk)
    REPORT["assets"]["materials"] = made
    return made


# --------------------------------------------------------------------------
# 4. IK rigs
# --------------------------------------------------------------------------
MANUAL_CHAINS = [  # fallback if UE cannot auto-detect the skeleton template
    ("Spine", "spine_01", "spine_05", ""), ("Neck", "neck_01", "neck_02", ""),
    ("Head", "head", "head", ""),
    ("LeftClavicle", "clavicle_l", "clavicle_l", ""), ("RightClavicle", "clavicle_r", "clavicle_r", ""),
    ("LeftArm", "upperarm_l", "hand_l", "hand_l_Goal"), ("RightArm", "upperarm_r", "hand_r", "hand_r_Goal"),
    ("LeftLeg", "thigh_l", "ball_l", "foot_l_Goal"), ("RightLeg", "thigh_r", "ball_r", "foot_r_Goal"),
]


def new_ik_rig(dest, name, mesh):
    path = f"{dest}/{name}"
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    try:
        ikr = unreal.IKRigDefinitionFactory.create_new_ik_rig_asset(dest, name)
    except Exception:
        ikr = AT.create_asset(name, dest, unreal.IKRigDefinition, unreal.IKRigDefinitionFactory())
    c = unreal.IKRigController.get_controller(ikr)
    c.set_skeletal_mesh(mesh)
    auto_chains = bool(c.apply_auto_generated_retarget_definition())
    auto_fbik = bool(c.apply_auto_fbik())
    if not auto_chains:
        warn(f"{name}: skeleton template not recognised - using manual chains")
        c.set_retarget_root("pelvis")
        for chain, start, end, goal in MANUAL_CHAINS:
            try:
                c.add_retarget_chain(chain, start, end, goal)
            except Exception as e:
                warn(f"chain {chain}: {e}")
    EAL.save_loaded_asset(ikr)
    return ikr, {"auto_chains": auto_chains, "auto_fbik": auto_fbik}


@step("target_ik_rig")
def target_ik_rig(man, base, sk):
    dest = f"{base}/Rig"
    ensure_dir(dest)
    ikr, info = new_ik_rig(dest, f"IK_{man['name']}", sk)
    REPORT["assets"]["ik_rig_target"] = ikr.get_path_name()
    REPORT["ik_rig_target"] = info
    return ikr


# --------------------------------------------------------------------------
# 5. source animations
# --------------------------------------------------------------------------
def pick_anim(paths, patterns):
    for pat in patterns:
        for p in paths:
            if pat.lower() in p.rsplit("/", 1)[-1].lower():
                return p
    return None


@step("source_animations")
def source_animations(man, base):
    cfg = man.get("anim_source", {"type": "manny"})
    if cfg.get("type") == "fbx":
        return import_fbx_anim_source(man, base, cfg)
    root = cfg.get("search_root", "/Game")
    meshes = find_assets({"SkeletalMesh"}, root, lambda n: n in (
        "SKM_Manny_Simple", "SKM_Manny", "SKM_Quinn_Simple", "SKM_Quinn"))
    if not meshes:
        raise RuntimeError("UE Mannequin not found. Add the Third Person template content "
                           "(Content Drawer > Add > Add Feature or Content Pack > Third Person) "
                           "or set anim_source.type = fbx in config.")
    src_mesh = unreal.load_asset(meshes[0])
    src_skel = src_mesh.get_editor_property("skeleton").get_path_name()
    anims = [p for p in find_assets({"AnimSequence"}, root)
             if unreal.load_asset(p).get_editor_property("skeleton").get_path_name() == src_skel]
    idle = cfg.get("idle") or pick_anim(anims, ["MM_Idle", "Unarmed_Idle", "Idle"])
    walk = cfg.get("walk") or pick_anim(anims, ["Unarmed_Walk_Fwd", "Walk_Fwd", "Walk"])
    rigs = find_assets({"IKRigDefinition"}, root, lambda n: n in ("IK_Mannequin", "IK_Manny"))
    if rigs:
        src_rig = unreal.load_asset(rigs[0])
    else:
        ensure_dir(f"{base}/Rig")
        src_rig, _ = new_ik_rig(f"{base}/Rig", "IK_Source_Mannequin", src_mesh)
    abps = find_assets({"AnimBlueprint"}, root, lambda n: n in ("ABP_Unarmed", "ABP_Manny", "ABP_Quinn"))
    REPORT["anim_source"] = {"type": "manny", "mesh": meshes[0], "idle": idle, "walk": walk,
                             "ik_rig": src_rig.get_path_name(), "anim_blueprint": abps[0] if abps else None}
    if not idle or not walk:
        warn(f"idle/walk animation not found automatically (idle={idle}, walk={walk})")
    return {"mesh": src_mesh, "rig": src_rig, "anims": {"Idle": idle, "Walk": walk},
            "abp": abps[0] if abps else None}


def import_fbx_anim_source(man, base, cfg):
    """Automates the README procedure for a mocap dataset: import the actor
    mesh+skeleton, then each animation FBX with 'Import Only Animation'."""
    dest_mesh = f"{base}/AnimSource/Mesh"
    dest_anim = f"{base}/AnimSource/Anims"
    ensure_dir(dest_mesh)
    ensure_dir(dest_anim)
    set_legacy_fbx(True)
    try:
        ui = unreal.FbxImportUI()
        ui.set_editor_property("import_mesh", True)
        ui.set_editor_property("import_as_skeletal", True)
        ui.set_editor_property("mesh_type_to_import", unreal.FBXImportType.FBXIT_SKELETAL_MESH)
        ui.set_editor_property("import_animations", False)
        ui.set_editor_property("create_physics_asset", False)
        assets = import_file(cfg["skeleton_fbx"], dest_mesh, None, ui)
        src_mesh = next(a for a in assets if isinstance(a, unreal.SkeletalMesh))
        skel = src_mesh.get_editor_property("skeleton")
        names = {}
        for role, fbx in (cfg.get("anims") or {}).items():
            aui = unreal.FbxImportUI()
            aui.set_editor_property("import_mesh", False)
            aui.set_editor_property("import_as_skeletal", True)
            aui.set_editor_property("mesh_type_to_import", unreal.FBXImportType.FBXIT_ANIMATION)
            aui.set_editor_property("import_animations", True)
            aui.set_editor_property("skeleton", skel)
            got = import_file(fbx, dest_anim, f"A_Src_{role}", aui)
            seq = next((g for g in got if isinstance(g, unreal.AnimSequence)), None)
            names[role] = seq.get_path_name().split(".")[0] if seq else None
    finally:
        set_legacy_fbx(False)
    ensure_dir(f"{base}/Rig")
    src_rig, info = new_ik_rig(f"{base}/Rig", "IK_Source_Mocap", src_mesh)
    REPORT["anim_source"] = {"type": "fbx", "mesh": src_mesh.get_path_name(), "anims": names,
                             "ik_rig": src_rig.get_path_name(), "ik_rig_info": info}
    return {"mesh": src_mesh, "rig": src_rig, "anims": names, "abp": None}


# --------------------------------------------------------------------------
# 6. retarget
# --------------------------------------------------------------------------
@step("retargeter")
def retargeter(man, base, src, sk, tgt_rig):
    dest = f"{base}/Rig"
    name = f"RTG_{man['name']}"
    if EAL.does_asset_exist(f"{dest}/{name}"):
        EAL.delete_asset(f"{dest}/{name}")
    rtg = AT.create_asset(name, dest, unreal.IKRetargeter, unreal.IKRetargetFactory())
    c = unreal.IKRetargeterController.get_controller(rtg)
    c.set_ik_rig(SRC, src["rig"])
    c.set_ik_rig(TGT, tgt_rig)
    # UE 5.6+ uses a retarget op stack
    if hasattr(c, "add_default_ops"):
        try:
            c.add_default_ops()
            c.assign_ik_rig_to_all_ops(SRC, src["rig"])
            c.assign_ik_rig_to_all_ops(TGT, tgt_rig)
        except Exception as e:
            warn(f"op stack setup: {e}")
    c.set_preview_mesh(SRC, src["mesh"])
    c.set_preview_mesh(TGT, sk)
    c.auto_map_chains(unreal.AutoMapChainType.FUZZY, True)
    try:  # align target ref pose (A/T pose) to the source pose
        c.auto_align_all_bones(TGT)
    except Exception as e:
        warn(f"auto align: {e}")
    EAL.save_loaded_asset(rtg)
    REPORT["assets"]["retargeter"] = rtg.get_path_name()
    return rtg


def batch_retarget(asset_paths, src_mesh, sk, rtg, suffix, dest):
    """Duplicate + retarget. UE 5.7's duplicate_and_retarget has no output-folder
    argument: new assets land in /Game and referenced assets (blend spaces,
    anims used by an ABP) are not all returned. So we collect every new asset
    in /Game whose name ends with the suffix and move them in ONE batch rename
    (keeps references between them valid)."""
    ads = [EAL.find_asset_data(p) for p in asset_paths]
    res = unreal.IKRetargetBatchOperation.duplicate_and_retarget(
        ads, src_mesh, sk, rtg, suffix=suffix, include_referenced_assets=True,
        overwrite_existing_files=True)
    created = {str(ad.package_name) for ad in res}
    ar = unreal.AssetRegistryHelpers.get_asset_registry()
    ar.scan_paths_synchronous(["/Game"], True)
    for ad in ar.get_assets_by_path("/Game", recursive=False):
        if str(ad.asset_name).endswith(suffix):
            created.add(str(ad.package_name))
    ensure_dir(dest)
    moves, out = [], []
    for p in sorted(created):
        if p.startswith(dest + "/"):
            out.append(p)
            continue
        obj = unreal.load_asset(p)
        if obj is None:
            continue
        name = p.rsplit("/", 1)[-1]
        if EAL.does_asset_exist(f"{dest}/{name}"):
            EAL.delete_asset(f"{dest}/{name}")
        moves.append(unreal.AssetRenameData(asset=obj, new_package_path=dest, new_name=name))
        out.append(f"{dest}/{name}")
    if moves:
        AT.rename_assets(moves)
    return out


@step("retarget_animations")
def retarget_animations(man, base, src, sk, rtg):
    dest = f"{base}/Animations"
    ensure_dir(dest)
    todo = {k: v for k, v in src["anims"].items() if v}
    made = batch_retarget(list(todo.values()), src["mesh"], sk, rtg, f"_{man['name']}", dest)
    result = {}
    for role, srcp in todo.items():
        base_name = srcp.rsplit("/", 1)[-1]
        hit = next((m for m in made if m.rsplit("/", 1)[-1].startswith(base_name)), None)
        result[role] = hit
    REPORT["assets"]["animations"] = result
    if not all(result.values()):
        warn(f"some animations were not retargeted: {result}")
    return result


@step("anim_blueprint")
def anim_blueprint(man, base, src, sk, rtg):
    if not src.get("abp") or not man.get("retarget_anim_blueprint", True):
        return None
    dest = f"{base}/Animations/ABP"
    ensure_dir(dest)
    suffix = f"_{man['name']}"
    made = batch_retarget([src["abp"]], src["mesh"], sk, rtg, suffix, dest)
    abp = next((m for m in made if m.rsplit("/", 1)[-1].startswith("ABP_")), None)
    if abp is None:
        # UE 5.7 creates the duplicated ABP but does not return it (and may put it
        # in /Game) - find it by name and move it next to the other assets
        want = src["abp"].rsplit("/", 1)[-1].split(".")[0] + suffix
        hits = find_assets({"AnimBlueprint"}, "/Game", lambda n: n == want, rescan=True)
        if hits:
            abp = hits[0]
            target = f"{dest}/{want}"
            if abp != target:
                if EAL.does_asset_exist(target):
                    EAL.delete_asset(target)
                if EAL.rename_asset(abp, target):
                    abp = target
        else:
            warn(f"retargeted anim blueprint {want} not found")
    REPORT["assets"]["anim_blueprint"] = abp
    return abp


@step("bake_ik_bones")
def bake_ik_bones(man, base, sk):
    """Make ik_foot_l/r follow the feet in every retargeted animation.

    The retargeter only drives bones that belong to a retarget chain; the IK helper
    bones keep their reference pose, i.e. they stay where the feet were in the
    reference pose while the character walks. The template's foot-IK Control Rig
    then pins the feet to that spot (ankles glued together, no stride). Here the
    helpers get the foot's root-relative transform for every frame, which is what
    the Mannequin's own animations contain."""
    skel = sk.get_editor_property("skeleton").get_path_name()
    AP = unreal.AnimPoseExtensions
    W = unreal.AnimPoseSpaces.WORLD
    done = []
    for p in EAL.list_assets(f"{base}/Animations", recursive=True, include_folder=False):
        a = unreal.load_asset(p)
        if not isinstance(a, unreal.AnimSequence) or a.get_editor_property("skeleton").get_path_name() != skel:
            continue
        n = unreal.AnimationLibrary.get_num_keys(a)
        tracks = {str(t) for t in unreal.AnimationLibrary.get_animation_track_names(a)}
        keys = {b: ([], [], []) for b in ("ik_foot_root", "ik_foot_l", "ik_foot_r")}
        opts = unreal.AnimPoseEvaluationOptions()
        for f in range(n):
            pose = AP.get_anim_pose_at_frame(a, f, opts)
            root = AP.get_bone_pose(pose, "root", W)
            for b, src in (("ik_foot_root", "root"), ("ik_foot_l", "foot_l"), ("ik_foot_r", "foot_r")):
                t = (unreal.Transform() if b == "ik_foot_root" else
                     unreal.MathLibrary.make_relative_transform(AP.get_bone_pose(pose, src, W), root))
                keys[b][0].append(t.translation)
                keys[b][1].append(t.rotation)
                keys[b][2].append(unreal.Vector(1, 1, 1))
        c = a.controller
        c.open_bracket("bake ik bones", False)
        for b, (pos, rot, scl) in keys.items():
            if b not in tracks:
                c.add_bone_track(b, False)
            c.set_bone_track_keys(b, pos, rot, scl, False)
        c.close_bracket(False)
        EAL.save_loaded_asset(a)
        done.append(p.rsplit("/", 1)[-1].split(".")[0])
    REPORT["ik_bones_baked"] = done
    if not done:
        warn("bake_ik_bones: no animations found")
    return done


@step("character_blueprint")
def character_blueprint(man, base, sk, abp_path):
    """Duplicate the template third-person character and swap mesh + anim BP."""
    if not abp_path:
        return None
    bps = find_assets({"Blueprint"}, "/Game", lambda n: n == "BP_ThirdPersonCharacter")
    if not bps:
        warn("BP_ThirdPersonCharacter not found - playable character skipped")
        return None
    dst = f"{base}/BP_{man['name']}_Character"
    if EAL.does_asset_exist(dst):
        EAL.delete_asset(dst)
    bp = EAL.duplicate_asset(bps[0], dst)
    gen = unreal.BlueprintEditorLibrary.generated_class(bp)
    cdo = unreal.get_default_object(gen)
    comp = cdo.get_editor_property("mesh")
    try:
        comp.set_editor_property("skeletal_mesh_asset", sk)
    except Exception:
        comp.set_editor_property("skeletal_mesh", sk)
    abp = unreal.load_asset(abp_path)
    comp.set_editor_property("anim_class", unreal.BlueprintEditorLibrary.generated_class(abp))
    unreal.BlueprintEditorLibrary.compile_blueprint(bp)
    EAL.save_loaded_asset(bp)
    REPORT["assets"]["character_blueprint"] = dst
    return dst


@step("game_mode")
def game_mode(man, base, char_bp):
    """Duplicate the template GameMode with DefaultPawnClass = our character, so
    pressing Play in the preview level spawns *this* character (not Manny)."""
    if not char_bp:
        return None
    gms = find_assets({"Blueprint"}, "/Game", lambda n: n == "BP_ThirdPersonGameMode")
    if not gms:
        warn("BP_ThirdPersonGameMode not found - Play will use the project default pawn")
        return None
    dst = f"{base}/GM_{man['name']}"
    if EAL.does_asset_exist(dst):
        EAL.delete_asset(dst)
    bp = EAL.duplicate_asset(gms[0], dst)
    cdo = unreal.get_default_object(unreal.BlueprintEditorLibrary.generated_class(bp))
    cdo.set_editor_property("default_pawn_class",
                            unreal.BlueprintEditorLibrary.generated_class(unreal.load_asset(char_bp)))
    unreal.BlueprintEditorLibrary.compile_blueprint(bp)
    EAL.save_loaded_asset(bp)
    REPORT["assets"]["game_mode"] = dst
    return dst


# --------------------------------------------------------------------------
# 7. preview level
# --------------------------------------------------------------------------
def spawn(cls, loc, rot=(0, 0, 0), label=None):
    """rot = (pitch, yaw, roll) in degrees."""
    sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    r = unreal.Rotator(pitch=rot[0], yaw=rot[1], roll=rot[2])
    a = sub.spawn_actor_from_class(cls, unreal.Vector(*loc), r)
    if label:
        a.set_actor_label(label)
    return a


def anim_actor(sk, anim_path, loc, label):
    a = spawn(unreal.SkeletalMeshActor, loc, (0, 0, 0), label)
    comp = a.get_editor_property("skeletal_mesh_component")
    try:
        comp.set_skeletal_mesh_asset(sk)
    except Exception:
        comp.set_editor_property("skeletal_mesh", sk)
    if anim_path:
        comp.set_editor_property("animation_mode", unreal.AnimationMode.ANIMATION_SINGLE_NODE)
        data = comp.get_editor_property("animation_data")
        data.set_editor_property("anim_to_play", unreal.load_asset(anim_path))
        data.set_editor_property("saved_looping", True)
        data.set_editor_property("saved_playing", True)
        comp.set_editor_property("animation_data", data)
        try:
            comp.set_editor_property("update_animation_in_editor", True)
        except Exception:
            pass
    t = spawn(unreal.TextRenderActor, (loc[0], loc[1], loc[2] + 210), (0, 90, 0), label + "_Label")
    tc = t.get_editor_property("text_render")
    tc.set_editor_property("text", label)
    tc.set_editor_property("horizontal_alignment", unreal.HorizTextAligment.EHTA_CENTER)
    tc.set_editor_property("world_size", 24)
    return a


@step("preview_level")
def preview_level(man, base, sk, anims, char_bp):
    if not man.get("make_level", True):
        return None
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    lvl = f"{base}/Maps/L_{man['name']}_Preview"
    ensure_dir(f"{base}/Maps")
    if EAL.does_asset_exist(lvl):
        EAL.delete_asset(lvl)
    les.new_level(lvl)
    floor = spawn(unreal.StaticMeshActor, (0, 0, 0), label="Floor")
    floor.static_mesh_component.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Plane"))
    floor.set_actor_scale3d(unreal.Vector(30, 30, 1))
    # key light from the camera side (+Y) so the characters' fronts are lit
    spawn(unreal.DirectionalLight, (0, 0, 500), (-40, -120, 0), "Sun")
    sky = spawn(unreal.SkyLight, (0, 0, 300), label="SkyLight")
    sky.light_component.set_editor_property("real_time_capture", True)
    spawn(unreal.SkyAtmosphere, (0, 0, 0), label="SkyAtmosphere")
    spawn(unreal.ExponentialHeightFog, (0, 0, 0), label="Fog")
    # Blender front (-Y) arrives as UE +Y (same as the Mannequin), so line the
    # characters up along X and look at them from +Y
    PREVIEW["all_actors"] = [anim_actor(sk, None, (-250, 0, 0), "RefPose")]
    PREVIEW["anim_actors"] = [(anim_actor(sk, (anims or {}).get("Idle"), (0, 0, 0), "Idle"), 0.8),
                              (anim_actor(sk, (anims or {}).get("Walk"), (250, 0, 0), "Walk"), 0.35)]
    if char_bp:
        gm = game_mode(man, base, char_bp)
        ws = None
        try:
            ws = unreal.EditorLevelLibrary.get_editor_world().get_world_settings()
            ws.set_editor_property("default_game_mode",
                                   unreal.BlueprintEditorLibrary.generated_class(unreal.load_asset(gm)))
        except Exception as e:
            ws = None
            warn(f"GameMode override not set ({e}); placing a possessed character instead")
        if ws is None or not gm:
            gen = unreal.BlueprintEditorLibrary.generated_class(unreal.load_asset(char_bp))
            pc = spawn(gen, (0, -400, 100), label="PlayableCharacter")
            pc.set_editor_property("auto_possess_player", unreal.AutoReceiveInput.PLAYER0)
            PREVIEW["hidden"] = [pc]  # keep it out of the screenshot
        # Play spawns the character here, facing the Ref / Idle / Walk line-up
        spawn(unreal.PlayerStart, (0, -500, 100), (0, 90, 0), "PlayerStart")
    PREVIEW["camera"] = spawn(unreal.CameraActor, (0, 700, 130), (-4, -90, 0), "PreviewCamera")
    les.save_current_level()
    REPORT["assets"]["preview_level"] = lvl
    return lvl


# --------------------------------------------------------------------------
@step("save_all")
def save_everything(base):
    """Imported Skeleton / PhysicsAsset and retarget outputs are created in
    memory only - without this they never reach disk and the mesh breaks the
    next time the project is opened."""
    unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
    # remove redirectors left behind by moving retarget outputs out of /Game
    reds = find_assets({"ObjectRedirector"}, "/Game", rescan=True)
    if reds:
        objs = [unreal.load_asset(r) for r in reds]
        objs = [o for o in objs if o is not None]
        if objs:
            AT.fixup_referencers(objs)
        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
    missing = [p for p in find_assets({"Skeleton", "PhysicsAsset"}, base, rescan=True)]
    REPORT["saved_skeleton_physics"] = missing


def main():
    man_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AICHAR_MANIFEST")
    with open(man_path, "r", encoding="utf-8") as f:
        man = json.load(f)
    base = f"{man.get('content_root', '/Game/AIChar')}/{man['name']}"
    if EAL.does_directory_exist(base):  # re-run = rebuild from scratch (idempotent)
        try:
            EAL.delete_directory(base)
        except Exception as e:
            warn(f"could not clear {base} ({e}); existing assets will be overwritten")
    ensure_dir(base)
    REPORT["content_base"] = base
    textures = import_textures(man, base)
    sk = import_skeletal_mesh(man, base)
    if sk is not None:
        materials(man, base, sk, textures)
        tgt_rig = target_ik_rig(man, base, sk)
        src = source_animations(man, base)
        anims = abp = char_bp = None
        if src and tgt_rig:
            rtg = retargeter(man, base, src, sk, tgt_rig)
            if rtg:
                anims = retarget_animations(man, base, src, sk, rtg)
                abp = anim_blueprint(man, base, src, sk, rtg)
                bake_ik_bones(man, base, sk)
                char_bp = character_blueprint(man, base, sk, abp) if man.get("make_character_bp", True) else None
        preview_level(man, base, sk, anims, char_bp)
        preview_screenshot(man)
    save_everything(base)
    must = ["import_skeletal_mesh", "materials", "target_ik_rig", "retarget_animations"]
    REPORT["ok"] = all(REPORT["steps"].get(s, {}).get("ok") for s in must)
    out = man.get("report") or os.path.join(os.path.dirname(man_path), "ue_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(REPORT, f, indent=2, ensure_ascii=False, default=str)
    log("finished, ok =", REPORT["ok"], "report:", out)


def _entry():
    try:
        main()
    except Exception as e:  # never leave the pipeline without a report
        REPORT["steps"]["main"] = {"ok": False, "error": str(e), "trace": traceback.format_exc()[-2000:]}
        man_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AICHAR_MANIFEST")
        try:
            with open(man_path, "r", encoding="utf-8") as f:
                out = json.load(f).get("report")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(REPORT, f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass
        unreal.log_error(f"[AICharPipeline] fatal: {e}")
    finally:
        if os.environ.get("AICHAR_QUIT") == "1":  # headless editor mode
            unreal.SystemLibrary.quit_editor()


def capture_preview(out_png, width=1600, height=900):
    """Render the preview level through a SceneCapture2D into a render target and
    export it as PNG - synchronous, so it works in the headless editor (which
    exits a few frames after -ExecutePythonScript finishes)."""
    cam = PREVIEW.get("camera")
    if cam is None:
        return None
    world = unreal.EditorLevelLibrary.get_editor_world()
    # advance Idle / Walk so the picture shows a real pose, not frame 0
    for a, t in PREVIEW.get("anim_actors", []):
        comp = a.get_editor_property("skeletal_mesh_component")
        for fn in (lambda: comp.set_position(t, False), lambda: comp.tick_animation(t, False),
                   lambda: comp.refresh_bone_transforms()):
            try:
                fn()
            except Exception:
                pass
    for a, _t in [(x, 0) for x in PREVIEW.get("all_actors", [])] + PREVIEW.get("anim_actors", []):
        comp = a.get_editor_property("skeletal_mesh_component")
        try:
            pts = {b: comp.get_socket_location(b) for b in ("root", "pelvis", "head", "foot_l", "hand_r", "hand_l", "lowerarm_l", "upperarm_l", "clavicle_l")}
            REPORT.setdefault("preview_bones", {})[a.get_actor_label()] = {
                k: [round(v.x), round(v.y), round(v.z)] for k, v in pts.items()}
        except Exception as e:
            REPORT.setdefault("preview_bones", {})[a.get_actor_label()] = str(e)
    # Freshly imported textures only have their low mips resident in the
    # headless editor (streaming never ticks) -> blurry screenshot. Force the
    # full mip chain of this character's textures before capturing.
    forced = []
    for tp in find_assets({"Texture2D"}, REPORT.get("content_base", "/Game/AIChar"), rescan=False):
        try:
            tex = unreal.load_asset(tp)
            tex.set_force_mip_levels_to_be_resident(30.0, 0)
            if not tex.get_editor_property("never_stream"):
                tex.set_editor_property("never_stream", True)  # restored after capture
                forced.append(tex)
        except Exception as e:
            REPORT.setdefault("preview_notes", []).append(f"{tp}: {e}")
    try:
        unreal.AssetCompilingManager.get().finish_all_compilation()  # may not be exposed
    except Exception:
        pass
    rt = unreal.RenderingLibrary.create_render_target2d(
        world, width, height, unreal.TextureRenderTargetFormat.RTF_RGBA8,  # LDR output is already gamma-encoded; an sRGB target would encode twice
        unreal.LinearColor(0, 0, 0, 1))
    cap = spawn(unreal.SceneCapture2D, (0, 0, 0), label="PreviewCapture")
    cap.set_actor_transform(cam.get_actor_transform(), False, False)
    cc = cap.get_editor_property("capture_component2d")
    cc.set_editor_property("texture_target", rt)
    cc.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    cc.set_editor_property("fov_angle", 60.0)
    for h in PREVIEW.get("hidden", []):
        for fn in (lambda: cc.hide_actor_components(h, True), lambda: cc.hide_actor_components(h),
                   lambda: h.set_is_temporarily_hidden_in_editor(True)):
            try:
                fn()
                break
            except Exception as e:
                REPORT.setdefault("preview_notes", []).append(f"hide: {e}")
    cc.capture_scene()  # first pass requests the mips ...
    cc.capture_scene()  # ... second pass renders with them
    for tex in forced:
        try:
            tex.set_editor_property("never_stream", False)
        except Exception:
            pass
    unreal.RenderingLibrary.export_render_target(world, rt, os.path.dirname(out_png),
                                                 os.path.basename(out_png))
    cap.destroy_actor()
    return out_png if os.path.exists(out_png) else None


@step("preview_screenshot")
def preview_screenshot(man):
    out = os.path.join(os.path.dirname(man.get("report")), "ue_preview.png")
    if os.path.exists(out):
        os.remove(out)
    shot = capture_preview(out)
    REPORT["assets"]["preview_screenshot"] = shot
    if not shot:
        warn("preview screenshot was not written")
    return shot


_entry()
