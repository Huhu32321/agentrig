"""All pipeline steps + their automatic validators.

Each step:  fn(run) -> report dict  (raises on hard failure)
Each validator: fn(run, report) -> list of issues, every issue carrying
machine-readable fix suggestions. Both the rule-based autopilot and the LLM
agent use the same issues, so decisions are explainable and logged.
"""
import os
import time

from . import gen3d, llm, pose2d, preprocess, ue_runner
from .core import cfg_get, log, read_json, run_blender, write_json

STEP_ORDER = ["preprocess", "analyze", "generate3d", "cleanup", "render_views", "pose2d",
              "orientation_check", "autorig", "ue_build", "qa_report"]

STEP_DOC = {
    "preprocess": "background removal + split multi-view sheet + square crop",
    "analyze": "VLM reads the concept: which view is the front, garments/hair/accessories, skirt or wide trousers",
    "generate3d": "image-to-3D (Hunyuan3D / Tripo / manual)",
    "cleanup": "Blender: merge / orient / scale / ground / decimate / export textures",
    "render_views": "Blender: orthographic front / side renders",
    "pose2d": "MediaPipe 2D pose estimation",
    "orientation_check": "is the model facing away from the camera? (pose + VLM)",
    "autorig": "Blender: joints + skeleton + skinning + stress-pose renders + FBX",
    "ue_build": "UE5.7: import / materials / IK Rig / retarget Idle+Walk / preview level",
    "qa_report": "summary report + contact sheet",
}


# ==========================================================================
# steps
# ==========================================================================
def s_preprocess(run):
    return preprocess.run(run.state["input_image"], run.d("01_preprocess"),
                          size=cfg_get(run.cfg, "preprocess.size", 1024),
                          use_rembg=cfg_get(run.cfg, "preprocess.remove_background", True),
                          model=cfg_get(run.cfg, "preprocess.rembg_model", "isnet-anime"))


ANALYZE_PROMPT = """You are a technical artist for game characters. The images are {n} views
(indexed from 0) cut from the same character concept sheet. Output JSON only:
{{
 "front_view_index": int,     // which view shows the front (face visible)
 "back_view_index": int or null,
 "character_summary": "one sentence",
 "style": "e.g. anime / western cartoon / realistic / chibi",
 "head_to_body_ratio": number,  // body height in heads, e.g. 5.5
 "pose": "A-pose|T-pose|other",
 "garments": ["..."],
 "loose_lower_garment": true/false,   // skirt, robe hem, wide trousers: cloth spanning the legs
 "long_hair_or_ponytail": true/false,
 "accessories": ["earrings", ...],
 "metallic_parts": ["..."],
 "rigging_risks": ["things likely to break automatic rigging / skinning"]
}}"""


def s_analyze(run):
    views = run.report("preprocess")["views"]
    white = [v.replace(".png", "_white.png") for v in views]
    fv = cfg_get(run.cfg, "preprocess.front_view", "auto")
    try:
        info = llm.vision_json(run.cfg, ANALYZE_PROMPT.format(n=len(white)), white)
        info["source"] = "vlm"
    except llm.LLMUnavailable as e:
        log("analyze: VLM unavailable -", e)
        info = {"front_view_index": 0, "back_view_index": 1 if len(views) > 1 else None,
                "source": "default", "note": str(e)}
    if fv != "auto":
        info["front_view_index"] = {"left": 0, "right": len(views) - 1}.get(fv, int(fv))
        info["source"] += "+config"
    fi = max(0, min(int(info.get("front_view_index", 0)), len(views) - 1))
    bi = info.get("back_view_index")
    info["front_image"] = views[fi]
    info["back_image"] = views[bi] if isinstance(bi, int) and 0 <= bi < len(views) and bi != fi else None
    return info


def s_generate3d(run):
    a = run.report("analyze")
    return gen3d.run(run.cfg, a["front_image"], a.get("back_image"), run.d("02_gen3d"))


def s_cleanup(run):
    out = run.p("03_cleanup", "clean.blend")
    rep = run.p("03_cleanup", "report.json")
    c = run.cfg
    secs = run_blender(c, "bl_cleanup.py", [
        "--in", run.report("generate3d")["model"], "--out", out, "--report", rep,
        "--texdir", run.d("03_cleanup", "textures"),
        "--height", cfg_get(c, "cleanup.target_height_m", 1.75),
        "--max-faces", cfg_get(c, "cleanup.max_tris", 50000),
        "--merge-dist", cfg_get(c, "cleanup.merge_distance", 1e-5),
        f"--front={cfg_get(c, 'cleanup.front_axis', '-Y')}"], run.p("logs", "cleanup.log"))
    r = read_json(rep)
    r["blender_seconds"] = round(secs, 1)
    return r


def s_render_views(run):
    outdir = run.d("04_views")
    run_blender(run.cfg, "bl_render_views.py", [
        "--in", run.report("cleanup")["output_blend"], "--outdir", outdir,
        "--res", cfg_get(run.cfg, "rig.render_res", 768),
        "--engine", cfg_get(run.cfg, "rig.render_engine", "CYCLES"),
        "--samples", cfg_get(run.cfg, "rig.render_samples", 16)], run.p("logs", "views.log"))
    return read_json(os.path.join(outdir, "views.json"))


def s_pose2d(run):
    if cfg_get(run.cfg, "rig.joint_method") == "geometric":
        return {"ok": False, "reason": "skipped (joint_method=geometric)"}
    return pose2d.run(run.report("render_views")["front"]["image"], run.p("04_views", "keypoints.json"))


ORIENT_PROMPT = """Image 1 is the front view of a character concept, image 2 an orthographic
render of the generated 3D model. In image 2, does the model face the camera, face away from it,
or show its side? Output JSON only:
{"facing": "front|back|left|right", "confidence": 0-1, "shape_issues": ["obvious shape differences from the concept"]}"""


def s_orientation_check(run):
    views = run.report("render_views")
    kp = run.report("pose2d") or {}
    res = {"pose2d_mirrored_suspect": kp.get("mirrored_suspect"),
           "pose2d_face_visibility": kp.get("face_visibility")}
    try:
        v = llm.vision_json(run.cfg, ORIENT_PROMPT,
                            [run.report("analyze")["front_image"].replace(".png", "_white.png"),
                             views["front"]["image"]])
        res.update({"vlm": v, "source": "vlm"})
        res["facing"] = v.get("facing", "front")
    except llm.LLMUnavailable:
        res["source"] = "pose2d"
        if kp.get("ok") and kp.get("mirrored_suspect") and (kp.get("face_visibility") or 1) < 0.3:
            res["facing"] = "back"
        else:
            res["facing"] = "front" if kp.get("ok") else "unknown"
    return res


def s_autorig(run):
    outdir = run.d("05_rig")
    rep = run.p("05_rig", "report.json")
    c = run.cfg
    cloth = cfg_get(c, "rig.cloth_smoothing", "auto")
    a = run.report("analyze") or {}
    if cloth == "auto" and a.get("loose_lower_garment") is True:
        cloth = "on"  # VLM saw a skirt / robe / hakama
    kp = run.report("pose2d") or {}
    args = ["--in", run.report("cleanup")["output_blend"], "--outdir", outdir, "--report", rep,
            "--name", run.name, "--joint-method", cfg_get(c, "rig.joint_method", "auto"),
            "--weight-method", cfg_get(c, "rig.weight_method", "auto"),
            "--cloth-smoothing", cloth,
            "--vertical-reconcile", int(bool(cfg_get(c, "rig.vertical_reconcile", True))),
            "--skin-post", int(bool(cfg_get(c, "rig.skin_postprocess", True))),
            "--arm-anchor", int(bool(cfg_get(c, "rig.arm_anchor", True))),
            "--render-tests", int(bool(cfg_get(c, "rig.render_tests", True))),
            "--render-res", cfg_get(c, "rig.render_res", 768),
            "--render-samples", cfg_get(c, "rig.render_samples", 16),
            "--render-engine", cfg_get(c, "rig.render_engine", "CYCLES")]
    if kp.get("ok"):
        args += ["--keypoints", run.p("04_views", "keypoints.json"),
                 "--views", os.path.join(os.path.dirname(run.report("render_views")["front"]["image"]),
                                         "views.json")]
    secs = run_blender(c, "bl_autorig.py", args, run.p("logs", "autorig.log"))
    r = read_json(rep)
    r["blender_seconds"] = round(secs, 1)
    _draw_bone_overlay(r.get("qa_renders", {}).get("rest_front"))
    return r


def _draw_bone_overlay(front_png):
    """Blender's Python has no PIL - draw the skeleton QA image here from bones.json."""
    if not front_png:
        return
    js = front_png.replace(".png", "_bones.json")
    if not os.path.exists(js):
        return
    from PIL import Image, ImageDraw
    data = read_json(js)
    im = Image.open(front_png).convert("RGB")
    d = ImageDraw.Draw(im)
    for s in data["bones"]:
        col = (255, 60, 60) if s["deform"] else (90, 160, 255)
        d.line([tuple(s["head"]), tuple(s["tail"])], fill=col, width=3 if s["deform"] else 1)
        x, y = s["head"]
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 220, 0))
    im.save(data["out"])


def s_ue_build(run):
    c = run.cfg
    cl = run.report("cleanup")
    rig = run.report("autorig")
    man = {
        "name": run.name, "fbx": rig["fbx"], "content_root": cfg_get(c, "ue.content_root"),
        "materials": cl.get("textures", {}), "anim_source": cfg_get(c, "ue.anim_source"),
        "make_level": cfg_get(c, "ue.make_level", True),
        "retarget_anim_blueprint": cfg_get(c, "ue.retarget_anim_blueprint", True),
        "make_character_bp": cfg_get(c, "ue.make_character_bp", True),
        "report": run.p("06_ue", "ue_report.json"),
    }
    mpath = run.p("06_ue", "manifest.json")
    write_json(mpath, man)
    if os.path.exists(man["report"]):
        os.remove(man["report"])
    secs = ue_runner.run(c, mpath, run.p("logs", "ue.log"))
    if not os.path.exists(man["report"]):
        raise RuntimeError("UE finished without writing ue_report.json - see logs/ue.log")
    r = read_json(man["report"])
    r["ue_seconds"] = round(secs, 1)
    level = (r.get("assets") or {}).get("preview_level")
    if cfg_get(c, "ue.play_test", True) and level and (r.get("assets") or {}).get("game_mode"):
        try:
            pt = ue_runner.play_test(c, level.split(".")[0], run.p("06_ue", "play_test.json"),
                                     run.p("logs", "ue_play_test.log"))
            r["play_test"] = {k: pt.get(k) for k in ("idle", "walk", "error")}
        except Exception as e:  # the build itself succeeded; report, don't fail
            r["play_test"] = {"error": str(e)[-500:]}
    return r


def s_qa_report(run):
    from . import qa
    return qa.build(run)


STEPS = {
    "preprocess": s_preprocess, "analyze": s_analyze, "generate3d": s_generate3d,
    "cleanup": s_cleanup, "render_views": s_render_views, "pose2d": s_pose2d,
    "orientation_check": s_orientation_check, "autorig": s_autorig, "ue_build": s_ue_build,
    "qa_report": s_qa_report,
}


# ==========================================================================
# validators -> issues with fix suggestions
# ==========================================================================
def issue(msg, severity="warn", fixes=None):
    return {"issue": msg, "severity": severity, "fixes": fixes or []}


def fix(overrides, rerun_from, why):
    return {"overrides": overrides, "rerun_from": rerun_from, "why": why}


def v_preprocess(run, r):
    if len(r["views"]) == 1:
        return [issue("only 1 view found (no back view) - multi-view generation unavailable", "info")]
    if len(r["views"]) > 4:
        return [issue(f"split into {len(r['views'])} views - accessories or text may have become separate views", "warn")]
    return []


def v_cleanup(run, r):
    out = []
    if r["tris"] < 3000:
        out.append(issue(f"very low triangle count ({r['tris']}) - generation probably failed", "error"))
    if r.get("non_manifold_edges", 0) > 0.02 * r["faces"]:
        out.append(issue("many non-manifold edges - bone heat may fail", "warn",
                         [fix({"rig.weight_method": "proxy_heat"}, "autorig", "a voxel proxy mesh is more robust")]))
    if not r.get("textures"):
        out.append(issue("no textures extracted (model may be untextured)", "warn"))
    return out


def v_orientation(run, r):
    if r.get("facing") == "back":
        cur = cfg_get(run.cfg, "cleanup.front_axis", "-Y")
        new = "+Y" if cur == "-Y" else "-Y"
        return [issue("model faces away from the camera", "error",
                      [fix({"cleanup.front_axis": new}, "cleanup", "rotate 180 deg so the front faces -Y")])]
    if r.get("facing") in ("left", "right"):
        new = "+X" if r["facing"] == "left" else "-X"
        return [issue(f"model shows its side to the camera ({r['facing']})", "error",
                      [fix({"cleanup.front_axis": new}, "cleanup", "rotate 90 deg to fix the orientation")])]
    return []


def v_autorig(run, r):
    out = []
    sk = r["skinning"]
    if sk["zero_weight_ratio_final"] > 0.001:
        out.append(issue("vertices without weights", "error",
                         [fix({"rig.weight_method": "proxy_heat"}, "autorig", "skin through a proxy mesh")]))
    empty = [b for b, n in sk["verts_per_bone"].items() if n == 0
             and b.startswith(("upperarm", "lowerarm", "thigh", "calf", "head", "spine_01", "spine_02", "spine_03"))]
    if empty:
        out.append(issue(f"key bones influence no vertices: {empty}", "error",
                         [fix({"rig.joint_method": "geometric"}, "autorig", "2D keypoints may be off - use geometric estimation"),
                          fix({"rig.weight_method": "proxy_heat"}, "autorig", "switch skinning method")]))
    if r.get("sanity_fixes"):
        out.append(issue(f"joint sanity fixes: {r['sanity_fixes']}", "info"))
    # skeleton structure (catches e.g. a chibi rig whose pelvis landed in the chest
    # and neck inside the head - every skinning metric was still "healthy")
    J, H = r.get("joints") or {}, r.get("height_m") or 1.75
    if all(k in J for k in ("pelvis", "upperarm_l", "neck_01", "head")):
        pel, sh, neck, head = J["pelvis"][2], J["upperarm_l"][2], J["neck_01"][2], J["head"][2]
        bad = []
        if sh - pel < 0.15 * H:
            bad.append(f"pelvis ({pel:.2f} m) too close to shoulders ({sh:.2f} m)")
        if not (sh - 0.03 * H < neck < head):
            bad.append(f"neck ({neck:.2f} m) not between shoulders and head")
        if head - sh > 0.25 * H:
            bad.append(f"head bone start ({head:.2f} m) too far above shoulders")
        if bad:
            out.append(issue("implausible skeleton structure: " + "; ".join(bad), "error",
                             [fix({"rig.joint_method": "pose2d"}, "autorig", "force 2D keypoints"),
                              fix({"rig.joint_method": "geometric"}, "autorig", "use pure geometric estimation")]))
    g = r.get("geometry_analysis", {})
    if g.get("loose_lower_garment") and not sk.get("cloth_smoothing"):
        out.append(issue("skirt / wide trousers detected but cloth weight smoothing is off", "warn",
                         [fix({"rig.cloth_smoothing": "on"}, "autorig", "smooth cloth weights between the legs to avoid tearing")]))
    return out


def v_ue(run, r):
    out = []
    for name, st in r.get("steps", {}).items():
        if not st.get("ok"):
            out.append(issue(f"UE step failed: {name}: {st.get('error')}", "error"))
    # retarget sanity: in the preview level the animated characters must keep a
    # plausible head height compared with the reference pose (catches collapsed
    # or flying animations caused by unit / bone-scale problems)
    bones = r.get("preview_bones") or {}
    ref = bones.get("RefPose")
    if isinstance(ref, dict) and ref.get("head") and ref.get("root"):
        ref_h = ref["head"][2] - ref["root"][2]
        for name in ("Idle", "Walk"):
            b = bones.get(name)
            if not isinstance(b, dict) or not b.get("head"):
                continue
            ratio = (b["head"][2] - b["root"][2]) / max(ref_h, 1e-3)
            if not 0.75 < ratio < 1.25:
                out.append(issue(f"{name} retarget looks wrong: head height is {ratio:.2f}x the reference pose"
                                 " (bone units / scale, or wrong retarget chain mapping)", "error"))
    # runtime check (Play in Editor): the pawn is driven forward; its feet must actually step
    pt = r.get("play_test") or {}
    walk = pt.get("walk")
    if walk and walk.get("speed_mean", 0) > 100:
        if walk["stride_max"] < 20:
            out.append(issue(f"in Play the feet do not step (max fore-aft foot separation "
                             f"{walk['stride_max']:.1f} cm while moving at {walk['speed_mean']:.0f} cm/s) - "
                             "the anim blueprint's foot IK is pinning them (ik_foot bones)", "error"))
    elif pt.get("error"):
        out.append(issue(f"play test could not run: {pt['error']}", "warn"))
    for w in r.get("warnings", []):
        out.append(issue(f"UE: {w}", "warn"))
    return out


VALIDATORS = {"preprocess": v_preprocess, "cleanup": v_cleanup,
              "orientation_check": v_orientation, "autorig": v_autorig, "ue_build": v_ue}


def execute(run, name, actor="pipeline"):
    """Run one step, store its report + issues in the run state."""
    log(f"===== {name}: {STEP_DOC[name]}")
    run.set_step(name, "running", actor=actor)
    t0 = time.time()
    try:
        rep = STEPS[name](run)
    except Exception as e:
        run.set_step(name, "failed", error=str(e)[-2500:], seconds=round(time.time() - t0, 1),
                     actor=actor)
        raise
    issues = VALIDATORS.get(name, lambda *_: [])(run, rep)
    rep = dict(rep)
    rep["issues"] = issues
    run.set_step(name, "done", report=rep, seconds=round(time.time() - t0, 1), actor=actor)
    for i in issues:
        log(f"  [{i['severity']}] {i['issue']}")
    return rep
