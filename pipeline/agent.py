"""Orchestration: rule-based Autopilot and the LLM Agent.

Both drive the same steps and read the same validator issues. The Autopilot
is deterministic (good for batch / CI); the Agent is an LLM with tool calling
that reads reports, can look at QA renders through the VLM, changes
parameters and re-runs steps, and writes its reasoning into the run log.
"""
import json

from . import llm
from .core import cfg_get, log
from .steps import STEP_DOC, STEP_ORDER, execute

NONFATAL = {"analyze", "pose2d", "orientation_check"}


def summarize(step, rep):
    """Short, token-friendly view of a step report for the LLM."""
    if rep is None:
        return None
    keep = {
        "preprocess": ["views", "bg_method"],
        "analyze": ["front_view_index", "style", "pose", "loose_lower_garment",
                    "long_hair_or_ponytail", "accessories", "rigging_risks", "source"],
        "generate3d": ["backend", "model", "seconds", "size_mb"],
        "cleanup": ["tris", "verts", "islands", "non_manifold_edges", "height_m",
                    "auto_stood_up", "decimate_ratio", "materials"],
        "render_views": ["front", "side"],
        "pose2d": ["ok", "reason", "mean_body_visibility", "face_visibility", "mirrored_suspect"],
        "orientation_check": ["facing", "source", "vlm"],
        "autorig": ["joint_method", "pose2d_joints_used", "sanity_fixes", "bone_count",
                    "deform_bone_count", "qa_renders", "fbx"],
        "ue_build": ["ok", "steps", "warnings", "assets", "mesh", "play_test"],
        "qa_report": ["report_md", "contact_sheet", "verdict"],
    }.get(step, [])
    out = {k: rep.get(k) for k in keep if k in rep}
    if step == "autorig":
        sk = rep.get("skinning", {})
        out["skinning"] = {k: sk.get(k) for k in ("method_used", "cloth_smoothing",
                                                  "zero_weight_ratio_final", "orphan_vertices_fixed")}
        out["geometry"] = {k: rep.get("geometry_analysis", {}).get(k)
                           for k in ("t_pose", "loose_lower_garment", "notes")}
    if step == "ue_build" and "steps" in out:
        out["steps"] = {k: v.get("ok") for k, v in out["steps"].items()}
    out["issues"] = rep.get("issues", [])
    return out


# ==========================================================================
# rule-based autopilot
# ==========================================================================
def autopilot(run, start="preprocess", stop_after=None, max_retries=2):
    i = STEP_ORDER.index(start)
    tried, retries = set(), {}
    while i < len(STEP_ORDER):
        name = STEP_ORDER[i]
        try:
            rep = execute(run, name, actor="autopilot")
        except Exception as e:
            if name in NONFATAL:
                log(f"{name} failed but is optional: {e}")
                i += 1
                continue
            log(f"FAILED at {name}: {e}")
            if name != "qa_report":
                try:
                    execute(run, "qa_report", actor="autopilot")
                except Exception as qe:
                    log("qa_report failed:", qe)
            return False
        jumped = False
        for iss in rep.get("issues", []):
            for f in iss.get("fixes", []):
                key = json.dumps(f["overrides"], sort_keys=True)
                if key in tried:
                    continue
                target = STEP_ORDER.index(f["rerun_from"])
                if iss["severity"] == "error" or target > i:
                    tried.add(key)
                    for k, v in f["overrides"].items():
                        run.override(k, v, "autopilot", f"{iss['issue']} -> {f['why']}")
                    if target <= i:
                        retries[name] = retries.get(name, 0) + 1
                        if retries[name] > max_retries:
                            log(f"{name}: retry limit reached, continuing with best result")
                            break
                        i, jumped = target, True
                    break
            if jumped:
                break
        if jumped:
            continue
        if stop_after == name:
            break
        i += 1
    return True


# ==========================================================================
# LLM agent
# ==========================================================================
SYSTEM = """You are a technical-art agent. Your job is to turn one humanoid character concept
image into a game-ready UE5.7 character (textured Skeletal Mesh + skeleton + skin weights,
playing Idle/Walk) by driving an automatic pipeline.

Steps (in order):
{steps}

Rules:
1. Call run_step for each step in order; read the returned report and issues after every step.
2. When there are issues, decide yourself: change parameters with set_config, then re-run from the
   suggested step. At most 2 retries per step.
3. After autorig, call inspect_image on the stress_front / stress_side renders and look for
   interpenetration, tearing or misplaced bones; if there are clear problems, try
   rig.joint_method / rig.weight_method / rig.cloth_smoothing.
4. After ue_build, call inspect_image on report.assets.preview_screenshot (UE preview level:
   reference pose / Idle / Walk) and check textures, natural Idle/Walk poses and that the
   character faces the camera; if issues mention a retarget problem, check cleanup.front_axis
   and the skeleton first.
5. analyze / pose2d / orientation_check may fail and be skipped (geometric fallbacks exist).
   On a hard failure of generate3d or ue_build, explain why and call finish; do not retry forever.
6. Always run qa_report last, then call finish with a short English summary: result, key
   decisions, and anything that still needs a human.
7. Every set_config must state its evidence in `reason` (it is written to the decision log)."""

TOOLS = [
    {"type": "function", "function": {
        "name": "run_step", "description": "Run one pipeline step; returns a summary report and its issues",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "enum": STEP_ORDER}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "set_config", "description": "Change a config parameter (dotted path, e.g. rig.weight_method)",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string"}, "value": {}, "reason": {"type": "string"}},
            "required": ["key", "value", "reason"]}}},
    {"type": "function", "function": {
        "name": "get_report", "description": "Read the full report of a finished step (JSON, may be long)",
        "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": STEP_ORDER}},
                       "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "inspect_image", "description": "Ask the vision model a question about an image (e.g. a QA render)",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "question": {"type": "string"}},
            "required": ["path", "question"]}}},
    {"type": "function", "function": {
        "name": "finish", "description": "End the run with a summary",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},
                       "required": ["summary"]}}},
]

TUNABLE = {"cleanup.front_axis", "cleanup.target_height_m", "cleanup.max_tris", "rig.joint_method",
           "rig.weight_method", "rig.cloth_smoothing", "rig.vertical_reconcile", "gen3d.backend", "gen3d.use_multiview",
           "gen3d.hunyuan.seed", "gen3d.hunyuan.octree_resolution", "preprocess.front_view",
           "preprocess.remove_background", "ue.anim_source.idle", "ue.anim_source.walk"}

# legal values for enumerated keys (LLM once invented rig.weight_method="geodesic")
ENUMS = {"cleanup.front_axis": ["-Y", "+Y", "-X", "+X"],
         "rig.joint_method": ["auto", "pose2d", "geometric"],
         "rig.weight_method": ["auto", "heat", "proxy_heat"],
         "rig.cloth_smoothing": ["auto", "on", "off"],
         "gen3d.backend": ["hunyuan_local", "tripo", "manual"]}


def _tool(run, name, args, counts):
    if name == "run_step":
        s = args["name"]
        counts[s] = counts.get(s, 0) + 1
        if counts[s] > 3:
            return {"error": f"{s} already ran {counts[s] - 1} times; stop retrying"}
        try:
            rep = execute(run, s, actor="agent")
            return {"status": "done", "report": summarize(s, rep)}
        except Exception as e:
            return {"status": "failed", "error": str(e)[-1500:]}
    if name == "set_config":
        if args["key"] not in TUNABLE:
            return {"error": f"key not tunable; allowed: {sorted(TUNABLE)}"}
        legal = ENUMS.get(args["key"])
        if legal and args["value"] not in legal:
            return {"error": f"illegal value {args['value']!r} for {args['key']}; allowed: {legal}"}
        run.override(args["key"], args["value"], "agent", args.get("reason", ""))
        return {"ok": True}
    if name == "get_report":
        return json.loads(json.dumps(run.report(args["name"]), default=str)[:12000])
    if name == "inspect_image":
        try:
            ans = llm.vision_json(run.cfg, args["question"] + "\nOutput JSON only: {\"answer\":\"...\", "
                                  "\"problems\": [\"...\"], \"severity\": \"none|minor|major\"}",
                                  [args["path"]])
            run.decide("agent-vision", f"inspect {args['path']}", json.dumps(ans, ensure_ascii=False))
            return ans
        except Exception as e:
            return {"error": str(e)}
    return {"error": f"unknown tool {name}"}


def agent(run, max_turns=None):
    max_turns = max_turns or cfg_get(run.cfg, "llm.max_turns", 40)
    steps_txt = "\n".join(f"- {s}: {STEP_DOC[s]}" for s in STEP_ORDER)
    done = {k: v.get("status") for k, v in run.state["steps"].items()}
    system = SYSTEM.format(steps=steps_txt)
    tools = TOOLS
    vlm_ok = bool(cfg_get(run.cfg, "vlm.enabled", False)) and bool(
        llm.env_key(cfg_get(run.cfg, "vlm.api_key_env", "")))
    if not vlm_ok:  # don't offer a tool that can only fail (the LLM kept retrying it)
        tools = [t for t in TOOLS if t["function"]["name"] != "inspect_image"]
        system += ("\n\nNOTE: no vision model is available in this run - inspect_image does not exist "
                   "and you must not try to enable the VLM. For rules 3 and 4 use get_report and the "
                   "quantitative metrics instead (autorig: skinning.verts_per_bone, "
                   "far_weight_vertices_pruned, head_attachment_vertices, geometry_analysis.notes; "
                   "ue_build: head / pelvis heights of Idle and Walk vs RefPose in preview_bones). "
                   "In finish, list the images a human should check (stress_front/stress_side, ue_preview).")
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"Character name {run.name}, input image {run.state.get('input_image')}. "
                                        f"Current step states: {json.dumps(done, ensure_ascii=False)}. Start."}]
    counts = {}
    for turn in range(max_turns):
        msg = llm.chat(run.cfg, msgs, tools=tools)
        msgs.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        if msg.get("content"):
            log("[agent]", msg["content"][:600])
        calls = msg.get("tool_calls") or []
        if not calls:
            msgs.append({"role": "user", "content": "Keep calling tools; call finish when everything is done."})
            continue
        for c in calls:
            fn = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            log(f"[agent] -> {fn}({json.dumps(args, ensure_ascii=False)[:200]})")
            if fn == "finish":
                run.decide("agent", "finish", args.get("summary", ""))
                run.state["agent_summary"] = args.get("summary", "")
                run.save()
                return True
            result = _tool(run, fn, args, counts)
            msgs.append({"role": "tool", "tool_call_id": c["id"],
                         "content": json.dumps(result, ensure_ascii=False, default=str)[:8000]})
    log("agent: max turns reached")
    return False


def orchestrate(run, mode="auto", start="preprocess"):
    if mode == "agent":
        try:
            return agent(run)
        except llm.LLMUnavailable as e:
            log("LLM unavailable -> autopilot:", e)
            run.decide("pipeline", "fallback to autopilot", str(e))
    return autopilot(run, start=start)
