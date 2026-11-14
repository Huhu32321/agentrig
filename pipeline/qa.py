"""Step 10 - QA report: markdown summary + contact sheet image."""
import os

from PIL import Image, ImageDraw, ImageFont

from .core import log
from .steps import STEP_DOC, STEP_ORDER


def _font(size):
    for f in ("arial.ttf", "DejaVuSans.ttf", "msyh.ttc"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()


def contact_sheet(items, out, cell=360):
    items = [(t, p) for t, p in items if p and os.path.exists(p)]
    if not items:
        return None
    cols = min(4, len(items))
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 30)), "white")
    d = ImageDraw.Draw(sheet)
    f = _font(18)
    for i, (title, p) in enumerate(items):
        im = Image.open(p).convert("RGBA")
        bg = Image.new("RGBA", im.size, "white")
        im = Image.alpha_composite(bg, im).convert("RGB")
        im.thumbnail((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + 30)
        sheet.paste(im, (x + (cell - im.width) // 2, y + 30))
        d.text((x + 8, y + 5), title, fill=(30, 30, 30), font=f)
    sheet.save(out)
    return out


def build(run):
    st = run.state["steps"]
    rep = lambda s: (st.get(s) or {}).get("report") or {}  # noqa: E731
    a, rig, ue = rep("analyze"), rep("autorig"), rep("ue_build")
    qa_r = rig.get("qa_renders", {})
    sheet = contact_sheet([
        ("Concept (front)", a.get("front_image")),
        ("Concept (back)", a.get("back_image")),
        ("3D front render", (rep("render_views").get("front") or {}).get("image")),
        ("3D side render", (rep("render_views").get("side") or {}).get("image")),
        ("Skeleton", qa_r.get("rest_front_bones")),
        ("Stress pose front", qa_r.get("stress_front")),
        ("Stress pose side", qa_r.get("stress_side")),
        ("UE5 preview (Ref / Idle / Walk)", (ue.get("assets") or {}).get("preview_screenshot")),
    ], run.p("07_qa", "contact_sheet.png"))

    L = [f"# Character pipeline report: {run.name}", "",
         f"- Input: `{run.state.get('input_image')}`",
         f"- Started: {run.state.get('created')}", ""]
    L += ["## Steps", "", "| Step | What it does | Status | Time (s) | Actor | Issues |",
          "|---|---|---|---|---|---|"]
    for s in STEP_ORDER:
        if s == "qa_report":
            continue
        r = st.get(s, {})
        iss = "; ".join(i["issue"] for i in (r.get("report") or {}).get("issues", [])) or ""
        actor = (r.get("history") or [{}])[-1].get("actor", "")
        L.append(f"| {s} | {STEP_DOC[s]} | {r.get('status', '-')} | {r.get('seconds', '')} | "
                 f"{actor} | {iss[:120]} |")
    if a:
        L += ["", "## Concept analysis (VLM)", "", "```json",
              str({k: a.get(k) for k in ("style", "pose", "garments", "loose_lower_garment",
                                         "long_hair_or_ponytail", "accessories", "rigging_risks")}),
              "```"]
    cl = rep("cleanup")
    if cl:
        L += ["", "## Mesh", "",
              f"- {cl.get('tris')} triangles, {cl.get('verts')} vertices, {cl.get('islands')} islands, "
              f"{cl.get('non_manifold_edges')} non-manifold edges, height {cl.get('height_m', 0):.2f} m",
              f"- Materials: {cl.get('materials')}"]
    if rig:
        sk = rig.get("skinning", {})
        L += ["", "## Rig / skinning", "",
              f"- Joint estimation: {rig.get('joint_method')} "
              f"({len(rig.get('pose2d_joints_used', []))} joints from 2D keypoints)",
              f"- Bones: {rig.get('bone_count')} ({rig.get('deform_bone_count')} deforming), UE Mannequin naming",
              f"- Skinning: {sk.get('method_used')}, cloth smoothing {sk.get('cloth_smoothing')}, "
              f"orphan vertices fixed {sk.get('orphan_vertices_fixed')}, max influences {sk.get('max_influences')}",
              f"- Geometry notes: {rig.get('geometry_analysis', {}).get('notes')}",
              f"- Sanity fixes: {rig.get('sanity_fixes')}",
              f"- FBX: `{rig.get('fbx')}`"]
    if ue:
        L += ["", "## UE 5.7", "",
              f"- Result: {'success' if ue.get('ok') else 'some sub-steps failed'}",
              f"- Mesh: {ue.get('mesh')}"]
        for k, v in (ue.get("assets") or {}).items():
            L.append(f"- {k}: `{v}`")
        if ue.get("anim_source"):
            L.append(f"- Animation source: {ue['anim_source']}")
        for w in ue.get("warnings", []):
            L.append(f"- WARNING: {w}")
        for i in ue.get("issues", []):
            L.append(f"- ISSUE: {i['issue']}")
        shot = (ue.get("assets") or {}).get("preview_screenshot")
        if shot:
            rel = os.path.relpath(shot, run.p("07_qa")).replace(os.sep, "/")
            L += ["", f"![UE preview]({rel})"]
    if run.state.get("decisions"):
        L += ["", "## Decision log (agent / rules)", ""]
        for d in run.state["decisions"]:
            L.append(f"- {d['time']} **{d['by']}**: {d['what']} - {d['why']}")
    if run.state.get("agent_summary"):
        L += ["", "## Agent summary", "", run.state["agent_summary"]]
    if sheet:
        L += ["", "## Contact sheet", "", "![contact sheet](contact_sheet.png)"]
    md = run.p("07_qa", "REPORT.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    ok = bool(rig) and (not ue or ue.get("ok"))
    log("QA report:", md)
    return {"report_md": md, "contact_sheet": sheet, "verdict": "pass" if ok else "needs_review"}
