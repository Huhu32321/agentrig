"""
One-click artist UI (Gradio):  python ui/app.py  -> open http://127.0.0.1:7860
Upload a concept -> enter a name -> pick the 3D backend / orchestration mode -> Run.
"""
import os
import subprocess
import sys
import gradio as gr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.core import load_config, cfg_get  # noqa: E402


def run_pipeline(image, name, backend, manual_model, mode, joint, weight, cloth):
    if not image or not name:
        yield "Upload a concept image and enter a character name (ASCII).", None, None
        return
    cmd = [sys.executable, os.path.join(ROOT, "run_pipeline.py"), "--image", image, "--name", name,
           "--mode", mode, "--set", f"gen3d.backend={backend}", "--set", f"rig.joint_method={joint}",
           "--set", f"rig.weight_method={weight}", "--set", f"rig.cloth_smoothing={cloth}"]
    if backend == "manual" and manual_model:
        cmd += ["--set", f"gen3d.manual_model={manual_model}"]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    lines = []
    for line in proc.stdout:
        if "%|" in line:  # progress bars
            continue
        lines.append(line.rstrip())
        yield "\n".join(lines[-400:]), None, None
    proc.wait()
    out = os.path.join(cfg_get(load_config(), "paths.output_root", "output"), name, "07_qa")
    out = os.path.abspath(os.path.join(ROOT, out)) if not os.path.isabs(out) else out
    sheet = os.path.join(out, "contact_sheet.png")
    md = os.path.join(out, "REPORT.md")
    report = open(md, encoding="utf-8").read() if os.path.exists(md) else "(no report was written)"
    yield "\n".join(lines[-400:]), sheet if os.path.exists(sheet) else None, report


with gr.Blocks(title="AgentRig") as demo:
    gr.Markdown("# AgentRig\nConcept image -> 3D -> automatic rig + skin -> UE5.7 "
                "(materials / Idle / Walk / preview level)")
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="filepath", label="Character concept (a front/back sheet is fine)")
            name = gr.Textbox(label="Character name (ASCII, used for UE asset names)", value="Explorer")
            backend = gr.Dropdown(["hunyuan_local", "tripo", "manual"],
                                  value=cfg_get(load_config(), "gen3d.backend", "manual"),
                                  label="Image-to-3D backend")
            manual_model = gr.File(label="manual backend: upload an existing model (glb/obj/fbx)",
                                   file_types=[".glb", ".gltf", ".obj", ".fbx"], type="filepath")
            mode = gr.Radio(["auto", "agent"], value="agent", label="Orchestration: rule autopilot / LLM agent")
            with gr.Accordion("Advanced (usually leave as is)", open=False):
                joint = gr.Dropdown(["auto", "pose2d", "geometric"], value="auto", label="Joint placement")
                weight = gr.Dropdown(["auto", "heat", "proxy_heat"], value="auto", label="Skinning")
                cloth = gr.Dropdown(["auto", "on", "off"], value="auto", label="Skirt / wide-trouser smoothing")
            go = gr.Button("Run", variant="primary")
        with gr.Column(scale=2):
            logs = gr.Textbox(label="Log", lines=24, max_lines=24, autoscroll=True)
            sheet = gr.Image(label="QA contact sheet")
            report = gr.Markdown()
    go.click(run_pipeline, [image, name, backend, manual_model, mode, joint, weight, cloth],
             [logs, sheet, report])

if __name__ == "__main__":
    demo.launch()
