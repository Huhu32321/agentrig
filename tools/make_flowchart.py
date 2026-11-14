"""
Draw the pipeline overview -> docs/pipeline_flow.png (1920x1080, also usable as a video title card).

    python tools/make_flowchart.py
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "docs", "pipeline_flow.png")
S = 2                                   # supersampling factor; final image is 1920x1080
W, H = 1920, 1080
FONTS = [("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
         ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]
FONT, FONT_B = next((r, b) for r, b in FONTS if os.path.exists(r))

BG = (248, 248, 246)
INK = (34, 34, 38)
MUTED = (110, 110, 118)
KIND = {  # executor -> (stripe colour, tag text)
    "io":      ((60, 60, 66), "input / output"),
    "script":  ((120, 128, 140), "Python"),
    "ai":      ((132, 84, 196), "AI model"),
    "manual":  ((214, 64, 58), "manual or API"),
    "blender": ((232, 128, 32), "Blender (headless)"),
    "ue":      ((38, 110, 210), "UE5.7 Python"),
}


def f(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size * S)


def sc(*v):
    return [x * S for x in v]


def box(d, x, y, w, h, kind, title, sub, lines, tag=None):
    col, default_tag = KIND[kind]
    dash = kind == "manual"
    d.rounded_rectangle(sc(x, y, x + w, y + h), radius=14 * S, fill=(255, 255, 255),
                        outline=col, width=(3 if dash else 2) * S)
    if dash:  # white gaps on the border -> dashed look
        for i in range(0, w, 18):
            d.line(sc(x + i + 9, y, x + i + 15, y), fill=(255, 255, 255), width=4 * S)
            d.line(sc(x + i + 9, y + h, x + i + 15, y + h), fill=(255, 255, 255), width=4 * S)
    d.rounded_rectangle(sc(x, y, x + 10, y + h), radius=6 * S, fill=col)
    tx = x + 24
    d.text(sc(tx, y + 14), title, font=f(25, True), fill=INK)
    d.text(sc(tx, y + 52), sub, font=f(16), fill=MUTED)
    yy = y + 86
    for ln in lines:
        d.text(sc(tx, yy), ln, font=f(18), fill=INK)
        yy += 29
    t = tag or default_tag
    tw = d.textlength(t, font=f(15, True)) / S
    d.rounded_rectangle(sc(tx - 2, y + h - 40, tx + tw + 18, y + h - 12), radius=12 * S, fill=col)
    d.text(sc(tx + 8, y + h - 39), t, font=f(15, True), fill=(255, 255, 255))


def arrow(d, pts, col=INK, width=3, head=12, dashed=False):
    pts = [(x * S, y * S) for x, y in pts]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if dashed:
            n = max(2, int(max(abs(x1 - x0), abs(y1 - y0)) / (14 * S)))
            for i in range(0, n, 2):
                a, b = i / n, min(1, (i + 1) / n)
                d.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
                       fill=col, width=width * S)
        else:
            d.line([(x0, y0), (x1, y1)], fill=col, width=width * S)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    a = math.atan2(y1 - y0, x1 - x0)
    hs = head * S
    d.polygon([(x1, y1), (x1 - hs * math.cos(a - 0.45), y1 - hs * math.sin(a - 0.45)),
               (x1 - hs * math.cos(a + 0.45), y1 - hs * math.sin(a + 0.45))], fill=col)


def main():
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)

    d.text(sc(56, 22), "AgentRig: one concept image  \u2192  a playable UE5 character", font=f(38, True), fill=INK)
    d.text(sc(58, 80), "10 steps  \u00b7  every step is checked by validators  \u00b7  an LLM agent changes "
           "parameters and re-runs from the right step  \u00b7  ~80 s after the 3D model exists",
           font=f(19), fill=MUTED)
    lx = 58
    for k in ("script", "ai", "manual", "blender", "ue"):
        col, t = KIND[k]
        d.rounded_rectangle(sc(lx, 122, lx + 22, 144), radius=5 * S, fill=col)
        d.text(sc(lx + 30, 118), t, font=f(17), fill=INK)
        lx += 52 + d.textlength(t, font=f(17)) / S

    bw, gap, x0 = 268, 44, 56
    xs = [x0 + i * (bw + gap) for i in range(6)]
    y1, h1 = 168, 270
    y2, h2 = 500, 300

    row1 = [
        ("io", "Concept image", "input", ["single view or a", "front / back sheet"], "input"),
        ("script", "1  Preprocess", "preprocess", ["background removal", "split views", "square crop"], None),
        ("ai", "2  Analyse", "analyze", ["style, proportions", "skirt / robe / props", "\u2192 rig parameters"],
         "VLM (Qwen-VL)"),
        ("manual", "3  Image \u2192 3D", "generate3d", ["web tool (manual)", "or Tripo API", "or local Hunyuan3D"],
         None),
        ("blender", "4  Clean-up", "cleanup", ["merge, stand up, face -Y", "scale 1.75 m, ground", "decimate, fix maps"],
         None),
        ("blender", "5  Ortho renders", "render_views", ["front / side views", "for pose + QA"], None),
    ]
    row2 = [
        ("ai", "6  2D pose", "pose2d", ["MediaPipe keypoints", "shoulder ... ankle"], "AI pose model"),
        ("ai", "7  Facing check", "orientation_check", ["facing away?", "\u2192 rotate 180\u00b0 and", "re-run clean-up"],
         "VLM + rules"),
        ("blender", "8  Auto-rig + skin", "autorig", ["arm-anchored joints", "+ 2D keypoints, arbitrated",
                                                   "UE Mannequin skeleton", "heat / voxel-proxy weights",
                                                   "props, hair, skirts"], None),
        ("ue", "9  UE build", "ue_build", ["import, materials", "IK Rig + Retargeter", "retarget Idle / Walk",
                                          "character BP, GameMode", "preview level + shot"], None),
        ("script", "10  QA report", "qa_report", ["decision log", "contact sheet", "stress poses"], None),
        ("io", "Playable character", "output", ["SK + materials + anims", "press Play: WASD", "Idle \u2194 Walk"],
         "output"),
    ]
    for (k, t, s, ls, tag), x in zip(row1, xs):
        box(d, x, y1, bw, h1, k, t, s, ls, tag)
    for (k, t, s, ls, tag), x in zip(row2, xs):
        box(d, x, y2, bw, h2, k, t, s, ls, tag)
    for i in range(5):
        arrow(d, [(xs[i] + bw + 4, y1 + h1 / 2), (xs[i + 1] - 6, y1 + h1 / 2)])
        arrow(d, [(xs[i] + bw + 4, y2 + h2 / 2), (xs[i + 1] - 6, y2 + h2 / 2)])
    ym = (y1 + h1 + y2) / 2
    arrow(d, [(xs[5] + bw / 2, y1 + h1 + 4), (xs[5] + bw / 2, ym), (xs[0] + bw / 2, ym), (xs[0] + bw / 2, y2 - 6)])

    ay, ah = 860, 150
    col = (20, 140, 110)
    d.rounded_rectangle(sc(56, ay, W - 56, ay + ah), radius=16 * S, fill=(232, 246, 241), outline=col, width=2 * S)
    d.text(sc(80, ay + 14), "Orchestration: LLM agent (tool calling) or deterministic rule autopilot",
           font=f(25, True), fill=col)
    steps = ["after each step: validators (skeleton structure, skin metrics, facing)",
             "issue \u2192 set_config (values checked against legal enums)",
             "re-run from the right step (max 2 retries)", "every decision logged"]
    sx = 80
    for i, t in enumerate(steps):
        d.text(sc(sx, ay + 64), t, font=f(18), fill=INK)
        tw = d.textlength(t, font=f(18)) / S
        if i < len(steps) - 1:
            arrow(d, [(sx + tw + 10, ay + 78), (sx + tw + 36, ay + 78)], col=col, width=2, head=9)
        sx += tw + 48
    d.text(sc(80, ay + 104), "Example: validator 'pelvis too close to the shoulders' \u2192 agent sets "
           "rig.joint_method=pose2d with a reason \u2192 only autorig re-runs, earlier steps are reused",
           font=f(17), fill=MUTED)
    for x in xs:
        arrow(d, [(x + bw / 2, ay - 4), (x + bw / 2, y2 + h2 + 8)], col=col, width=2, head=9, dashed=True)

    d.text(sc(58, H - 50), "Dashed red = the only step that may need a human (no image-to-3D API key); "
           "Tripo API and local Hunyuan3D back-ends are implemented and switch with one config line.",
           font=f(16), fill=MUTED)

    img = img.resize((W, H), Image.LANCZOS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
