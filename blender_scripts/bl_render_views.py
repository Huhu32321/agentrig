"""
Render orthographic front / side views of the cleaned character.

The front view feeds the 2D pose estimator (pose2d.py); the camera
parameters are written next to it so 2D keypoints can be mapped back to
world space exactly (orthographic => linear mapping).

Usage:
  blender -b -P bl_render_views.py -- --in clean.blend --outdir views/ [--res 1024]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bl_common as C  # noqa: E402

import bpy  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--engine", default="CYCLES")
    p.add_argument("--samples", type=int, default=16)
    return p.parse_args(C.script_args())


def main():
    a = parse()
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.inp))
    body = C.get_main_mesh()
    os.makedirs(a.outdir, exist_ok=True)
    C.setup_render(a.engine, a.res, a.samples)
    _, front = C.front_camera_for(body)
    front_png = C.render_to(os.path.join(os.path.abspath(a.outdir), "front.png"))
    _, side = C.side_camera_for(body)
    side_png = C.render_to(os.path.join(os.path.abspath(a.outdir), "side.png"))
    mn, mx = C.world_bbox(body)
    C.write_json(os.path.join(a.outdir, "views.json"), {
        "front": {"image": front_png, "res": a.res, **front},
        "side": {"image": side_png, "res": a.res, **side},
        "bbox_min": list(mn), "bbox_max": list(mx),
    })
    C.log("views rendered", front_png, side_png)


if __name__ == "__main__":
    main()
