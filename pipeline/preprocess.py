"""Step 1 - concept sheet preprocessing.

* background removal (rembg if installed, else white-background keying)
* split a turnaround sheet into separate views (front / back / side)
* square crop + padding + resize for the 3D generator
"""
import os

import numpy as np
from PIL import Image

from .core import log


def remove_background(im, use_rembg=True, model="isnet-anime"):
    im = im.convert("RGBA")
    if use_rembg:
        try:
            from rembg import new_session, remove
            session = new_session(model)
            out = remove(im, session=session)
            return out, f"rembg:{model}"
        except Exception as e:  # model download blocked / not installed
            log("rembg unavailable -> white keying:", str(e)[:120])
    a = np.asarray(im).astype(np.int16)
    rgb = a[..., :3]
    # background = near-white and low saturation (concept sheets are on white/grey)
    bg_col = np.median(np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]), axis=0)
    dist = np.abs(rgb - bg_col).sum(axis=-1)
    alpha = np.clip((dist - 18) * 8, 0, 255).astype(np.uint8)
    out = np.asarray(im).copy()
    out[..., 3] = alpha
    return Image.fromarray(out), "keying"


def split_views(rgba, min_gap_frac=0.015, min_width_frac=0.08):
    """Split by empty columns in the alpha mask. Returns list of (x0, x1)."""
    a = np.asarray(rgba)[..., 3] > 32
    cols = a.sum(axis=0) > max(2, 0.002 * a.shape[0])
    W = a.shape[1]
    runs, start = [], None
    for x, c in enumerate(cols):
        if c and start is None:
            start = x
        if not c and start is not None:
            runs.append([start, x])
            start = None
    if start is not None:
        runs.append([start, W])
    # merge runs separated by tiny gaps (e.g. thin gap between arm and body)
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_gap_frac * W:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged if (r[1] - r[0]) > min_width_frac * W]


def square_crop(rgba, x0, x1, size=1024, pad=0.08):
    a = np.asarray(rgba)[..., 3]
    sub = a[:, x0:x1] > 32
    ys = np.where(sub.any(axis=1))[0]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = rgba.crop((x0, y0, x1, y1))
    side = int(max(crop.width, crop.height) * (1 + 2 * pad))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def on_white(rgba):
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def run(image_path, outdir, size=1024, use_rembg=True, model="isnet-anime"):
    os.makedirs(outdir, exist_ok=True)
    src = Image.open(image_path)
    rgba, method = remove_background(src, use_rembg, model)
    rgba.save(os.path.join(outdir, "00_alpha.png"))
    views = split_views(rgba)
    if not views:
        views = [(0, rgba.width)]
    files = []
    for i, (x0, x1) in enumerate(views):
        v = square_crop(rgba, x0, x1, size)
        p = os.path.join(outdir, f"view_{i}.png")
        v.save(p)
        on_white(v).save(os.path.join(outdir, f"view_{i}_white.png"))
        files.append(p)
    log(f"preprocess: {len(files)} view(s) via {method}")
    return {"ok": True, "bg_method": method, "views": files,
            "view_boxes": [list(v) for v in views], "source_size": list(src.size)}
