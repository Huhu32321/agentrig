"""Step 3 - image -> textured 3D model. Pluggable backends.

hunyuan_local : Tencent Hunyuan3D-2.1 `api_server.py` (open source, local GPU)
tripo         : Tripo3D OpenAPI (commercial; supports front/back multiview)
manual        : use an existing model file (e.g. made in ComfyUI / web UI)
"""
import base64
import os
import shutil
import time

import requests

from .core import cfg_get, log


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# --------------------------------------------------------------------------
def hunyuan_local(cfg, front, back, out_path):
    url = cfg_get(cfg, "gen3d.hunyuan_url").rstrip("/")
    p = cfg_get(cfg, "gen3d.hunyuan", {})
    body = {"image": _b64(front), "remove_background": False,
            "texture": bool(p.get("texture", True)), "seed": int(p.get("seed", 1234)),
            "octree_resolution": int(p.get("octree_resolution", 384)),
            "num_inference_steps": int(p.get("num_inference_steps", 30)),
            "guidance_scale": float(p.get("guidance_scale", 5.0)),
            "face_count": int(p.get("face_count", 60000))}
    requests.get(f"{url}/health", timeout=10).raise_for_status()
    r = requests.post(f"{url}/send", json=body, timeout=60)
    r.raise_for_status()
    uid = r.json()["uid"]
    log("hunyuan task", uid)
    t0, timeout = time.time(), cfg_get(cfg, "gen3d.timeout_s", 1800)
    while time.time() - t0 < timeout:
        s = requests.get(f"{url}/status/{uid}", timeout=30).json()
        if s.get("status") == "completed" and s.get("model_base64"):
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(s["model_base64"]))
            return {"backend": "hunyuan_local", "task": uid, "params": body | {"image": "<b64>"}}
        if s.get("status") == "error":
            raise RuntimeError(f"hunyuan error: {s.get('message')}")
        time.sleep(5)
    raise TimeoutError("hunyuan generation timed out")


# --------------------------------------------------------------------------
def _tripo_upload(base, key, path):
    with open(path, "rb") as f:
        r = requests.post(f"{base}/upload", headers={"Authorization": f"Bearer {key}"},
                          files={"file": (os.path.basename(path), f, "image/png")}, timeout=120)
    r.raise_for_status()
    return r.json()["data"]["image_token"]


def tripo(cfg, front, back, out_path):
    t = cfg_get(cfg, "gen3d.tripo", {})
    from .llm import env_key
    key = env_key(t.get("api_key_env", "TRIPO_API_KEY"))
    if not key:
        raise RuntimeError("TRIPO_API_KEY not set")
    base = t.get("base_url", "https://api.tripo3d.ai/v2/openapi").rstrip("/")
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    tok_front = _tripo_upload(base, key, front)
    if back and cfg_get(cfg, "gen3d.use_multiview", True):
        tok_back = _tripo_upload(base, key, back)
        # order: front, left, back, right ; empty dict = view not provided
        task = {"type": "multiview_to_model",
                "files": [{"type": "png", "file_token": tok_front}, {},
                          {"type": "png", "file_token": tok_back}, {}]}
    else:
        task = {"type": "image_to_model", "file": {"type": "png", "file_token": tok_front}}
    task.update({"model_version": t.get("model_version"), "texture": True, "pbr": True,
                 "texture_quality": t.get("texture_quality", "detailed")})
    r = requests.post(f"{base}/task", headers=hdr, json=task, timeout=60)
    r.raise_for_status()
    tid = r.json()["data"]["task_id"]
    log("tripo task", tid)
    t0, timeout = time.time(), cfg_get(cfg, "gen3d.timeout_s", 1800)
    while time.time() - t0 < timeout:
        d = requests.get(f"{base}/task/{tid}", headers=hdr, timeout=30).json()["data"]
        st = d.get("status")
        if st == "success":
            out = d.get("output", {})
            url = out.get("pbr_model") or out.get("model")
            with requests.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    shutil.copyfileobj(resp.raw, f)
            return {"backend": "tripo", "task": tid, "request": task}
        if st in ("failed", "cancelled", "banned", "expired"):
            raise RuntimeError(f"tripo task {st}")
        time.sleep(5)
    raise TimeoutError("tripo generation timed out")


# --------------------------------------------------------------------------
def manual(cfg, front, back, out_path):
    src = cfg_get(cfg, "gen3d.manual_model")
    if not src or not os.path.exists(src):
        raise FileNotFoundError(f"gen3d.manual_model not found: {src}")
    ext = os.path.splitext(src)[1]
    out = os.path.splitext(out_path)[0] + ext
    shutil.copy2(src, out)
    return {"backend": "manual", "source": src, "model": out}


BACKENDS = {"hunyuan_local": hunyuan_local, "tripo": tripo, "manual": manual}


def run(cfg, front, back, outdir):
    os.makedirs(outdir, exist_ok=True)
    backend = cfg_get(cfg, "gen3d.backend", "hunyuan_local")
    out_path = os.path.join(outdir, "raw_model.glb")
    t0 = time.time()
    info = BACKENDS[backend](cfg, front, back, out_path)
    model = info.get("model", out_path)
    info.update({"ok": True, "model": model, "seconds": round(time.time() - t0, 1),
                 "size_mb": round(os.path.getsize(model) / 1e6, 2)})
    return info
