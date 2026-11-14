"""
Joint-placement and skinning benchmark on artist-rigged humanoids.

    python tools/benchmark/download.py --data D:/bench_data
    python tools/benchmark/run_benchmark.py --data D:/bench_data [--cases kay_knight,cesium_man] [--poses rest,apose]
    python tools/benchmark/report.py --data D:/bench_data

For every model and pose (rest pose = mostly T-pose; A-pose = arms lowered 45 deg) the
artist rig is baked into an unrigged mesh, the pipeline's cleanup -> render_views -> pose2d
steps run once, then `autorig` runs once per ablation variant. Results are appended to
<data>/results.jsonl (re-running skips finished rows).
"""
import argparse
import copy
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from datasets import CASES  # noqa: E402
from pipeline.core import Run, cfg_get, cfg_set, load_config, log, run_proc  # noqa: E402
from pipeline.steps import execute  # noqa: E402

JOINTS = ["neck_01", "head", "upperarm_l", "lowerarm_l", "hand_l", "upperarm_r", "lowerarm_r", "hand_r",
          "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r"]

# ablations: every variant only changes the autorig step
VARIANTS = {
    # joint placement
    "geo_profile":      {"rig.joint_method": "geometric", "rig.arm_anchor": False},
    "geo_profile+kp2d": {"rig.arm_anchor": False, "rig.vertical_reconcile": False},
    "geo_profile+kp2d+reconcile": {"rig.arm_anchor": False},   # = the original system
    "geo_arm":          {"rig.joint_method": "geometric"},
    "full":             {},                                     # arm anchor + kp2d + arbitration
    # skinning (with full joints)
    "full_proxyheat":   {"rig.weight_method": "proxy_heat"},
    "full_nopost":      {"rig.skin_postprocess": False},
}
SHARED_STEPS = ("cleanup", "render_views", "pose2d")


def blender(cfg, script, args, log_path):
    exe = cfg_get(cfg, "paths.blender_exe")
    return run_proc([exe, "-b", "--factory-startup", "-P", os.path.join(HERE, script), "--"] + args, log_path)


def to_norm(P, norm):
    import math
    P = np.asarray(P, dtype=np.float64)
    if norm.get("stand_up_euler_deg"):
        raise NotImplementedError("benchmark meshes are upright")
    rz = math.radians(norm.get("rot_z_deg") or 0)
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return np.asarray(norm["offset"]) + norm["scale"] * (P @ Rz.T)


def joint_errors(gt, rig_report, norm, height):
    out = {}
    for j in JOINTS:
        g, p = gt["joints"].get(j), (rig_report.get("joints") or {}).get(j)
        if g is None or p is None:
            continue
        out[j] = float(np.linalg.norm(to_norm([g], norm)[0] - np.asarray(p)) / height)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cases", default="")
    ap.add_argument("--poses", default="rest,apose")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    data = os.path.abspath(a.data)
    base_cfg = load_config()
    cfg_set(base_cfg, "llm.enabled", False)
    cfg_set(base_cfg, "vlm.enabled", False)
    cfg_set(base_cfg, "rig.render_tests", False)
    cfg_set(base_cfg, "rig.render_engine", "BLENDER_WORKBENCH")
    want = set(filter(None, a.cases.split(",")))
    res_path = os.path.join(data, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["case"], r["pose"], r["variant"]))

    for c in CASES:
        if want and c["id"] not in want:
            continue
        for pose in a.poses.split(","):
            todo = [v for v in a.variants.split(",") if (c["id"], pose, v) not in done]
            if not todo:
                continue
            key = f"{c['id']}__{pose}"
            case_dir = os.path.join(data, "cases", key)
            os.makedirs(os.path.join(data, "logs"), exist_ok=True)
            if not os.path.exists(os.path.join(case_dir, "gt.json")):
                blender(base_cfg, "bl_prepare.py",
                        ["--in", os.path.join(data, "raw", c["id"] + ".glb"), "--outdir", case_dir,
                         "--apose", "45" if pose == "apose" else "0"],
                        os.path.join(data, "logs", f"prepare_{key}.log"))
            gt = json.load(open(os.path.join(case_dir, "gt.json")))
            if pose == "apose" and not gt["a_posed"]:
                log(f"{key}: model is already arms-down, A-pose variant skipped")
                continue

            # shared steps
            scfg = copy.deepcopy(base_cfg)
            cfg_set(scfg, "paths.output_root", os.path.join(data, "runs", "_shared"))
            shared = Run(key, scfg)
            if shared.state["steps"].get("pose2d", {}).get("status") != "done":
                shared.state["steps"] = {
                    "analyze": {"status": "done", "report": {}, "history": []},
                    "generate3d": {"status": "done", "report": {"model": os.path.join(case_dir, "mesh.glb")},
                                   "history": []}}
                for s in SHARED_STEPS:
                    try:
                        execute(shared, s, actor="benchmark")
                    except Exception as e:
                        if s != "pose2d":
                            raise
                        shared.set_step("pose2d", "done", report={"ok": False, "reason": str(e)[:300]})
            norm = shared.report("cleanup")["normalisation"]
            kp = shared.report("pose2d") or {}

            for v in todo:
                vcfg = copy.deepcopy(base_cfg)
                cfg_set(vcfg, "paths.output_root", os.path.join(data, "runs", v))
                vr = Run(key, vcfg)
                vr.state["steps"] = {k: copy.deepcopy(shared.state["steps"][k])
                                     for k in ("analyze", "generate3d") + SHARED_STEPS}
                vr.state["overrides"] = {}
                for k2, val in VARIANTS[v].items():
                    vr.state["overrides"][k2] = val
                    cfg_set(vr.cfg, k2, val)
                kpf = shared.p("04_views", "keypoints.json")
                if os.path.exists(kpf):
                    shutil.copy2(kpf, vr.p("04_views", "keypoints.json"))
                t0 = time.time()
                row = {"case": c["id"], "pose": pose, "variant": v, "style": c["style"],
                       "family": gt["family"], "pose2d_ok": bool(kp.get("ok"))}
                try:
                    rep = execute(vr, "autorig", actor="benchmark")
                    H = rep.get("height_m") or 1.75
                    je = joint_errors(gt, rep, norm, H)
                    m_out = vr.p("eval.json")
                    with open(vr.p("norm.json"), "w") as f:
                        json.dump(norm, f)
                    blender(vcfg, "bl_eval.py", ["--rigged", rep["rigged_blend"], "--case", case_dir,
                                                 "--norm", vr.p("norm.json"), "--out", m_out],
                            vr.p("logs", "eval.log"))
                    m = json.load(open(m_out))
                    row.update({
                        "ok": True, "joint_err": je, "mpjpe": float(np.mean(list(je.values()))),
                        "pck05": float(np.mean([e < 0.05 for e in je.values()])),
                        "part_accuracy": m["part_accuracy"], "part_miou": m["part_miou"],
                        "part_iou": m["part_iou"],
                        "joint_method": rep.get("joint_method"),
                        "pose2d_joints_used": rep.get("pose2d_joints_used"),
                        "stylised": (rep.get("geometry_analysis") or {}).get("stylised_proportions"),
                        "weight_method": rep["skinning"].get("method_used"),
                        "validator_errors": [i["issue"] for i in rep.get("issues", []) if i["severity"] == "error"],
                        "seconds": round(time.time() - t0, 1)})
                except Exception as e:
                    row.update({"ok": False, "error": str(e)[-600:]})
                with open(res_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                log(f"{key:34s} {v:16s} " + (f"MPJPE {row['mpjpe']*100:5.2f}%H  part acc {row['part_accuracy']:.3f}"
                                             if row.get("ok") else "FAILED " + row.get("error", "")[-200:]))


if __name__ == "__main__":
    main()
