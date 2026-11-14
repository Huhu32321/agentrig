"""Config, run-state and process helpers shared by every step."""
import copy
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLENDER_SCRIPTS = os.path.join(ROOT, "blender_scripts")
UE_SCRIPTS = os.path.join(ROOT, "ue_scripts")


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path=None):
    with open(os.path.join(ROOT, "config.example.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for p in [path, os.path.join(ROOT, "config.yaml")]:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                cfg = _merge(cfg, yaml.safe_load(f) or {})
            break
    return cfg


def cfg_get(cfg, dotted, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def cfg_set(cfg, dotted, value):
    cur = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


# --------------------------------------------------------------------------
# run state (one folder per character)
# --------------------------------------------------------------------------
class Run:
    """Everything about one pipeline run lives in output/<name>/ :
    state.json (step status + reports), logs/, and one sub-folder per step."""

    def __init__(self, name, cfg, image=None):
        self.name = name
        self.cfg = cfg
        self.dir = os.path.abspath(os.path.join(cfg_get(cfg, "paths.output_root", "output"), name))
        os.makedirs(os.path.join(self.dir, "logs"), exist_ok=True)
        self.state_path = os.path.join(self.dir, "state.json")
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        else:
            self.state = {"name": name, "created": _now(), "steps": {}, "decisions": [],
                          "overrides": {}}
        if image:
            self.state["input_image"] = os.path.abspath(image)
        for k, v in self.state.get("overrides", {}).items():
            cfg_set(self.cfg, k, v)
        self.save()

    # paths
    def p(self, *parts):
        path = os.path.join(self.dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def d(self, *parts):
        path = os.path.join(self.dir, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def save(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    # step bookkeeping
    def report(self, step):
        return self.state["steps"].get(step, {}).get("report")

    def set_step(self, step, status, report=None, error=None, seconds=None, actor="pipeline"):
        rec = self.state["steps"].setdefault(step, {"history": []})
        rec.update({"status": status, "report": report, "error": error, "updated": _now(),
                    "seconds": seconds})
        rec["history"].append({"status": status, "time": _now(), "actor": actor,
                               "error": error})
        self.save()

    def decide(self, who, what, why):
        """Log a decision (agent or rule) - this becomes the dev log / audit trail."""
        self.state["decisions"].append({"time": _now(), "by": who, "what": what, "why": why})
        self.save()
        log(f"[decision:{who}] {what} - {why}")

    def override(self, key, value, who, why):
        self.state.setdefault("overrides", {})[key] = value
        cfg_set(self.cfg, key, value)
        self.decide(who, f"set {key} = {value!r}", why)


def _now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------
# processes
# --------------------------------------------------------------------------
def run_proc(cmd, log_path, timeout=None, env=None):
    log("$", " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.run([str(c) for c in cmd], stdout=lf, stderr=subprocess.STDOUT,
                              timeout=timeout, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = open(log_path, encoding="utf-8", errors="replace").read()[-3000:]
        raise RuntimeError(f"command failed (exit {proc.returncode}) - log tail:\n{tail}")
    return dt


def run_blender(cfg, script, args, log_path, timeout=3600):
    """Run a Blender-side script with Blender (if configured) or the bpy module."""
    exe = cfg_get(cfg, "paths.blender_exe") or ""
    script_path = os.path.join(BLENDER_SCRIPTS, script)
    if exe and os.path.exists(exe):
        cmd = [exe, "-b", "--factory-startup", "-P", script_path, "--"] + list(args)
    else:
        if exe:
            log(f"blender_exe not found ({exe}); falling back to the bpy python module")
        cmd = [sys.executable, script_path] + list(args)
    return run_proc(cmd, log_path, timeout=timeout)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def copy_into(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    return dst
