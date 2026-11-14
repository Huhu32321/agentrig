"""Step 8 - drive Unreal Engine 5.7.

editor     : (default) UnrealEditor.exe <uproject> -ExecutePythonScript=<py>
             -unattended -RenderOffscreen. A full editor without a window: it
             has Slate / rendering, so the import / IK Rig / retarget / level
             APIs behave exactly as in the GUI, and the script quits the editor
             when done. Editor must be closed (assets would be locked).
commandlet : UnrealEditor-Cmd.exe <uproject> -run=pythonscript -script="<py> <manifest>"
             Lighter, but in UE 5.7 importing assets in a commandlet asserts in
             Slate (no application) - kept for engines/projects where it works.
remote     : sends the script to an already-open editor through UE's
             Python Remote Execution (Project Settings > Plugins > Python >
             Enable Remote Execution). Lets the artist watch it happen.
"""
import os
import subprocess
import sys
import time

from .core import UE_SCRIPTS, cfg_get, log, run_proc

SCRIPT = os.path.join(UE_SCRIPTS, "ue_build_character.py")


def _fwd(path):
    # forward slashes: UE's command-line parser eats backslash escapes ("\ue_scripts" -> "_scripts")
    return os.path.abspath(path).replace("\\", "/")


def _paths(cfg):
    exe = cfg_get(cfg, "paths.ue_editor_cmd")
    proj = cfg_get(cfg, "paths.ue_project")
    for p, label in ((exe, "paths.ue_editor_cmd"), (proj, "paths.ue_project")):
        if not p or not os.path.exists(p):
            raise FileNotFoundError(f"{label} not found: {p}")
    return exe, proj


def _editor(cfg, manifest, log_path):
    cmd_exe, proj = _paths(cfg)
    exe = os.path.join(os.path.dirname(cmd_exe), "UnrealEditor.exe")
    if not os.path.exists(exe):
        exe = cmd_exe
    env = dict(os.environ, AICHAR_MANIFEST=_fwd(manifest), AICHAR_QUIT="1")
    if os.path.exists(log_path):
        os.remove(log_path)
    cmd = [exe, proj, f"-ExecutePythonScript={_fwd(SCRIPT)}", "-unattended", "-nop4",
           "-nosplash", "-RenderOffscreen", "-NoSound", "-NoLiveCoding", f"-abslog={_fwd(log_path)}"]
    log("$", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          timeout=cfg_get(cfg, "ue.timeout_s", 1800))
    report = _report_path(manifest)
    if not os.path.exists(report):
        tail = open(log_path, encoding="utf-8", errors="replace").read()[-3000:] \
            if os.path.exists(log_path) else ""
        raise RuntimeError(f"UE editor exited ({proc.returncode}) without a report - log tail:\n{tail}")
    return time.time() - t0


def _report_path(manifest):
    import json
    with open(manifest, encoding="utf-8") as f:
        return json.load(f).get("report")


def _commandlet(cfg, manifest, log_path):
    exe, proj = _paths(cfg)
    cmd = [exe, proj, "-run=pythonscript", f"-script={_fwd(SCRIPT)} {_fwd(manifest)}",
           "-unattended", "-nop4", "-nosplash", "-NullRHI", "-stdout", "-FullStdOutLogOutput"]
    return run_proc(cmd, log_path, timeout=cfg_get(cfg, "ue.timeout_s", 1800))


def _remote(cfg, manifest, log_path):
    exe = cfg_get(cfg, "paths.ue_editor_cmd")
    engine = os.path.abspath(os.path.join(os.path.dirname(exe), "..", "..", ".."))
    rex_dir = os.path.join(engine, "Engine", "Plugins", "Experimental", "PythonScriptPlugin",
                           "Content", "Python")
    sys.path.insert(0, rex_dir)
    import remote_execution as rex  # shipped with UE
    remote = rex.RemoteExecution(rex.RemoteExecutionConfig())
    remote.start()
    try:
        t0 = time.time()
        while not remote.remote_nodes and time.time() - t0 < 10:
            time.sleep(0.5)
        if not remote.remote_nodes:
            raise RuntimeError("no UE editor found - is it open with Remote Execution enabled?")
        remote.open_command_connection(remote.remote_nodes[0]["node_id"])
        t0 = time.time()
        res = remote.run_command(f'"{_fwd(SCRIPT)}" "{_fwd(manifest)}"', unattended=True,
                                 exec_mode=rex.MODE_EXEC_FILE)
        with open(log_path, "w", encoding="utf-8") as f:
            for o in res.get("output", []):
                f.write(f"{o.get('type')}: {o.get('output')}\n")
        if not res.get("success", False):
            raise RuntimeError(f"remote execution failed: {res.get('result')}")
        return time.time() - t0
    finally:
        remote.stop()


MODES = {"editor": _editor, "commandlet": _commandlet, "remote": _remote}
PLAY_TEST = os.path.join(UE_SCRIPTS, "ue_play_test.py")


def play_test(cfg, level, out_json, log_path):
    """Press Play in a windowless editor, walk the pawn forward, sample its feet.
    `-ExecCmds="py ..."` (not -ExecutePythonScript, which quits right after the script)
    keeps the editor ticking so PIE can run."""
    import json
    cmd_exe, proj = _paths(cfg)
    exe = os.path.join(os.path.dirname(cmd_exe), "UnrealEditor.exe")
    env = dict(os.environ, AICHAR_PLAYTEST=json.dumps({"level": level, "out": _fwd(out_json)}))
    if os.path.exists(out_json):
        os.remove(out_json)
    cmd = [exe, proj, f'-ExecCmds=py {_fwd(PLAY_TEST)}', "-unattended", "-nop4", "-nosplash",
           "-RenderOffscreen", "-NoSound", "-NoLiveCoding", f"-abslog={_fwd(log_path)}"]
    log("$", " ".join(cmd))
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    if not os.path.exists(out_json):
        raise RuntimeError("play test wrote no result - see " + log_path)
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def run(cfg, manifest, log_path):
    mode = cfg_get(cfg, "ue.mode", "editor")
    log("UE mode:", mode)
    return MODES.get(mode, _editor)(cfg, manifest, log_path)
