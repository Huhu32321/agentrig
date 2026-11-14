"""
Automated Play-in-Editor check of a generated character (run inside UE5.7):

    UnrealEditor.exe <project> -ExecutePythonScript=ue_play_test.py -unattended -RenderOffscreen
    env AICHAR_PLAYTEST = JSON {"level": "/Game/AIChar/X/Maps/L_X_Preview", "out": "result.json"}

Starts PIE in the character's preview level, lets the pawn stand, then drives it forward with
movement input (like holding W) and samples foot / knee / pelvis positions. This catches
problems that only exist at runtime in the template anim blueprint (e.g. the foot-IK Control
Rig pulling both ankles together) and never show in the preview level.
"""
import json
import os

import unreal

CFG = json.loads(os.environ.get("AICHAR_PLAYTEST", "{}"))
LEVEL = CFG.get("level")
OUT = CFG.get("out", os.path.join(os.path.dirname(os.path.abspath(__file__)), "play_test.json"))
BONES = ("pelvis", "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r")
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
state = {"tick": 0, "samples": [], "handle": None, "started": False, "error": None}


def game_world():
    try:
        return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
    except Exception:
        return None


def sample(phase, pawn):
    comp = pawn.get_editor_property("mesh")
    xf = pawn.get_actor_transform()
    pts = {}
    for b in BONES:
        w = comp.get_socket_location(b)
        loc = unreal.MathLibrary.inverse_transform_location(xf, w)
        pts[b] = [round(loc.x, 1), round(loc.y, 1), round(loc.z, 1)]
    v = pawn.get_velocity()
    state["samples"].append({"phase": phase, "tick": state["tick"], "speed": round(v.length(), 1), "bones": pts})


def finish():
    unreal.unregister_slate_post_tick_callback(state["handle"])
    try:
        les.editor_request_end_play()
    except Exception:
        pass
    res = {"level": LEVEL, "error": state["error"], "samples": state["samples"]}
    walk = [s for s in state["samples"] if s["phase"] == "walk"]
    idle = [s for s in state["samples"] if s["phase"] == "idle"]

    def stats(ss):
        if not ss:
            return None
        gap = [abs(s["bones"]["foot_l"][1] - s["bones"]["foot_r"][1]) for s in ss]   # lateral (actor Y)
        stride = [abs(s["bones"]["foot_l"][0] - s["bones"]["foot_r"][0]) for s in ss]  # fore-aft (actor X)
        hip_gap = [abs(s["bones"]["thigh_l"][1] - s["bones"]["thigh_r"][1]) for s in ss]
        return {"ankle_gap_min": min(gap), "ankle_gap_mean": sum(gap) / len(gap), "hip_gap": sum(hip_gap) / len(hip_gap),
                "stride_max": max(stride), "speed_mean": sum(s["speed"] for s in ss) / len(ss)}
    res["idle"], res["walk"] = stats(idle), stats(walk)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    unreal.SystemLibrary.quit_editor()


def on_tick(dt):
    state["tick"] += 1
    t = state["tick"]
    try:
        if not state["started"]:
            if t == 5:
                les.editor_request_begin_play()
                state["started"] = True
            return
        if t > 1500:
            state["error"] = "timeout"
            finish()
            return
        w = game_world()
        if w is None:
            return
        pawn = unreal.GameplayStatics.get_player_pawn(w, 0)
        if pawn is None:
            return
        state.setdefault("t0", t)
        k = t - state["t0"]
        if 60 <= k < 120 and k % 10 == 0:
            sample("idle", pawn)
        if k >= 120:
            pawn.add_movement_input(pawn.get_actor_forward_vector(), 1.0)
            if k >= 180 and k % 6 == 0:
                sample("walk", pawn)
        if k >= 300:
            finish()
    except Exception as e:  # never leave the editor hanging
        state["error"] = repr(e)
        finish()


if LEVEL:
    les.load_level(LEVEL)
state["handle"] = unreal.register_slate_post_tick_callback(on_tick)
