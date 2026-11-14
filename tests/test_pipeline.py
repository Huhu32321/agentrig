"""Unit tests for the parts that run without Blender / UE / network."""
import copy
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "benchmark"))

from pipeline import agent, steps  # noqa: E402
from pipeline.core import Run, cfg_get, cfg_set, load_config  # noqa: E402


@pytest.fixture
def run(tmp_path):
    cfg = load_config()
    cfg_set(cfg, "paths.output_root", str(tmp_path))
    return Run("T", cfg)


# ---------------------------------------------------------------- config
def test_config_template_loads_and_has_legal_defaults():
    cfg = load_config()
    for key, legal in agent.ENUMS.items():
        assert cfg_get(cfg, key) in legal, key


def test_cfg_set_creates_nested_keys():
    cfg = {}
    cfg_set(cfg, "a.b.c", 3)
    assert cfg_get(cfg, "a.b.c") == 3
    assert cfg_get(cfg, "a.x", "default") == "default"


def test_override_is_logged_as_a_decision(run):
    run.override("rig.weight_method", "proxy_heat", "test", "because")
    assert cfg_get(run.cfg, "rig.weight_method") == "proxy_heat"
    assert run.state["decisions"][-1]["by"] == "test"
    assert run.state["overrides"]["rig.weight_method"] == "proxy_heat"


# ---------------------------------------------------------------- agent tools
def test_set_config_rejects_invented_enum_value(run):
    res = agent._tool(run, "set_config", {"key": "rig.weight_method", "value": "geodesic", "reason": "x"}, {})
    assert "error" in res and "proxy_heat" in res["error"]
    assert cfg_get(run.cfg, "rig.weight_method") == "auto"


def test_set_config_rejects_untunable_key(run):
    res = agent._tool(run, "set_config", {"key": "paths.blender_exe", "value": "x", "reason": "x"}, {})
    assert "error" in res


def test_set_config_accepts_legal_value(run):
    res = agent._tool(run, "set_config", {"key": "cleanup.front_axis", "value": "+Y", "reason": "facing away"}, {})
    assert res == {"ok": True}
    assert cfg_get(run.cfg, "cleanup.front_axis") == "+Y"


def test_run_step_retry_cap(run):
    counts = {"autorig": 3}
    res = agent._tool(run, "run_step", {"name": "autorig"}, counts)
    assert "error" in res


# ---------------------------------------------------------------- validators
def _rig_report(pelvis=0.95, shoulder=1.45, neck=1.50, head=1.58):
    return {
        "height_m": 1.75,
        "joints": {"pelvis": [0, 0, pelvis], "upperarm_l": [0.18, 0, shoulder], "neck_01": [0, 0, neck],
                   "head": [0, 0, head]},
        "skinning": {"zero_weight_ratio_final": 0.0,
                     "verts_per_bone": {"head": 10, "upperarm_l": 10, "thigh_l": 10}, "cloth_smoothing": False},
        "geometry_analysis": {},
    }


def test_structure_validator_accepts_plausible_skeleton(run):
    assert not [i for i in steps.v_autorig(run, _rig_report()) if i["severity"] == "error"]


def test_structure_validator_catches_pelvis_in_chest(run):
    # the chibi failure: skin metrics fine, pelvis just below the shoulders
    issues = steps.v_autorig(run, _rig_report(pelvis=1.30))
    errs = [i for i in issues if i["severity"] == "error"]
    assert errs and "pelvis" in errs[0]["issue"]
    assert {f["rerun_from"] for f in errs[0]["fixes"]} == {"autorig"}


def test_structure_validator_catches_neck_inside_head(run):
    issues = steps.v_autorig(run, _rig_report(neck=1.70, head=1.62))
    assert any("neck" in i["issue"] for i in issues if i["severity"] == "error")


def test_orientation_validator_flips_front_axis(run):
    issues = steps.v_orientation(run, {"facing": "back"})
    assert issues[0]["fixes"][0]["overrides"] == {"cleanup.front_axis": "+Y"}


def test_ue_validator_detects_collapsed_retarget(run):
    rep = {"steps": {}, "preview_bones": {"RefPose": {"head": [0, 0, 170], "root": [0, 0, 0]},
                                          "Idle": {"head": [0, 0, 60], "root": [0, 0, 0]}}}
    assert any("Idle" in i["issue"] for i in steps.v_ue(run, rep))


def test_ue_validator_play_test_feet_pinned(run):
    # measured in UE5.7 PIE before / after the ik_foot fix
    before = {"steps": {}, "play_test": {"walk": {"stride_max": 0.4, "speed_mean": 481.7}}}
    after = {"steps": {}, "play_test": {"walk": {"stride_max": 103.0, "speed_mean": 481.7}}}
    assert any("do not step" in i["issue"] for i in steps.v_ue(run, before))
    assert not any("do not step" in i["issue"] for i in steps.v_ue(run, after))


def test_autopilot_applies_fix_and_reruns(run, monkeypatch):
    """Autopilot: an error issue with a fix triggers the override and a re-run of that step."""
    calls = []

    def fake_execute(r, name, actor="x"):
        calls.append(name)
        if name == "autorig" and calls.count("autorig") == 1:
            return {"issues": [steps.issue("bad", "error", [steps.fix({"rig.weight_method": "proxy_heat"},
                                                                       "autorig", "why")])]}
        return {"issues": []}

    monkeypatch.setattr(agent, "execute", fake_execute)
    assert agent.autopilot(run, start="autorig", stop_after="autorig")
    assert calls == ["autorig", "autorig"]
    assert cfg_get(run.cfg, "rig.weight_method") == "proxy_heat"


# ---------------------------------------------------------------- benchmark maths
def test_normalisation_round_trip():
    from run_benchmark import to_norm
    norm = {"rot_z_deg": 180, "scale": 0.5, "offset": [1.0, 2.0, 0.0], "stand_up_euler_deg": None}
    p = to_norm([[2.0, 0.0, 4.0]], norm)[0]
    assert np.allclose(p, [1.0 - 1.0, 2.0, 2.0])


def test_joint_errors_are_height_normalised():
    from run_benchmark import joint_errors
    gt = {"joints": {"head": [0, 0, 1.0], "neck_01": None}}
    norm = {"rot_z_deg": 0, "scale": 1.0, "offset": [0, 0, 0]}
    err = joint_errors(gt, {"joints": {"head": [0, 0, 1.175]}}, norm, 1.75)
    assert err == {"head": pytest.approx(0.1)}


def test_report_tables_render():
    import report
    rows = [{"case": "x", "variant": v, "style": "chibi", "mpjpe": 0.05, "pck05": 0.5,
             "part_accuracy": 0.8, "part_miou": 0.6} for v, _ in report.JOINT_VARIANTS]
    md = report.table(copy.deepcopy(rows), report.JOINT_VARIANTS,
                      [("Stylised", report.by_group("stylised", "mpjpe"))], "test")
    assert "5.0" in md
