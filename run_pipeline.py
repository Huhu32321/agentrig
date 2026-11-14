"""
AgentRig - command line entry point

  python run_pipeline.py --image concept.png --name Hero                 # rule-based autopilot
  python run_pipeline.py --image concept.png --name Hero --mode agent    # LLM agent orchestration
  python run_pipeline.py --name Hero --from autorig                      # resume from a step
  python run_pipeline.py --name Hero --only ue_build                     # run a single step
  python run_pipeline.py --name Hero --set rig.weight_method=proxy_heat --from autorig
"""
import argparse
import sys

import yaml

from pipeline.agent import autopilot, orchestrate
from pipeline.core import Run, load_config, log
from pipeline.steps import STEP_ORDER, execute


def main():
    # piped output (web UI / logs) must be UTF-8 on Windows
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="character concept image (a front/back sheet is fine)")
    ap.add_argument("--name", required=True, help="character name (ASCII, used for asset names)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", default="auto", choices=["auto", "agent"])
    ap.add_argument("--from", dest="start", default=None, choices=STEP_ORDER)
    ap.add_argument("--only", default=None, choices=STEP_ORDER)
    ap.add_argument("--set", action="append", default=[],
                    help="key=value config override (recorded in the decision log)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    run = Run(a.name, cfg, a.image)
    if not run.state.get("input_image") and (a.start in (None, "preprocess")) and not a.only:
        ap.error("--image is required for a new run")
    for kv in a.set:
        k, v = kv.split("=", 1)
        run.override(k, yaml.safe_load(v), "user", "command line --set")

    if a.only:
        execute(run, a.only, actor="user")
        ok = True
    elif a.mode == "agent":
        ok = orchestrate(run, "agent")
    else:
        ok = autopilot(run, start=a.start or "preprocess")
    log("DONE" if ok else "STOPPED", "- see", run.p("07_qa", "REPORT.md"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
