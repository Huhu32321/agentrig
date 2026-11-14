"""Step 6 - 2D pose estimation on the orthographic front render (MediaPipe).

The 33 MediaPipe landmarks (normalised image coords + visibility) are saved
to JSON; bl_autorig.py lifts them back to 3D. If MediaPipe is not
installed the step reports ok=False and rigging falls back to geometry.
"""
import os
import urllib.request

from .core import log

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
NAMES = ["nose", "l_eye_in", "l_eye", "l_eye_out", "r_eye_in", "r_eye", "r_eye_out", "l_ear",
         "r_ear", "mouth_l", "mouth_r", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
         "l_wrist", "r_wrist", "l_pinky", "r_pinky", "l_index", "r_index", "l_thumb",
         "r_thumb", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle", "l_heel",
         "r_heel", "l_foot_index", "r_foot_index"]


def _model_path():
    cache = os.path.join(os.path.expanduser("~"), ".cache", "aichar_pipeline")
    os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, "pose_landmarker_heavy.task")
    if not os.path.exists(p):
        log("downloading MediaPipe pose model ...")
        urllib.request.urlretrieve(MODEL_URL, p)
    return p


def run(front_png, out_json):
    import json
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        opts = vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=_model_path()),
            running_mode=vision.RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3)
        with vision.PoseLandmarker.create_from_options(opts) as det:
            res = det.detect(mp.Image.create_from_file(front_png))
        if not res.pose_landmarks:
            out = {"ok": False, "reason": "no person detected"}
        else:
            lms = [{"name": NAMES[i], "x": l.x, "y": l.y, "z": l.z,
                    "visibility": float(getattr(l, "visibility", 1.0) or 0.0)}
                   for i, l in enumerate(res.pose_landmarks[0])]
            vis = [l["visibility"] for l in lms[11:29]]
            face = max(lms[0]["visibility"], lms[7]["visibility"], lms[8]["visibility"])
            out = {"ok": True, "landmarks": lms, "mean_body_visibility": sum(vis) / len(vis),
                   "face_visibility": face,
                   # person's left must appear on the image right in a front view
                   "mirrored_suspect": lms[11]["x"] < lms[12]["x"]}
    except Exception as e:
        out = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log("pose2d:", "ok" if out.get("ok") else out.get("reason"))
    return out
