"""
Step 4 - Automatic skeleton + skinning + FBX export (Blender, headless).

Joint placement
  * pose2d    : 2D keypoints (MediaPipe) on an orthographic front render are
                lifted back to 3D (exact for ortho cameras); depth is found by
                ray-casting through the mesh and taking the mid-point.
  * geometric : pure mesh analysis (slices, silhouette gaps, ray casts).
  * auto      : geometric as base, pose2d overrides joints it is confident in.
  Every joint is then sanity-checked and mirrored for left/right symmetry.

Skeleton
  Bone names / hierarchy follow the UE5 Mannequin (Manny) so UE can
  auto-generate IK Rig retarget chains and reuse Manny animations directly.
  Finger bones exist (for template matching) but are non-deforming because AI
  meshes normally have fused "mitten" fingers.

Skinning
  * heat       : Blender bone-heat automatic weights
  * proxy_heat : weights computed on a watertight voxel-remeshed proxy and
                 transferred back (robust on non-manifold AI meshes)
  * auto       : heat -> verify -> fall back to proxy_heat
  Post: orphan-vertex fix, optional loose-cloth smoothing, max-4 influences,
  normalisation.

Usage:
  blender -b -P bl_autorig.py -- --in clean.blend --outdir rig/ --report r.json
          [--keypoints kp.json --views views.json] [--joint-method auto]
          [--weight-method auto] [--cloth-smoothing auto|on|off] [--name Char]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bl_common as C  # noqa: E402

import bpy  # noqa: E402
import mathutils  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
from mathutils.kdtree import KDTree  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--name", default="AIChar")
    p.add_argument("--keypoints", default=None)
    p.add_argument("--views", default=None)
    p.add_argument("--joint-method", default="auto", choices=["auto", "pose2d", "geometric"])
    p.add_argument("--weight-method", default="auto", choices=["auto", "heat", "proxy_heat"])
    p.add_argument("--cloth-smoothing", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--vertical-reconcile", type=int, default=1,
                   help="0 disables the stylised-proportion reconciliation (ablation)")
    p.add_argument("--arm-anchor", type=int, default=1,
                   help="0 disables arm-anchored shoulders and 2D/geometry arbitration (ablation)")
    p.add_argument("--skin-post", type=int, default=1,
                   help="0 disables far-weight pruning / rigid islands / head attachments (ablation)")
    p.add_argument("--render-tests", type=int, default=1)
    p.add_argument("--render-res", type=int, default=512)
    p.add_argument("--render-samples", type=int, default=8)
    p.add_argument("--render-engine", default="CYCLES")
    return p.parse_args(C.script_args())


# ==========================================================================
# 1. Joint estimation
# ==========================================================================
class MeshProbe:
    """Geometry queries on the character mesh (world space, Z-up, facing -Y)."""

    def __init__(self, obj):
        self.obj = obj
        self.V = C.vertex_array(obj)
        self.H = float(self.V[:, 2].max())
        dg = bpy.context.evaluated_depsgraph_get()
        self.bvh = BVHTree.FromObject(obj, dg)
        mn, mx = C.world_bbox(obj)
        self.ymin, self.ymax = mn.y - 1.0, mx.y + 1.0

    # ---- slices --------------------------------------------------------
    def clusters(self, z, thick=None, gap=None):
        """1-D clusters of vertex X coordinates inside a horizontal band."""
        H = self.H
        thick = thick or 0.01 * H
        gap = gap or 0.012 * H
        m = np.abs(self.V[:, 2] - z) < thick / 2
        xs = np.sort(self.V[m, 0])
        if xs.size == 0:
            return []
        cuts = np.where(np.diff(xs) > gap)[0]
        out, start = [], 0
        for c in list(cuts) + [xs.size - 1]:
            seg = xs[start:c + 1]
            out.append((float(seg[0]), float(seg[-1]), int(seg.size)))
            start = c + 1
        return out

    def central(self, z, **kw):
        cl = self.clusters(z, **kw)
        if not cl:
            return None
        for c in cl:
            if c[0] <= 0 <= c[1]:
                return c
        return min(cl, key=lambda c: min(abs(c[0]), abs(c[1])))

    def central_half(self, z, **kw):
        """Half width of the body part that contains the centre line at height z.

        Rays are shot from inside the body (x=0) towards +X and -X; the first
        exit hit is the surface of the central part, so arms that are separated
        by a gap are ignored while a shoulder fused to an arm is included.
        Falls back to vertex clustering when x=0 is not inside the mesh."""
        ymid, ok = self.depth_mid(0.0, z)
        if ok:
            f = self.bvh.ray_cast(Vector((0, self.ymin, z)), Vector((0, 1, 0)))[0]
            b = self.bvh.ray_cast(Vector((0, self.ymax, z)), Vector((0, -1, 0)))[0]
            d = (b.y - f.y) if (f is not None and b is not None) else 0.0
            vals = []
            for dy in (-0.25 * d, 0.0, 0.25 * d):
                o = Vector((0.0, ymid + dy, z))
                hp = self.exit_distance(o, Vector((1, 0, 0)))
                hn = self.exit_distance(o, Vector((-1, 0, 0)))
                if hp is not None and hn is not None:
                    vals.append(max(hp, hn))
            if vals:
                return float(np.median(vals))
        c = self.central(z, **kw)
        return None if c is None or c[2] < 20 else max(abs(c[0]), abs(c[1]))

    # ---- rays ----------------------------------------------------------
    def exit_distance(self, origin, direction, gap=None, max_dist=None):
        """Distance from an interior point to the outer boundary of the part
        it is in. Walks through successive hits so stacked layers (skin under
        clothes, AI double shells) are skipped, but stops at a real gap of
        empty space (e.g. between torso and a hanging arm)."""
        gap = gap or 0.02 * self.H
        max_dist = max_dist or 0.6 * self.H
        hits, o, travelled = [], origin.copy(), 0.0
        for _ in range(24):
            h = self.bvh.ray_cast(o, direction, max_dist - travelled)
            if h[0] is None:
                break
            travelled += h[3]
            hits.append(travelled)
            o = h[0] + direction * 1e-4
            travelled += 1e-4
        if not hits:
            return None
        boundary = hits[0]
        # hits come in pairs (exit, enter); a following pair closer than `gap`
        # means another layer of the same part
        k = 1
        while k + 1 < len(hits):
            if hits[k] - boundary < gap:
                boundary = hits[k + 1]
                k += 2
            else:
                break
        return boundary

    def depth_mid(self, x, z, bias=0.5):
        """Mid point (in Y) of the mesh along a front->back ray at (x, z)."""
        hit_f = self.bvh.ray_cast(Vector((x, self.ymin, z)), Vector((0, 1, 0)))
        hit_b = self.bvh.ray_cast(Vector((x, self.ymax, z)), Vector((0, -1, 0)))
        if hit_f[0] is not None and hit_b[0] is not None and hit_b[0].y >= hit_f[0].y:
            return hit_f[0].y + bias * (hit_b[0].y - hit_f[0].y), True
        # fallback: centroid of nearby vertices
        d = np.hypot(self.V[:, 0] - x, self.V[:, 2] - z)
        idx = np.argsort(d)[:30]
        return float(self.V[idx, 1].mean()), False

    def solid_at(self, x, z):
        hit = self.bvh.ray_cast(Vector((x, self.ymin, z)), Vector((0, 1, 0)))
        return hit[0] is not None


def vertical_intervals(probe, x, y, merge=0.012):
    """Inside-mesh intervals [(lo, hi), ...] of the vertical line through (x, y),
    from ray casting (independent of vertex density). Nested shells are merged."""
    H = probe.H
    hits, o = [], Vector((x, y, H + 0.05 * H))
    for _ in range(40):
        h = probe.bvh.ray_cast(o, Vector((0, 0, -1)), 3 * H)
        if h[0] is None:
            break
        hits.append(h[0].z)
        o = h[0] + Vector((0, 0, -1e-4))
    iv = [(hits[i + 1], hits[i]) for i in range(0, len(hits) - 1, 2)]
    iv.sort(key=lambda t: -t[1])
    out = []
    for lo, hi in iv:
        if out and out[-1][0] - hi < merge * H:
            out[-1] = (min(out[-1][0], lo), out[-1][1])
        else:
            out.append((lo, hi))
    return out


def arm_sweep(probe, side, tip, step=0.006, debug=False):
    """Arm-anchored shoulder: walk from the hand tip towards the body and, at each
    x, take the inside-mesh interval of a vertical line through the arm. While we
    are on the arm the interval is short; where the arm merges into the torso it
    jumps to roughly the shoulder-to-hip distance. Unlike the width profile this
    assumes nothing about head size, so it holds for chibi and realistic bodies.

    Returns {"ok", "attach": [x, z], "shoulder": [x, z], "thickness"} or {"ok": False}."""
    H = probe.H
    s = 1 if side == "l" else -1
    x0 = s * tip[0]
    if x0 < 0.12 * H:  # arms hanging straight down: no lateral arm to follow
        return {"ok": False, "reason": "arm not lateral"}
    zc, track, misses = tip[2], [], 0
    for x in np.arange(x0 - 0.01 * H, 0.0, -step * H):
        y, _ = probe.depth_mid(s * x, zc)
        ivs = []
        for dy in (-0.012 * H, 0.0, 0.012 * H):
            ivs += vertical_intervals(probe, s * x, y + dy)
        cand = [iv for iv in ivs if iv[0] - 0.04 * H <= zc <= iv[1] + 0.04 * H]
        if not cand:
            misses += 1
            if misses > 4:
                return {"ok": False, "reason": "lost the arm", **({"track": track} if debug else {})}
            continue
        misses = 0
        lo, hi = min(c[0] for c in cand), max(c[1] for c in cand)
        track.append((float(x), lo, hi))
        ext = hi - lo
        if len(track) >= 4:
            recent = float(np.median([t[2] - t[1] for t in track[-9:-1]]))
            if ext > max(0.14 * H, 2.0 * recent):
                k = len(track) - 2
                while k > 0 and (track[k][2] - track[k][1]) > 1.5 * recent:
                    k -= 1
                ax, alo, ahi = track[k]
                pts = np.array([(t[0], 0.5 * (t[1] + t[2])) for t in track[: k + 1]])
                d = pts[-1] - pts[len(pts) // 3] if len(pts) > 3 else np.array([-1.0, 0.0])
                d = d / (np.linalg.norm(d) + 1e-9)
                thick = float(np.median([t[2] - t[1] for t in track[max(0, k - 8): k + 1]]))
                attach = np.array([ax, 0.5 * (alo + ahi)])
                sh = attach + d * 0.35 * thick  # the joint sits inside the deltoid
                return {"ok": True, "attach": [s * attach[0], attach[1]], "shoulder": [s * sh[0], sh[1]],
                        "thickness": thick, **({"track": track} if debug else {})}
        zc = 0.5 * (lo + hi)
    return {"ok": False, "reason": "no attachment found", **({"track": track} if debug else {})}


def estimate_geometric(probe, use_arm_anchor=True):
    """Pure geometry joint estimation. Returns dict name -> [x, z] + notes."""
    V, H = probe.V, probe.H
    notes = []
    J = {}

    # --- hands -----------------------------------------------------------
    # Not simply the most lateral point: a wide hat brim, shoulder pads or a cape can
    # stick out further than the hands. Take the lateral-extent profile over height,
    # keep its strong local peaks and use the LOWEST one (hats/pads sit above the hands).
    hands = {}
    for side, s in (("l", 1), ("r", -1)):
        m = V[:, 2] > 0.15 * H
        zb = np.arange(0.15 * H, H, 0.02 * H)
        prof = np.array([(s * V[m & (np.abs(V[:, 2] - z) < 0.012 * H), 0]).max(initial=-1.0) for z in zb])
        gmax = prof.max()
        peaks = [i for i in range(len(zb)) if prof[i] >= 0.6 * gmax and prof[i] > 0.1 * H
                 and prof[i] >= prof[max(0, i - 3): i + 4].max() - 1e-6]
        zsel = zb[min(peaks)] if peaks else zb[int(np.argmax(prof))]
        mm = m & (np.abs(V[:, 2] - zsel) < 0.06 * H)
        X = s * V[mm, 0]
        pts = V[mm][X >= np.quantile(X, 0.99)]
        hands[side] = pts.mean(axis=0)
    hand_z = float(min(hands["l"][2], hands["r"][2]))
    arms_up = hand_z > 0.7 * H  # T-pose: the neck must be above the arm line
    neck_floor = max(0.6 * H, hand_z + 0.015 * H) if arms_up else 0.48 * H

    # --- width profile of the central body part ------------------------
    zs = np.linspace(0.98 * H, 0.4 * H, 117)
    w = np.array([probe.central_half(z) or np.nan for z in zs])
    valid = np.isfinite(w)
    wf = np.where(valid, w, 0.0)
    # head = widest point in the top 22 %; shoulders = widest point below it
    # (above 0.45H); neck = narrowest slice between the two. Bounding the
    # search on both sides keeps waists / gaps between layers out of it.
    top = zs > 0.78 * H
    if arms_up:  # T-pose: horizontal arms are fused into the slices -> search above them
        top = zs > max(0.78 * H, neck_floor + 0.04 * H)
    i_head = int(np.argmax(np.where(top, wf, -1)))
    z_head = float(zs[i_head])
    lower = ((np.arange(zs.size) > i_head + 2) & (zs > max(0.4 * H, z_head - 0.3 * H))
             & (zs > neck_floor - 0.05 * H))
    i_sh = int(np.argmax(np.where(lower, wf, -1)))
    between = (np.arange(zs.size) > i_head) & (np.arange(zs.size) < i_sh) & (zs > neck_floor)
    if not between.any():
        between = (np.arange(zs.size) > i_head) & (zs > neck_floor)
    cand = np.where(between & valid & (wf > 0.008 * H), wf, np.inf)
    if np.isfinite(cand).any():
        i_neck = int(np.argmin(cand))
        neck_z, neck_half = float(zs[i_neck]), float(wf[i_neck])
    else:  # no usable slice between head and shoulders -> proportion, and say so
        neck_z = z_head - 0.08 * H
        neck_half = float(np.nanmedian(w[valid])) * 0.5 if valid.any() else 0.03 * H
        notes.append("neck not found between head and shoulders; using proportion")

    # --- shoulders: widest band within 0.15H below the neck ----------------
    band = (zs < neck_z - 0.01 * H) & (zs > neck_z - 0.16 * H)
    wmax = float(np.max(np.where(band, wf, -1)))
    in_band = band & (wf > 0.9 * wmax)
    shoulder_z = float(zs[in_band].mean()) if in_band.any() else neck_z - 0.08 * H
    t_pose_like = wmax > 4.0 * neck_half  # arms fused into the shoulder slice
    chest_z = shoulder_z - 0.07 * H
    torso_half = probe.central_half(chest_z) or 2.5 * neck_half
    if torso_half > 0.95 * wmax:  # arms fused with the chest (A-pose, wide sleeves)
        torso_half = 0.7 * wmax
    torso_half = float(np.clip(torso_half, 0.055 * H, 0.15 * H))

    # --- arm-anchored shoulders (overrides the width-profile guess) -------
    sweeps = ({sd: arm_sweep(probe, sd, hands[sd]) for sd in ("l", "r")} if use_arm_anchor
              else {sd: {"ok": False, "reason": "disabled"} for sd in ("l", "r")})
    arm_anchored = False
    if all(sw["ok"] for sw in sweeps.values()):
        zl, zr = sweeps["l"]["shoulder"][1], sweeps["r"]["shoulder"][1]
        if abs(zl - zr) < 0.05 * H and 0.3 * H < 0.5 * (zl + zr) < 0.92 * H:
            arm_anchored = True
            sz = 0.5 * (zl + zr)
            if abs(sz - shoulder_z) > 0.03 * H:
                notes.append(f"arm-anchored shoulders at {sz:.3f} (width profile said {shoulder_z:.3f})")
            shoulder_z = sz
            torso_half = float(np.clip(0.5 * (abs(sweeps["l"]["shoulder"][0]) + abs(sweeps["r"]["shoulder"][0])),
                                       0.04 * H, 0.2 * H))
            # neck: narrowest central slice in the band above the shoulder line
            zz = np.linspace(shoulder_z + 0.015 * H, min(0.97 * H, shoulder_z + 0.2 * H), 40)
            ww = np.array([probe.central_half(z) or np.nan for z in zz])
            okw = np.isfinite(ww) & (ww > 0.008 * H)
            if okw.any():
                i = int(np.argmin(np.where(okw, ww, np.inf)))
                neck_z, neck_half = float(zz[i]), float(ww[i])
            t_pose_like = False
            wmax = torso_half

    # --- feet ------------------------------------------------------------
    feet = {}
    for side, s in (("l", 1), ("r", -1)):
        band = (np.abs(V[:, 2] - 0.065 * H) < 0.015 * H) & (s * V[:, 0] > 0.005 * H)
        if band.sum() < 5:
            band = (V[:, 2] < 0.1 * H) & (s * V[:, 0] > 0)
        ank = V[band].mean(axis=0)
        low = (V[:, 2] < 0.05 * H) & (s * V[:, 0] > 0.005 * H)
        toe_y = float(V[low, 1].min()) if low.any() else ank[1] - 0.06 * H
        feet[side] = {"ankle": [float(ank[0]), 0.065 * H, float(ank[1])],
                      "toe": [float(ank[0]), 0.015 * H, toe_y + 0.01 * H]}

    # --- crotch: lowest Z where a front->back ray at x=0 hits the body ---
    crotch_z, found = find_crotch(probe, shoulder_z)
    if not found:
        notes.append("crotch gap not found; using proportion (skirt / wide trousers?)")
        loose_lower = True
    else:
        loose_lower = crotch_z < 0.42 * shoulder_z  # gap only very low => robe/hakama
    hip_x = float(np.clip(0.55 * torso_half, 0.035 * H, 0.075 * H))
    hip_z = crotch_z + 0.04 * H

    # --- arms: straight line shoulder -> hand tip -------------------------
    t_pose = (hands["l"][2] > shoulder_z - 0.12 * H)
    for side, s in (("l", 1), ("r", -1)):
        if arm_anchored:
            sh = np.array([sweeps[side]["shoulder"][0], 0.0, shoulder_z])
        else:
            sh = np.array([s * 1.0 * torso_half, 0.0, shoulder_z - (0.0 if t_pose_like else 0.015 * H)])
        tip = np.array([hands[side][0], 0.0, hands[side][2]])
        wrist = sh + (tip - sh) * 0.80
        elbow = sh + (wrist - sh) * 0.5
        J[f"upperarm_{side}"] = [sh[0], sh[2]]
        J[f"lowerarm_{side}"] = [elbow[0], elbow[2]]
        J[f"hand_{side}"] = [wrist[0], wrist[2]]
        J[f"handtip_{side}"] = [tip[0], tip[2]]
        J[f"clavicle_{side}"] = [s * 0.12 * torso_half, neck_z - 0.5 * (neck_z - shoulder_z) - 0.01 * H]
        hip = np.array([s * hip_x, hip_z])
        ank = np.array([feet[side]["ankle"][0], feet[side]["ankle"][1]])
        J[f"thigh_{side}"] = list(hip)
        J[f"calf_{side}"] = list(hip + 0.5 * (ank - hip))
        J[f"foot_{side}"] = list(ank)
        J[f"_toe_{side}"] = [feet[side]["toe"][0], feet[side]["toe"][1]]
        J[f"_toe_y_{side}"] = feet[side]["toe"][2]
        J[f"_ankle_y_{side}"] = feet[side]["ankle"][2]

    J["pelvis"] = [0.0, hip_z + 0.01 * H]
    J["neck_01"] = [0.0, 0.5 * (neck_z + shoulder_z)]
    J["head"] = [0.0, neck_z + 0.01 * H]
    J["_top"] = [0.0, H]
    info = {
        "neck_z": neck_z, "neck_half": neck_half, "shoulder_z": shoulder_z,
        "torso_half": torso_half, "crotch_z": crotch_z, "t_pose": bool(t_pose),
        "shoulder_width_max": wmax, "arm_anchored": arm_anchored,
        "arm_sweeps": {k: {kk: vv for kk, vv in v.items()} for k, v in sweeps.items()},
        "loose_lower_garment": bool(loose_lower), "notes": notes,
    }
    return J, info


# MediaPipe Pose landmark indices
MP = {"shoulder": (11, 12), "elbow": (13, 14), "wrist": (15, 16), "hip": (23, 24),
      "knee": (25, 26), "ankle": (27, 28), "heel": (29, 30), "foot_index": (31, 32),
      "index": (19, 20)}


def apply_pose2d(J, kp, views, probe, min_vis=0.5):
    """Override geometric joints with confident 2D keypoints lifted to 3D."""
    f = views["front"]
    S, cx, cz = f["ortho_scale"], f["center_x"], f["center_z"]

    def to_world(lm):
        return [cx + (lm["x"] - 0.5) * S, cz + (0.5 - lm["y"]) * S]

    lms = kp["landmarks"]
    used = []
    mapping = [("shoulder", "upperarm"), ("elbow", "lowerarm"), ("wrist", "hand"),
               ("knee", "calf"), ("ankle", "foot")]
    for side_i, side in ((0, "l"), (1, "r")):
        for mp_name, bone in mapping:
            lm = lms[MP[mp_name][side_i]]
            if lm.get("visibility", 1.0) >= min_vis:
                J[f"{bone}_{side}"] = to_world(lm)
                used.append(f"{bone}_{side}")
        # hand tip: extend wrist->index by 1.6 (MediaPipe "index" is the knuckle)
        w, i = lms[MP["wrist"][side_i]], lms[MP["index"][side_i]]
        if min(w.get("visibility", 1), i.get("visibility", 1)) >= min_vis:
            W, I = np.array(to_world(w)), np.array(to_world(i))
            J[f"handtip_{side}"] = list(W + 1.6 * (I - W))
        # hips: MediaPipe hips sit on the iliac crest -> keep geometric Z,
        # take X from the keypoint (more reliable than the torso-width ratio)
        lm = lms[MP["hip"][side_i]]
        if lm.get("visibility", 1.0) >= min_vis:
            x = to_world(lm)[0]
            J[f"thigh_{side}"][0] = 0.8 * x
            used.append(f"thigh_{side}.x")
    # shoulders from MediaPipe sit on the deltoid surface; pull slightly inwards
    for side in ("l", "r"):
        if f"upperarm_{side}" in used:
            J[f"upperarm_{side}"][0] *= 0.92
    return used


def find_crotch(probe, shoulder_z, knee_z=None):
    """Lowest Z where a front->back ray at x=0 hits the body (gap between legs).
    Returns (crotch_z, found). Fallback when legs are hidden by a robe/skirt:
    with confident knees use knee + 0.36*(shoulder-knee) - this ratio holds for
    realistic (≈0.35) and chibi (≈0.38) bodies alike - else 0.59*shoulder."""
    H = probe.H
    crotch_z = None
    for z in np.linspace(0.06 * H, 0.72 * shoulder_z, 160):
        if probe.solid_at(0.0, z):
            crotch_z = float(z)
            break
    lo, hi = 0.35 * shoulder_z, 0.72 * shoulder_z
    if crotch_z is not None and lo <= crotch_z <= hi:
        return crotch_z, True
    if knee_z is not None and 0.03 * H < knee_z < 0.6 * shoulder_z:
        return float(knee_z + 0.36 * (shoulder_z - knee_z)), False
    return 0.59 * shoulder_z, False


def reconcile_vertical(J, info, kp, views, probe, min_vis=0.6):
    """Stylised (chibi / big-head) bodies break the width-profile assumptions
    (the head is wider than the torso, so "widest slice = shoulders" lands in
    the head). When confident 2D shoulders disagree with the geometric
    shoulder line by > 8 % H, rebuild the vertical structure (neck, head,
    clavicles, pelvis, hips) from the keypoints instead."""
    H = probe.H
    f = views["front"]
    S, cz = f["ortho_scale"], f["center_z"]
    lms = kp["landmarks"]

    def z_of(i):
        lm = lms[i]
        return cz + (0.5 - lm["y"]) * S if lm.get("visibility", 1.0) >= min_vis else None

    shz = [z_of(i) for i in MP["shoulder"]]
    if None in shz:
        return None
    sh_z = float(np.mean(shz))
    if abs(sh_z - info["shoulder_z"]) < 0.08 * H:
        return None
    nose = z_of(0)
    knees = [z for z in (z_of(i) for i in MP["knee"]) if z is not None]
    knee_z = float(np.mean(knees)) if knees else None
    notes = [f"stylised proportions: 2D shoulders at {sh_z:.3f} vs geometric "
             f"{info['shoulder_z']:.3f} -> vertical structure rebuilt from keypoints"]

    # neck = narrowest central slice between the shoulder line and the chin
    top = 0.5 * (sh_z + nose) if nose is not None and nose > sh_z + 0.04 * H else sh_z + 0.1 * H
    zs = np.linspace(sh_z + 0.01 * H, top, 30)
    w = np.array([probe.central_half(z) or np.nan for z in zs])
    ok = np.isfinite(w) & (w > 0.008 * H)
    if ok.any():
        i = int(np.argmin(np.where(ok, w, np.inf)))
        neck_z, neck_half = float(zs[i]), float(w[i])
    else:
        neck_z, neck_half = sh_z + 0.04 * H, info["neck_half"]
        notes.append("neck: no valid slice above the 2D shoulders; using shoulder + 4%H")

    crotch_z, found = find_crotch(probe, sh_z, knee_z)
    if not found:
        notes.append(f"crotch from proportion between knee and shoulder ({crotch_z:.3f})")
    hip_z = crotch_z + 0.04 * H
    for s in ("l", "r"):
        sgn = 1 if s == "l" else -1
        J[f"clavicle_{s}"] = [sgn * 0.12 * info["torso_half"], neck_z - 0.5 * (neck_z - sh_z) - 0.01 * H]
        J[f"thigh_{s}"][1] = hip_z
        hip, ank = np.array(J[f"thigh_{s}"]), np.array(J[f"foot_{s}"])
        if not (hip[1] > J[f"calf_{s}"][1] > ank[1]):
            J[f"calf_{s}"] = list(hip + 0.5 * (ank - hip))
    J["pelvis"] = [0.0, hip_z + 0.01 * H]
    J["neck_01"] = [0.0, 0.5 * (neck_z + sh_z)]
    J["head"] = [0.0, neck_z + 0.01 * H]
    info.update({"neck_z": neck_z, "neck_half": neck_half, "shoulder_z": sh_z,
                 "crotch_z": crotch_z, "stylised_proportions": True})
    # re-decide with the corrected shoulder line (the geometric pass may have
    # flagged "no crotch gap" only because it searched at the wrong height)
    info["loose_lower_garment"] = (not found) or crotch_z < 0.42 * sh_z
    info["notes"] = info.get("notes", []) + notes
    return notes


def symmetrise(J):
    for k in list(J):
        if k.endswith("_l") and k.replace("_l", "_r") in J and isinstance(J[k], list):
            kr = k[:-2] + "_r"
            l, r = J[k], J[kr]
            x = 0.5 * (abs(l[0]) + abs(r[0]))
            z = 0.5 * (l[1] + r[1])
            J[k], J[kr] = [x, z], [-x, z]
    for k in ("_toe_y", "_ankle_y"):
        v = 0.5 * (J[f"{k}_l"] + J[f"{k}_r"])
        J[f"{k}_l"] = J[f"{k}_r"] = v


def sanity(J, H):
    """Fix obviously broken limbs (order along the chain must be monotonic)."""
    issues = []
    for s in ("l", "r"):
        hip, knee, ank = J[f"thigh_{s}"], J[f"calf_{s}"], J[f"foot_{s}"]
        if not (hip[1] > knee[1] > ank[1]):
            J[f"calf_{s}"] = [0.5 * (hip[0] + ank[0]), 0.5 * (hip[1] + ank[1])]
            issues.append(f"knee_{s} re-centred")
        sh, el, wr = (np.array(J[f"upperarm_{s}"]), np.array(J[f"lowerarm_{s}"]),
                      np.array(J[f"hand_{s}"]))
        a, b = np.linalg.norm(el - sh), np.linalg.norm(wr - el)
        if min(a, b) < 0.03 * H or max(a, b) / max(min(a, b), 1e-6) > 1.8:
            J[f"lowerarm_{s}"] = list(sh + 0.5 * (wr - sh))
            issues.append(f"elbow_{s} re-centred")
    return issues


def lift_to_3d(J, probe):
    """Turn [x, z] joints into 3D points using mesh depth."""
    H = probe.H
    P = {}
    for k, v in J.items():
        if k.startswith("_"):
            continue
        y, ok = probe.depth_mid(v[0], v[1])
        P[k] = Vector((v[0], y, v[1]))
    # spine / head along the body centre line, biased slightly to the back
    for s in ("l", "r"):
        P[f"foot_{s}"].y = J[f"_ankle_y_{s}"]
        P[f"_toe_{s}"] = Vector((J[f"_toe_{s}"][0], J[f"_toe_y_{s}"], J[f"_toe_{s}"][1]))
        P[f"ball_{s}"] = P[f"foot_{s}"].lerp(P[f"_toe_{s}"], 0.62)
        P[f"ball_{s}"].z = 0.02 * H
        P[f"handtip_{s}"] = Vector((J[f"handtip_{s}"][0], P[f"hand_{s}"].y, J[f"handtip_{s}"][1]))
        # give knees / elbows a tiny bend so IK knows the bend direction
        P[f"calf_{s}"].y -= 0.008 * H
        P[f"lowerarm_{s}"].y += 0.006 * H
    for k in ("pelvis", "neck_01", "head"):
        y, _ = probe.depth_mid(0.0, J[k][1], bias=0.55)
        P[k] = Vector((0.0, y, J[k][1]))
    P["_top"] = Vector((0.0, P["head"].y, H))
    return P


# ==========================================================================
# 2. Skeleton (UE5 Mannequin naming)
# ==========================================================================
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def build_armature(P, H):
    arm_data = bpy.data.armatures.new("Armature")
    arm = bpy.data.objects.new("Armature", arm_data)  # "Armature" => UE skips it as root
    bpy.context.scene.collection.objects.link(arm)
    C.select_only([arm])
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones

    def bone(name, head, tail, parent=None, deform=True, connect=False):
        b = eb.new(name)
        b.head, b.tail = Vector(head), Vector(tail)
        if (b.tail - b.head).length < 1e-4:
            b.tail = b.head + Vector((0, 0, 0.01 * H))
        if parent:
            b.parent = eb[parent]
            b.use_connect = connect
        b.use_deform = deform
        return b

    bone("root", (0, 0, 0), (0, 0.1 * H, 0), deform=False)
    pelvis, neck1, head = P["pelvis"], P["neck_01"], P["head"]
    spine_top = neck1.copy()
    spine_pts = [pelvis.lerp(spine_top, t) for t in (0.05, 0.24, 0.43, 0.60, 0.76)]
    bone("pelvis", pelvis, spine_pts[0], "root")
    prev = "pelvis"
    for i in range(5):
        tail = spine_pts[i + 1] if i < 4 else spine_top
        bone(f"spine_0{i + 1}", spine_pts[i], tail, prev, connect=(i > 0))
        prev = f"spine_0{i + 1}"
    neck2 = neck1.lerp(head, 0.5)
    bone("neck_01", neck1, neck2, "spine_05", connect=True)
    bone("neck_02", neck2, head, "neck_01", connect=True)
    bone("head", head, P["_top"], "neck_02", connect=True)

    for s in ("l", "r"):
        sh, el, wr, tip = P[f"upperarm_{s}"], P[f"lowerarm_{s}"], P[f"hand_{s}"], P[f"handtip_{s}"]
        clav = Vector((P[f"clavicle_{s}"].x, spine_top.y, P[f"clavicle_{s}"].z))
        bone(f"clavicle_{s}", clav, sh, "spine_05")
        bone(f"upperarm_{s}", sh, el, f"clavicle_{s}", connect=True)
        bone(f"lowerarm_{s}", el, wr, f"upperarm_{s}", connect=True)
        hand_vec = tip - wr
        bone(f"hand_{s}", wr, wr + hand_vec * 0.45, f"lowerarm_{s}", connect=True)
        # procedural, non-deforming fingers (needed for template matching)
        hv = hand_vec.normalized()
        fwd = Vector((0, -1, 0))
        L = hand_vec.length
        for fi, fname in enumerate(FINGERS):
            spread = (fi - 2) * 0.10 * L
            if fname == "thumb":
                base = wr + hv * 0.15 * L + fwd * 0.12 * L
                d = (hv + fwd * 0.8).normalized()
                seg = 0.18 * L
                parent = f"hand_{s}"
                chain = ["01", "02", "03"]
            else:
                meta = wr + hv * 0.05 * L + fwd * (-spread * 0.6)
                base = wr + hv * 0.45 * L + fwd * (-spread)
                bone(f"{fname}_metacarpal_{s}", meta, base, f"hand_{s}", deform=False)
                d = hv
                seg = 0.17 * L
                parent = f"{fname}_metacarpal_{s}"
                chain = ["01", "02", "03"]
            p = base
            for c in chain:
                n = f"{fname}_{c}_{s}"
                bone(n, p, p + d * seg, parent, deform=False, connect=(c != "01"))
                parent, p = n, p + d * seg
        hip, knee, ank, ball, toe = (P[f"thigh_{s}"], P[f"calf_{s}"], P[f"foot_{s}"],
                                     P[f"ball_{s}"], P[f"_toe_{s}"])
        bone(f"thigh_{s}", hip, knee, "pelvis")
        bone(f"calf_{s}", knee, ank, f"thigh_{s}", connect=True)
        bone(f"foot_{s}", ank, ball, f"calf_{s}", connect=True)
        bone(f"ball_{s}", ball, toe, f"foot_{s}", connect=True)

    # Non-deforming IK helpers with Manny's names AND Manny's hierarchy
    # (root -> ik_foot_root -> ik_foot_l/r). The template's foot-IK Control Rig
    # (CR_Mannequin_FootIK, run by ABP_Unarmed in Play) uses them as full-body-IK
    # targets and evaluates them in its own (Manny) hierarchy, so they must sit under
    # ik_foot_root - parenting them to the feet put both targets at the origin and glued
    # the ankles together. ue_build bakes their animation to follow the feet.
    bone("ik_foot_root", (0, 0, 0), (0, 0.1 * H, 0), "root", deform=False)
    for s in ("l", "r"):
        ank, ball = P[f"foot_{s}"], P[f"ball_{s}"]
        bone(f"ik_foot_{s}", ank, ball, "ik_foot_root", deform=False)

    # consistent rolls: limbs point their local Z towards world -Y (front)
    for b in eb:
        try:
            b.align_roll(Vector((0, -1, 0)) if abs((b.tail - b.head).normalized().y) < 0.9
                         else Vector((0, 0, 1)))
        except Exception:
            pass
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm


# ==========================================================================
# 3. Skinning
# ==========================================================================
def deform_bone_names(arm):
    return [b.name for b in arm.data.bones if b.use_deform]


def weights_matrix(obj, names):
    idx = {obj.vertex_groups[n].index: i for i, n in enumerate(names) if n in obj.vertex_groups}
    W = np.zeros((len(obj.data.vertices), len(names)), dtype=np.float32)
    for v in obj.data.vertices:
        for g in v.groups:
            j = idx.get(g.group)
            if j is not None:
                W[v.index, j] = g.weight
    return W


def write_weights(obj, names, W):
    for n in names:
        if n in obj.vertex_groups:
            obj.vertex_groups.remove(obj.vertex_groups[n])
    groups = [obj.vertex_groups.new(name=n) for n in names]
    for j, g in enumerate(groups):
        nz = np.nonzero(W[:, j] > 1e-4)[0]
        # group verts by identical weights would be faster; per-vertex is fine at <=60k verts
        for i in nz:
            g.add([int(i)], float(W[i, j]), "REPLACE")


def bone_heat(mesh, arm):
    C.select_only([mesh, arm], active=arm)
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")


def proxy_heat(mesh, arm, H):
    """Bone heat on a watertight voxel proxy, then transfer weights back."""
    proxy = mesh.copy()
    proxy.data = mesh.data.copy()
    proxy.name = "WeightProxy"
    bpy.context.scene.collection.objects.link(proxy)
    for vg in list(proxy.vertex_groups):
        proxy.vertex_groups.remove(vg)
    rm = proxy.modifiers.new("Remesh", "REMESH")
    rm.mode = "VOXEL"
    rm.voxel_size = H / 140.0
    C.select_only([proxy])
    bpy.ops.object.modifier_apply(modifier=rm.name)
    if len(proxy.data.polygons) > 40000:
        dec = proxy.modifiers.new("Dec", "DECIMATE")
        dec.ratio = 40000 / len(proxy.data.polygons)
        bpy.ops.object.modifier_apply(modifier=dec.name)
    bone_heat(proxy, arm)
    # transfer
    for n in deform_bone_names(arm):
        if n not in mesh.vertex_groups:
            mesh.vertex_groups.new(name=n)
    dt = mesh.modifiers.new("WeightTransfer", "DATA_TRANSFER")
    dt.object = proxy
    dt.use_vert_data = True
    dt.data_types_verts = {"VGROUP_WEIGHTS"}
    dt.vert_mapping = "POLYINTERP_NEAREST"
    dt.layers_vgroup_select_src = "ALL"
    dt.layers_vgroup_select_dst = "NAME"
    C.select_only([mesh])
    bpy.ops.object.modifier_move_to_index(modifier=dt.name, index=0)
    bpy.ops.object.modifier_apply(modifier=dt.name)
    bpy.data.objects.remove(proxy, do_unlink=True)
    # parent with empty groups (weights already present)
    C.select_only([mesh, arm], active=arm)
    bpy.ops.object.parent_set(type="ARMATURE_NAME")


def fix_orphans(mesh, W):
    """Vertices with no weight copy weights from the nearest weighted vertex
    (accessories such as earrings or charms are separate islands)."""
    V = C.vertex_array(mesh)
    tot = W.sum(axis=1)
    bad = np.nonzero(tot < 1e-3)[0]
    good = np.nonzero(tot >= 1e-3)[0]
    if bad.size == 0 or good.size == 0:
        return 0
    kd = KDTree(good.size)
    for i, gi in enumerate(good):
        kd.insert(V[gi], i)
    kd.balance()
    for b in bad:
        _, i, _ = kd.find(V[b])
        W[b] = W[good[i]]
    return int(bad.size)


def prune_far_weights(mesh, W, names, arm, H):
    """Drop influences from bones that are far away compared with the
    vertex's nearest bone (proxy-heat / data-transfer occasionally gives head
    hair to a hand 0.4 m away). Keeps b if d_b <= max(3*d_min, d_min+0.15H) (gross errors only)."""
    V = C.vertex_array(mesh)
    mw = arm.matrix_world
    A = np.array([list(mw @ arm.data.bones[n].head_local) for n in names])
    B = np.array([list(mw @ arm.data.bones[n].tail_local) for n in names])
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    D = np.empty((len(V), len(names)))
    for j in range(len(names)):
        t = np.clip(((V - A[j]) @ AB[j]) / L2[j], 0, 1)
        D[:, j] = np.linalg.norm(V - (A[j] + t[:, None] * AB[j]), axis=1)
    dmin = D.min(1, keepdims=True)
    far = D > np.maximum(3.0 * dmin, dmin + 0.15 * H)
    hit = (far & (W > 0.05)).any(1)
    W[far] = 0.0
    empty = W.sum(1) < 1e-6
    W[empty, D[empty].argmin(1)] = 1.0
    return int(hit.sum())


def bind_head_hangers(mesh, W, names, P, H, neck_half, nb):
    """Earrings / hair strands hang from the head below the chin and next to
    the shoulders, so heat weighting hands them to clavicle/upperarm and they
    stretch when the arm moves. Flood-fill from the top of the head over mesh
    edges, never entering the neck column; everything reached below the chin
    is a head attachment and gets 100 % head weight."""
    if "head" not in names:
        return 0
    V = C.vertex_array(mesh)
    head_z = P["head"].z
    cap_z = head_z + 0.03 * H
    floor_z = min(P["clavicle_l"].z, P["upperarm_l"].z) - 0.06 * H
    col = 1.6 * max(neck_half, 0.02 * H)
    allowed = (V[:, 2] > cap_z) | ((V[:, 2] > floor_z) & (np.abs(V[:, 0]) > col))
    seen = np.zeros(len(V), bool)
    stack = list(np.nonzero(V[:, 2] > cap_z)[0])
    seen[stack] = True
    while stack:
        i = stack.pop()
        for j in nb[i]:
            if not seen[j] and allowed[j]:
                seen[j] = True
                stack.append(j)
    hang = seen & (V[:, 2] < head_z)
    near_hand = min(np.linalg.norm(V[seen] - np.array(P[f"hand_{s}"]), axis=1).min()
                    for s in ("l", "r")) if seen.any() else 1e9
    if hang.mean() > 0.05 or near_hand < 0.08 * H:  # leaked into body / arms raised
        return -int(hang.sum())
    # head cap: only head / neck may influence it (proxy heat once gave the
    # earlobes to hand_l); hanging parts below the chin follow the head only
    cap = seen & ~hang
    keep = [names.index(n) for n in ("head", "neck_01", "neck_02") if n in names]
    Wc = np.zeros_like(W[cap])
    Wc[:, keep] = W[cap][:, keep]
    lost = Wc.sum(1) < 1e-6
    Wc[lost, names.index("head")] = 1.0
    W[cap] = Wc
    W[hang] = 0.0
    W[hang, names.index("head")] = 1.0
    return int(hang.sum())


def rigid_islands(mesh, W, max_frac=0.03):
    """Small separate islands (charms, badges) move rigidly with the body point
    they hang from: every vertex copies the weights of the nearest main-body
    vertex to the island."""
    V = C.vertex_array(mesh)
    n = len(V)
    e = np.empty(len(mesh.data.edges) * 2, dtype=np.int64)
    mesh.data.edges.foreach_get("vertices", e)
    parent = np.arange(n)

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r
    for a, b in e.reshape(-1, 2):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    ids, counts = np.unique(roots, return_counts=True)
    main = ids[np.argmax(counts)]
    body = np.nonzero(roots == main)[0]
    kd = KDTree(body.size)
    for k, vi in enumerate(body):
        kd.insert(V[vi], k)
    kd.balance()
    fixed = 0
    for r, c in zip(ids, counts):
        if r == main or c > max_frac * n:
            continue
        isl = np.nonzero(roots == r)[0]
        # attachment point = body vertex closest to the island
        best = min((kd.find(V[i]) + (i,) for i in isl), key=lambda t: t[2])
        W[isl] = W[body[best[1]]]
        fixed += int(c)
    return fixed


def adjacency(mesh):
    n = len(mesh.data.vertices)
    e = np.empty(len(mesh.data.edges) * 2, dtype=np.int64)
    mesh.data.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    nbrs = [[] for _ in range(n)]
    for a, b in e:
        nbrs[a].append(b)
        nbrs[b].append(a)
    return nbrs


def smooth_region(mesh, W, names, mask, cols, iters=12, alpha=0.6):
    nb = adjacency(mesh)
    ci = [names.index(c) for c in cols if c in names]
    idx = np.nonzero(mask)[0]
    for _ in range(iters):
        Wn = W.copy()
        for i in idx:
            if nb[i]:
                avg = W[nb[i]][:, ci].mean(axis=0)
                Wn[i, ci] = (1 - alpha) * W[i, ci] + alpha * avg
        W = Wn
    return W


def limit_normalise(W, k=4):
    if W.shape[1] > k:
        part = np.argpartition(-W, k, axis=1)[:, k:]
        np.put_along_axis(W, part, 0.0, axis=1)
    s = W.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return W / s


def skin(mesh, arm, P, H, method, cloth, loose_lower_hint, neck_half=None, post=True):
    names = deform_bone_names(arm)
    tried = []

    def evaluate():
        W = weights_matrix(mesh, names)
        zero = float((W.sum(axis=1) < 1e-3).mean())
        empty = [n for j, n in enumerate(names) if W[:, j].max() < 1e-3]
        return W, zero, empty

    order = {"heat": ["heat"], "proxy_heat": ["proxy_heat"], "auto": ["heat", "proxy_heat"]}[method]
    W = zero = empty = None
    best = None
    for m in order:
        for vg in list(mesh.vertex_groups):
            mesh.vertex_groups.remove(vg)
        mesh.parent = None
        for mod in list(mesh.modifiers):
            mesh.modifiers.remove(mod)
        try:
            bone_heat(mesh, arm) if m == "heat" else proxy_heat(mesh, arm, H)
            W, zero, empty = evaluate()
        except Exception as e:  # pragma: no cover
            tried.append({"method": m, "error": str(e)})
            continue
        # spine/neck must receive weight; a few tiny bones may legitimately be empty
        critical = [n for n in empty if not n.startswith(("ball_", "clavicle_", "neck_02", "pelvis"))]
        tried.append({"method": m, "zero_weight_ratio": zero, "empty_bones": empty})
        score = (len(critical), zero)
        if best is None or score < best[0]:
            best = (score, W.copy(), m)
        if zero < 0.02 and not critical:
            break
    # keep the BEST attempt, not the last one: the voxel proxy can fail completely on
    # meshes where heat was only slightly short of the threshold (found by the benchmark)
    _, W, used = best
    orphans = fix_orphans(mesh, W)
    pruned = rigid = hangers = 0
    if post:
        pruned = prune_far_weights(mesh, W, names, arm, H)
        rigid = rigid_islands(mesh, W)
        hangers = bind_head_hangers(mesh, W, names, P, H, neck_half or 0.03 * H, adjacency(mesh))
    smoothed = False
    use_cloth = cloth == "on" or (cloth == "auto" and loose_lower_hint)
    if use_cloth:
        V = C.vertex_array(mesh)
        knee_z = 0.5 * (P["calf_l"].z + P["calf_r"].z)
        hip_z = P["pelvis"].z
        hip_x = abs(P["thigh_l"].x)
        mask = (V[:, 2] > knee_z - 0.05 * H) & (V[:, 2] < hip_z) & (np.abs(V[:, 0]) < 1.8 * hip_x)
        W = smooth_region(mesh, W, names, mask, ["pelvis", "thigh_l", "thigh_r"])
        smoothed = True
    W = limit_normalise(W, 4)
    write_weights(mesh, names, W)
    return {
        "attempts": tried, "method_used": used, "orphan_vertices_fixed": orphans,
        "rigid_island_vertices": rigid,
        "far_weight_vertices_pruned": pruned,
        "head_attachment_vertices": hangers,  # negative = flood leaked, skipped
        "cloth_smoothing": smoothed,
        "zero_weight_ratio_final": float((W.sum(axis=1) < 1e-3).mean()),
        "max_influences": int((W > 1e-4).sum(axis=1).max()),
        "verts_per_bone": {n: int((W[:, j] > 0.05).sum()) for j, n in enumerate(names)},
    }


# ==========================================================================
# 4. QA renders: rest + stress pose, front + side
# ==========================================================================
def world_rot_to_local(arm, bone_name, axis, deg):
    b = arm.data.bones[bone_name]
    R = mathutils.Matrix.Rotation(math.radians(deg), 3, axis)
    M = b.matrix_local.to_3x3()
    return (M.inverted() @ R @ M).to_quaternion()


def stress_pose(arm):
    pose = {
        "upperarm_l": ("Y", -55), "upperarm_r": ("Y", 55),
        "lowerarm_l": ("X", -50), "lowerarm_r": ("X", -50),
        "thigh_l": ("X", -45), "calf_l": ("X", 70),
        "thigh_r": ("X", 20), "spine_03": ("Z", 20), "head": ("Z", 30),
    }
    for bn, (ax, deg) in pose.items():
        pb = arm.pose.bones.get(bn)
        if pb:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = world_rot_to_local(arm, bn, ax, deg)


def clear_pose(arm):
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)


def bone_overlay(arm, cam_info, img_path, res):
    """Write bone segments in image pixels (bones.json) and, if PIL is available
    inside Blender, draw them on the front render. Blender's bundled Python has
    no PIL, so the pipeline side (steps.s_autorig) draws it from bones.json."""
    S, cx, cz = cam_info["ortho_scale"], cam_info["center_x"], cam_info["center_z"]

    def px(v):
        return [((v.x - cx) / S + 0.5) * res, (0.5 - (v.z - cz) / S) * res]

    segs = []
    for b in arm.data.bones:
        h, t = arm.matrix_world @ b.head_local, arm.matrix_world @ b.tail_local
        segs.append({"name": b.name, "deform": bool(b.use_deform), "head": px(h), "tail": px(t)})
    out = img_path.replace(".png", "_bones.png")
    C.write_json(img_path.replace(".png", "_bones.json"), {"image": img_path, "out": out, "bones": segs})
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return out  # drawn later by the pipeline
    draw_bone_overlay(img_path, segs, out, Image, ImageDraw)
    return out


def draw_bone_overlay(img_path, segs, out, Image, ImageDraw):
    im = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for s in segs:
        col = (255, 60, 60) if s["deform"] else (90, 160, 255)
        d.line([tuple(s["head"]), tuple(s["tail"])], fill=col, width=3 if s["deform"] else 1)
        x, y = s["head"]
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 220, 0))
    im.save(out)


def qa_renders(mesh, arm, outdir, res, samples, engine="CYCLES"):
    os.makedirs(outdir, exist_ok=True)
    C.setup_render(engine, res, samples, flat=False)
    outs = {}
    clear_pose(arm)
    bpy.context.view_layer.update()
    _, fi = C.front_camera_for(mesh, 1.15)
    outs["rest_front"] = C.render_to(os.path.join(outdir, "rest_front.png"))
    outs["rest_front_bones"] = bone_overlay(arm, fi, outs["rest_front"], res)
    stress_pose(arm)
    bpy.context.view_layer.update()
    C.front_camera_for(mesh, 1.15)
    outs["stress_front"] = C.render_to(os.path.join(outdir, "stress_front.png"))
    C.side_camera_for(mesh, 1.15)
    outs["stress_side"] = C.render_to(os.path.join(outdir, "stress_side.png"))
    clear_pose(arm)
    return outs


# ==========================================================================
# 5. Export
# ==========================================================================
def export_fbx(mesh, arm, path):
    """UE-friendly FBX: 1 unit = 1 cm, armature named 'Armature' (no extra root).

    Scene unit scale 0.01 and the model itself scaled x100 (applied), so both
    the mesh AND the bone translations are in centimetres with identity object
    transforms. If the armature kept a x100 scale instead (default export),
    the mesh imports at the right size but retargeted animations collapse or
    fly away (bone translations off by 100x).

    The mesh must be un-parented while applying the scale: applying scale on a
    selected parent + child scales the child twice (character came out
    17500 cm tall). Each object is applied on its own, then re-parented."""
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 0.01
    world = mesh.matrix_world.copy()
    mesh.parent = None
    mesh.matrix_world = world
    for o in (arm, mesh):
        o.scale = (100.0, 100.0, 100.0)
        C.select_only([o])
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mesh.parent = arm
    mesh.matrix_parent_inverse.identity()
    C.select_only([arm, mesh], active=arm)
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"ARMATURE", "MESH"},
        apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE", global_scale=1.0,
        add_leaf_bones=False, bake_anim=False, use_armature_deform_only=False,
        primary_bone_axis="Y", secondary_bone_axis="X", armature_nodetype="NULL",
        mesh_smooth_type="FACE", use_tspace=True, path_mode="COPY", embed_textures=False,
        axis_forward="-Z", axis_up="Y",
    )


# ==========================================================================
def main():
    a = parse()
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.inp))
    mesh = C.get_main_mesh()
    probe = MeshProbe(mesh)
    H = probe.H

    J, geo_info = estimate_geometric(probe, use_arm_anchor=bool(a.arm_anchor))
    used_pose2d = []
    method = a.joint_method
    if method in ("auto", "pose2d") and a.keypoints and os.path.exists(a.keypoints):
        kp = C.read_json(a.keypoints)
        if kp.get("ok"):
            views = C.read_json(a.views)
            J_geo = {k: list(v) if isinstance(v, list) else v for k, v in J.items()}
            used_pose2d = apply_pose2d(J, kp, views, probe)
            if used_pose2d and geo_info.get("arm_anchored") and method == "auto":
                # arbitration: the arm-anchored shoulder line is a strong geometric cue;
                # 2D shoulders far from it are a keypoint failure (MediaPipe puts the
                # shoulders at the jaw of big-headed characters), so drop the 2D arms
                z2d = 0.5 * (J["upperarm_l"][1] + J["upperarm_r"][1])
                if abs(z2d - geo_info["shoulder_z"]) > 0.06 * probe.H:
                    for s in ("l", "r"):
                        for b in ("upperarm", "lowerarm", "hand", "handtip"):
                            J[f"{b}_{s}"] = J_geo[f"{b}_{s}"]
                    used_pose2d = [u for u in used_pose2d if not u.startswith(("upperarm", "lowerarm", "hand"))]
                    geo_info["notes"].append(f"2D shoulders at {z2d:.3f} rejected: arm-anchored geometry "
                                             f"says {geo_info['shoulder_z']:.3f}")
                    geo_info["pose2d_arms_rejected"] = True
            if (used_pose2d and a.vertical_reconcile and not geo_info.get("pose2d_arms_rejected")
                    and reconcile_vertical(J, geo_info, kp, views, probe)):
                used_pose2d.append("vertical_structure")
    if method == "pose2d" and not used_pose2d:
        C.log("pose2d requested but no usable keypoints - geometric used")
    symmetrise(J)
    fixes = sanity(J, H)
    P = lift_to_3d(J, probe)

    arm = build_armature(P, H)
    skin_report = skin(mesh, arm, P, H, a.weight_method, a.cloth_smoothing,
                       geo_info["loose_lower_garment"], geo_info.get("neck_half"), post=bool(a.skin_post))

    os.makedirs(a.outdir, exist_ok=True)
    renders = {}
    if a.render_tests:
        renders = qa_renders(mesh, arm, os.path.join(a.outdir, "qa"), a.render_res, a.render_samples,
                             a.render_engine)
    blend_path = os.path.join(os.path.abspath(a.outdir), f"{a.name}_rigged.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    fbx_path = os.path.join(os.path.abspath(a.outdir), f"SK_{a.name}.fbx")
    export_fbx(mesh, arm, fbx_path)

    report = {
        "ok": True,
        "fbx": fbx_path,
        "rigged_blend": blend_path,
        "height_m": H,
        "joint_method": "pose2d+geometric" if used_pose2d else "geometric",
        "pose2d_joints_used": used_pose2d,
        "geometry_analysis": geo_info,
        "sanity_fixes": fixes,
        "joints": {k: [round(c, 4) for c in v] for k, v in P.items() if not k.startswith("_")},
        "bone_count": len(arm.data.bones),
        "deform_bone_count": len(deform_bone_names(arm)),
        "skinning": skin_report,
        "qa_renders": renders,
    }
    C.write_json(a.report, report)
    C.log("autorig done", report["joint_method"], skin_report["method_used"],
          "zero:", skin_report["zero_weight_ratio_final"])


if __name__ == "__main__":
    main()
