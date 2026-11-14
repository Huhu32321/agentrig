# Engineering log

Problems found while building and running the pipeline end-to-end (Windows 11, Blender 5.1,
UE 5.7), what caused them and how they were fixed. Several of them shaped the design: the
validators, the arm-anchored joint estimator and the benchmark exist because of entries below.

## Lessons that changed the design

| # | Symptom | Cause | Fix / lesson |
|---|---|---|---|
| L1 | On chibi characters the pelvis ended up in the chest and the neck inside the head (reproduced on every KayKit model of the benchmark), yet **every skinning metric was green and the agent declared success** | The width-profile estimator assumes "widest slice below the head = shoulders"; a chibi head is wider than the torso | Added a **skeleton-structure validator** (pelvis must be ≥15 % H below the shoulders, neck between shoulders and head). Lesson: metrics on the *output of a step* (skin weights) cannot detect an error made *earlier* (joint placement); validate structure explicitly. |
| L2 | The fix for L1 (rebuild the vertical structure from MediaPipe shoulders when they disagree with geometry) made **big-headed realistic characters worse** (held-out realistic models: 3.4 → 5.0 % H joint error on the benchmark) | MediaPipe places the shoulders at the jaw of large heads - the 2D model fails on the same characters as the geometry | **Arm-anchored shoulders**: sweep from each hand towards the body with vertical ray casts; the shoulder is where the arm's cross-section jumps into the torso. It does not depend on head size, so it can **arbitrate**: 2D shoulders far from it are rejected. Found only because of the benchmark - single-character testing had made the first fix look good. |
| L3 | Chibi models with a wizard hat: the "hands" were placed on the hat brim | "Hand = most lateral vertex" | Hands = the *lowest* strong peak of the lateral-extent profile (brims, pauldrons and capes sit above the hands). |
| L4 | Character's feet bend into the ground **only when playing** (WASD), never in the preview level | The template's anim blueprint runs a foot-IK Control Rig that reads Manny's `ik_foot_l/r` bones as full-body-IK targets; the generated skeleton had no such bones, so the targets fell back to the origin | Took two rounds. Adding `ik_foot_l/r` as children of the feet was **not enough**: the Control Rig evaluates them in *its own* (Mannequin) hierarchy, where they hang under `ik_foot_root`, so a foot-relative transform still lands at the origin - ankles glued together, no stride. Fix: Mannequin hierarchy (`root → ik_foot_root → ik_foot_l/r`) **and** bake the feet's root-relative transform into those bones in every retargeted animation (the retargeter leaves non-chain bones at the reference pose). A headless **Play-in-Editor test** (pawn driven forward, feet sampled) now guards this: max stride 0.4 cm before, 103 cm after (Mannequin: 104 cm). Lesson: "Mannequin-compatible" means matching the hierarchy *and animation* of the helper bones the runtime logic uses - and runtime behaviour needs a runtime test. |
| L5 | The LLM agent set `rig.weight_method = "geodesic"` (does not exist) | LLMs invent plausible parameter values | `set_config` validates enumerated keys and returns the legal list; the agent recovers on the next turn (seen in logs). |
| L6 | Without a vision model the agent kept calling `inspect_image` and tried to enable the VLM | The tool was always offered | Tools that can only fail are not offered; the prompt switches to reading quantitative metrics and listing images for a human to check. |
| L7 | The pipeline reported success while Idle collapsed into a ball and Walk flew 88 m up | FBX armature carried a ×100 scale: mesh in cm, bone translations in m | Apply scale to mesh and armature separately before export; **UE preview screenshot + sampled bone positions** became part of `ue_build`, and a validator compares Idle/Walk head height with the reference pose. |
| L9 | The "no post-processing" ablation left 6 of 41 rigs with **no weights at all** | In `auto` mode, heat that is slightly short of the threshold (2.9 % unweighted) triggers the voxel-proxy fallback; when the proxy fails completely the loop kept the *last* attempt instead of the best one. Post-processing had been silently repairing the damage | Keep the best attempt. Lesson: ablations are also bug finders - a component that "helps" may be masking a bug elsewhere. |
| L8 | Wide T-pose characters were tipped onto their side by clean-up | "Stand up if X is longer than Z" - a long-armed T-pose is wider than tall | A lying body is thinnest along Z; only then rotate. Found by the benchmark (a cartoon robot). |

## Geometry and skinning

| Symptom | Cause | Fix |
|---|---|---|
| Neck detected at the waist | Clothes and body are two shells; rays from the centre hit the inner layer | Rays walk through layers: hits closer than 2 % H are the same part |
| Low-poly meshes: arm tracking lost after a few slices | Thin vertex slices contain 0-2 vertices | Use vertical ray casts (inside intervals) instead of vertex slices |
| Bone heat fails on AI meshes | Non-manifold geometry, internal faces, loose props | Detect unweighted vertices → compute weights on a voxel-remeshed proxy and transfer back; loose islands copy the nearest body vertex |
| Robe / skirt tears between the legs | Hem split between left and right thigh | Detect "no gap between the legs" (rays) or VLM "loose lower garment" → Laplacian smoothing of pelvis/thigh weights in that region |
| Earrings / hanging hair stretched by a raised arm | Parts hanging from the head below the chin sit next to the shoulder; distance-based weights give them to the arm | Flood from the head top over mesh edges (not entering the neck ring); reached parts below the chin follow `head` 100 %; far-weight pruning removes influences > 3× the nearest bone distance |
| Character centred off its root | Clean-up centred the bounding box; long hair, capes or wide sleeves shift it | Centre on the feet (lowest 6 % of vertices) |
| Character completely black in UE | Web image-to-3D exports an unlit material: base colour black, texture on Emissive | Clean-up moves an emissive-only texture to Base Color (`emissive_only_converted` in the report) |

## Unreal Engine 5.7 automation

| Symptom | Cause | Fix |
|---|---|---|
| Commandlet crashes when importing textures (Slate assertion) | 5.7 import touches the content browser | Default `editor` mode: `UnrealEditor.exe -ExecutePythonScript -unattended -RenderOffscreen`, quits when done |
| FBX import options ignored | 5.5+ routes FBX through Interchange | Temporarily disable `Interchange.FeatureFlags.Import.FBX` |
| Retargeted anim blueprint / blend space references broken | `duplicate_and_retarget` writes into `/Game` and does not return referenced assets | Collect all new assets with the character suffix and move them in **one** batch rename (keeps their references) |
| Skeleton "unavailable" after reopening the project | Generated assets only in memory | `save_dirty_packages` at the end of the build |
| Preview screenshots blurry / washed out | No texture streaming in a windowless editor; sRGB render target encoded twice | Force full mips on the character textures for the capture only; linear RGBA8 target |
| Normal maps inverted | glTF is OpenGL (+Y), UE DirectX (−Y) | `flip_green_channel` on import |
| Script path not found | UE command line eats `\u`, `\t` in Windows paths | Pass forward-slash paths |

## Environment

| Symptom | Cause | Fix |
|---|---|---|
| Keys set with `setx` invisible in open terminals | Environment applies to new processes only | Fall back to the user environment in the registry |
| Garbled output in the web UI | Windows pipes default to a legacy code page | Entry point reconfigures stdout to UTF-8 |
| Background removal killed by OOM | BiRefNet on large images > 6 GB | `isnet-anime` by default, BiRefNet optional |
