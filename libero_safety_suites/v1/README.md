# Dataset v1: 500 harmful / harmless minimal pairs with rendered scenes

Built from the LIBERO-Safety physical suites (affordance, human_safety, obstacle_avoidance, obstacle_avoidance_human) and its
semantic suite (reasoning_safety). Text pairs come from `../generate_suite_pairs.py`; scenes and images from `../render_v1.py`.

## What a pair is

One harmful and one harmless version of the same situation, differing in exactly one thing.

- **Explicit (prompt-based), 250 pairs:** same scene, two different sentences (one verb or one object differs, column `differs`).
- **Implicit (scene-based), 250 pairs:** identical sentence, two scenes that differ by one object in the same place.

Stages, filled in this order: 1 physical (28 pairs), 2 semantic (258), 3 new verbs that are not in the benchmark (214). Each stage is
half explicit, half implicit. Splits are about 60/20/20 inside every hazard family; an explicit sentence never appears in two splits.

## Use `clean == true`

| | all | scenes OK | clean |
|---|---|---|---|
| explicit | 250 | 247 | 237 |
| implicit | 250 | 131 | 123 |
| total | 500 | 378 | 360 |

Clean pairs by split: train 200, val 80, test 80. Nothing was dropped: the other 140 pairs are in the files with the reason.

- `scene_ok`: both scenes rendered; every added object landed where intended; nothing fell off the table that does not also fall in
  the unmodified benchmark scene. For implicit pairs also: both scenes come from the same simulator seed, every object the two scenes
  share sits within 3 cm of the same spot, and the two table-view images visibly differ. `scene_problem` lists what failed.
- `label_flag`: 26 pairs use the "notebook" as the harmless object. In the simulator the notebook is a laptop, so putting weight on it
  or tipping it off a tray is not clearly harmless.
- `clean` = `scene_ok` and no `label_flag`.

Why implicit pairs fail most often: when the swapped object has a different size, the simulator re-places the other objects, so for
62 pairs no seed gave two matching layouts; in 41 an object would not stay on its target (rolls off a plate, burner or book).

## Files

- `pairs.jsonl`: one line per pair. Sentences (`harmful`, `harmless`), `harm_type`, `differs`, `stage`, `family`, `split`, the hazardous
  and harmless object, `harm_reason`, and per side (`harmful_...`, `harmless_...`): `scene_key`, `bddl`, `images`
  (agentview = table view, wrist, front), `robot_state`. Implicit pairs also have `image_diff_frac` (share of table-view pixels that differ).
- `robot_state.pi05_state`: the 8 numbers pi0.5 takes as state (end-effector position, axis-angle orientation, 2 gripper joints), read
  after the same reset and 20 settling steps as the image. Also `eef_pos`, `eef_quat_xyzw`, `gripper_qpos`, `joint_pos`.
- `images/`: 256 x 256 PNG, right way up. pi0.5 was trained on images rotated 180 degrees (see openpi's LIBERO example).
- `scenes/`: one BDDL file per scene (benchmark scene + added objects / the swapped object).
- `manifest.json`: per scene: seed, placement checks, objects that fell. Some entries describe superseded renders whose images were not
  kept; `pairs.jsonl` is the source of truth for which image belongs to which pair.
- `dataset_v1.xlsx`: the pairs for reading, plus a "Render status" sheet.

## Known limits

- Labels come from hand-written safety rules and have not been reviewed by a person. 38 are medium confidence (mostly a person's hand
  in the robot's path, where harm depends on how the robot moves).
- Some harm reasons describe the notebook as paper; it is a laptop. The harmful labels still hold (fire, water, cutting).
- An object inside the microwave is small in the table view (often under 1% of pixels).
- The test split is not a hard test: almost every hazardous object in it also appears in training.
- Explicit pairs with a violent verb (102) can be told apart from the words alone; compare probes against a word-count baseline.

## Rebuild

    python libero_safety_suites/generate_suite_pairs.py
    MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/render_v1.py --workers 6        # render every scene
    MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/render_v1.py --states           # robot state + what falls in the benchmark's own scenes
    MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/render_v1.py --match-seeds      # implicit pairs: both scenes from one seed, same layout
    MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/render_v1.py --finish           # write pairs.jsonl and the Excel file
