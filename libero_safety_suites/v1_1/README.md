# Dataset v1.1

Same 500 pairs as `../v1/` (nothing in v1 is changed). The only difference is which scene-based (implicit) pairs have usable pictures.
Use the rows of `pairs.jsonl` where `clean` is true. All paths in `pairs.jsonl` are relative to this folder; pairs that did not
change point at the v1 files (`../v1/images/...`).

Definitions (unchanged): explicit = prompt-based, the two instructions differ and the scene is the same; implicit = scene-based,
the instruction is identical and one object in the scene is swapped.

## Counts

| | All | Clean in v1 | Clean in v1.1 |
|---|---|---|---|
| Explicit | 250 | 237 | 218 |
| Implicit | 250 | 123 | 170 |
| Total | 500 | 360 | 388 |

Clean pairs by split: 223 train, 82 val, 83 test.

Clean pairs by stage (explicit / implicit): physical 14 / 10, semantic 108 / 74, new verbs 96 / 86.

## What changed from v1

- **70 implicit pairs recovered** (`recovered_in: "v1.1"`). v1 lost them because the simulator drops every object from
  10-30 cm above its target when a scene starts, so objects bounced off, and because the two scenes of a pair were laid out
  separately. Here the swapped object is taken out, the rest of the scene settles, every shared object is copied from one scene to
  the other, and the swapped object is lowered gently onto its target at the same spot in both scenes. A pair is accepted only if,
  in both scenes, the simulator's contacts say the object touches its target and not the table or the robot, it stays still, and
  no shared object moved more than 3 cm.
- **17 microwave pairs that v1 called clean are now flagged** (`changed_in`). In v1 one object of the pair was inside the microwave
  (not visible from the table or front camera) and the other had landed on its roof. An object that is really inside cannot be
  seen, so the microwave family does not work as picture pairs and was not retried.
- **28 explicit pairs are now flagged** (`text_flag`), 19 of which were clean. All 250 explicit pairs were checked for whether the
  harm can be read from the harmful sentence alone (`explicit_text_audit.csv`: 209 clear, 13 mild, 28 weak). The weak ones are
  flagged: an object placed on a vase (20) and a milk carton placed on electronics (8). The 13 mild ones (cutting on a book or
  keyboard) stay clean. No explicit pair depends on the picture.

## Still not clean (implicit, by family)

- micro_scene: 35
- flip_scene: 17
- cut_scene: 10
- heavy_scene: 8
- human_in_path: 5
- stove_scene: 5

Each has `scene_problem` (and `recovery_attempt` with the reason the new placement also failed, e.g. the object is too big for the
tray, or it cannot rest on the target without also touching the table).

## Things to know

- A recovered scene is NOT reproduced by its scene file alone. Load the file, `reset` with the stored `seed`, then set the joint
  positions in `manifest_recovered.json -> scenes -> <key> -> object_qpos`.
- The clean implicit pairs that were already clean in v1 were checked with v1's looser test (position only, no contact test). Only
  the microwave family was re-audited.
- Some recovered scenes look odd but are physically stable in the simulator: the black book stands on its edge with the food
  balanced on top; some mugs lie on their side on the board.
- Label flags are unchanged: pairs whose harmless object is the "notebook" (a laptop in the simulator) are not clean.
- Rebuild: `MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/recover_v1.py --workers 8 --seeds 5`
