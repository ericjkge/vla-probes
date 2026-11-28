# Dataset v1.6

Same 504 pairs as v1.5 (same labels, splits, held-out set). The scenes change: every scene is our own minimal scene. Two text fixes:
LIBERO's `notebook` object is a laptop, so sentences, harm reasons and object fields say "laptop" (24 pairs); and in push_off pairs
("Push X off the cabinet" / "Place X on the cabinet") the harmless sentence no longer asks to place an object that is already there
(Place/Rest -> Leave, Put -> Keep, Set -> "Let X stay"; held-out pairs use held-out verbs: Steady, Inspect, Polish). No sentence is in two splits.

- Layout: LIBERO's plain `table` (pale table, grey wall), the layout of libero_spatial / libero_goal.
- At most one fixture, only if the sentence needs it (stove, microwave, cabinet), at the back corner of the table.
- Only the objects the sentence names, plus: the swapped object (implicit), a knife for cutting, a person's hand for hurt_hand,
  and 4 water balls in the mug for every sentence that pours water (more do not fit in a mug).
- Placement is direct, not random: objects stand in one row (two if too wide), spaced by their measured sizes; stacked objects are
  set on their support; 3 layout variants (row order / depth). An implicit pair's two scenes are identical except for the swapped
  object, at the same spot.

Build: `python ../make_v1_6.py` (scene contents -> `scenes.json`), then `MUJOCO_GL=cgl ../../.venv/bin/python ../place_v1_6.py --all`
(scene files -> `scenes/`, images -> `images/`, per pair -> `placed/`), then `python ../place_v1_6.py --finish` (-> `pairs.jsonl`).
The only automatic check is a support that is completely hidden; every pair is reviewed by eye and bad ones re-rendered with `--pairs`.
