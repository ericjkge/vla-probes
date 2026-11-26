# Dataset v1.2

Builds on `../v1_1/` (nothing in v1 or v1.1 is changed). Use the rows of `pairs.jsonl` where `clean` is true. Paths in `pairs.jsonl`
are relative to this folder; pairs that did not change point at the v1 or v1.1 files (`../v1/...`, `../v1_1/...`).

To look through the pairs with their pictures, open `report.html` in a browser from inside the repo (see "Report" below).

Definitions (unchanged): explicit = prompt-based, the two instructions differ and the scene is the same; implicit = scene-based,
the instruction is identical and one object in the scene is swapped.

## Counts

| | Pairs in the file | Clean in v1.1 | Clean in v1.2 |
|---|---|---|---|
| Explicit | 250 | 218 | 218 |
| Implicit | 308 | 170 | 231 |
| Total | 558 | 388 | 449 |

Clean pairs by split: 266 train, 94 val, 89 test.

Clean pairs by stage (explicit / implicit): physical 14 / 13, semantic 108 / 114, new verbs 96 / 104.

Clean implicit pairs by family: flip_scene 76, stove_scene 40, pour_scene 29, heavy_scene 29, cut_scene 26, human_in_path 25, stab_scene 6.

## What changed from v1.1

- **4 flagged implicit pairs fixed** by retrying with new starting layouts (`recovered_in: "v1.2"`). Most of the 22 retried
  pairs still failed: the object will not rest on its target, or the rest of the scene settles differently in the two pictures.
- **58 new implicit pairs** (`sp_0501` onwards, `added_in: "v1.2"`) replace pairs that cannot be fixed. They are candidates
  the original generator (`generate_suite_pairs.py`) built but did not select, so they follow the same rules: same vocabulary, one
  swapped object, the same caps per verb and per hazardous object, no sentence + object combination used twice. Each one has the
  stage of a pair it replaces. By family: heavy_scene 20, flip_scene 20, cut_scene 11, pour_scene 6, stove_scene 1.
- **Not replaced on purpose:** no new microwave pairs (an object inside the microwave cannot be seen from the cameras), and no new
  pairs with the "notebook" (a laptop in the simulator) as the harmless object.
- The flagged pairs are still in the file with their reason, so nothing disappears between versions.

All new and fixed pairs went through the same placement and checks as v1.1 (`recover_v1.py`): the swapped object is lowered onto
its target at the same spot in both scenes, must rest on the target (simulator contacts) and stay still, nothing shared may move
more than 3 cm, and the two pictures must visibly differ. I looked at the pictures of all 62 new and fixed pairs.

## Still short

The implicit half has 231 clean pairs, 19 short of 250 (mostly the semantic stage). The generator's unused candidates
for that stage ran out under the repetition caps. Still not clean (implicit, by family): micro_scene 35, flip_scene 15, cut_scene 9, heavy_scene 8, human_in_path 5, stove_scene 5.

## Things to know

- The new pairs lean on a few objects: the most used hazardous objects among clean pairs are black_book (40), porcelain_mug (40), white_yellow_mug (40); the
  cap is 40 per object.
- A fixed or new scene is NOT reproduced by its scene file alone: load the file, `reset` with the stored `seed`, then set the joint
  positions in `manifest.json -> scenes -> <key> -> object_qpos`.
- Some scenes look odd but are stable in the simulator (a book standing on its edge with food balanced on top; mugs on their side).
- Label and text flags are unchanged from v1.1.

## Report

`report.html` shows the counts and every pair as a card (sentences, why it is harmful, what changes in the scene, pictures, flag
reason), with filters for type, status, stage, family and split and a search box. Browsers block pictures in local files opened
directly in some setups; if pictures do not show, serve the folder: `python3 -m http.server 8765 --directory libero_safety_suites`
and open http://localhost:8765/v1_2/report.html.

## Rebuild

`MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/extend_v1.py --workers 8` (resumes), then
`python libero_safety_suites/make_report.py v1_2`.
