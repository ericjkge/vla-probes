# Dataset v1.3

Builds on `../v1_2/` (nothing in v1, v1.1 or v1.2 is changed). **Use the rows of `pairs.jsonl` where `balanced` is true**: 235
explicit + 235 implicit pairs, the same number in every stage. Paths in `pairs.jsonl` are relative to this folder; pairs that did
not change point at the older files (`../v1/...`, `../v1_1/...`, `../v1_2/...`).

To look through the pairs with their pictures, open `report.html` (see "Report" below).

Definitions (unchanged): explicit = prompt-based, the two instructions differ and the scene is the same; implicit = scene-based,
the instruction is identical and one object in the scene is swapped.

## Counts

| | Pairs in the file | Clean in v1.2 | Clean in v1.3 | Balanced set |
|---|---|---|---|---|
| Explicit | 276 | 218 | 236 | 235 |
| Implicit | 310 | 231 | 235 | 235 |
| Total | 586 | 449 | 471 | 470 |

Balanced set by stage (same for explicit and implicit): physical 13, semantic 115, new verbs 107.

Balanced set by split: explicit 139 / 47 / 49, implicit 144 / 46 / 45 (train / val / test). The splits are close to 60/20/20
but not identical between the halves, because pairs that share a sentence must stay in one split.

One clean explicit pair (`sp_0010`, physical stage) is left out of the balanced set, because the implicit half has only 13
physical pairs. It has `balanced: false` and a `balance_note`.

## What changed from v1.2

- **Goal: the same number of explicit and implicit pairs.** v1.2 had 218 clean explicit and 231 clean implicit pairs.
- **Implicit:** 2 flagged pairs fixed by retrying with more starting layouts (`sp_0174`, `sp_0424`, `recovered_in: "v1.3"`) and
  2 new pairs (`added_in: "v1.3"`). That is all the repetition caps allow. Every unused candidate either uses a hazardous object
  already at its cap (40 per object, 20 for objects that only appear in semantic-suite scenes: egg, eyeglasses, game console,
  keyboard, notebook) or a verb at its cap (pour, flip). The implicit half is therefore 235, not 250.
- **Explicit:** 26 new pairs (`sp_0560` onwards) from the generator's unused candidates, with the same caps (per verb, per
  hazardous object, at most 2 uses of a sentence) and no explicit sentence in two splits. None are of the two weak-sentence kinds
  flagged in v1.1 ("put X on the vase", "milk carton on electronics"), and none use the "notebook" (rendered as a laptop) as the
  harmless object. They fill the explicit half up to the implicit count, stage by stage.
- **8 pairs flagged by eye** (`scene_problem`), because what the sentence names is not visible in the picture:
  - `sp_0085`, `sp_0347`, `sp_0348` (in v1.2, previously counted as clean), `sp_0567`, `sp_0568`, `sp_0569`: the candle is
    knocked over or out of view. The automatic checks only catch objects that leave the table, not ones that tip over. No
    new candle pairs were added after this.
  - `sp_0582`: the wine bottle is not visible. `sp_0584`: the frypan is not visible.

I looked at the pictures of every new and fixed pair.

## Things to know

- The most used hazardous objects among clean pairs are porcelain_mug (41; one over the cap because a fixed original pair uses it),
  black_book (40) and white_yellow_mug (40).
- Getting to 250 + 250 would mean raising the caps above, so that some hazardous objects (mostly egg, eyeglasses and the
  game console) appear more often.
- As in v1.2, a fixed or new implicit scene is not reproduced by its scene file alone: load the file, `reset` with the stored
  `seed`, then set the joint positions in `manifest.json -> scenes -> <key> -> object_qpos`.

## Report

`report.html` shows the counts and every pair as a card, with filters and a search box. Pairs outside the balanced set are tagged
"not in balanced set". If pictures do not show when you open the file directly, serve the folder:
`python3 -m http.server 8765 --directory libero_safety_suites` and open http://localhost:8765/v1_3/report.html.

## Rebuild

`MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/extend_v2.py --workers 8` (resumes), then
`python libero_safety_suites/make_report.py v1_3`.
