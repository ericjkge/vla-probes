# Dataset v1.4

Builds on `../v1_3/` (nothing in v1 to v1.3 is changed). **Use the rows of `pairs.jsonl` where `balanced` is true**: 210
explicit + 210 implicit pairs, the same number in every stage. Paths point at the older files (`../v1_3/...`, `../v1_2/...`, ...).
No new scenes or pictures.

## What changed from v1.3

- **The 25 medium-confidence implicit pairs are out of the balanced set.** These are all the `human_in_path` pairs (a hand in the
  robot's path); they are the only pairs with `label_confidence: "medium"`. 21 are in the new-verbs stage and 4 in the physical stage.
- **25 explicit pairs are out too**, so the two halves stay equal in every stage (21 new verbs, 4 physical). They are taken from the
  same splits as the removed implicit pairs, newest first, from the family that had the most pairs there:
  sp_0585 sp_0336 sp_0346 sp_0316 sp_0334 sp_0345 sp_0586 sp_0576 sp_0581 sp_0583 sp_0575 sp_0303 sp_0580 sp_0579 sp_0574 sp_0302
  sp_0319 sp_0389 sp_0301 sp_0352 sp_0376 sp_0002 sp_0007 sp_0012 sp_0004.
- All 50 rows stay in the file with `balanced: false` and a `balance_note`. Their `clean` field is unchanged, so they can still be
  used for a separate test (for example, probe results with and without the hand pairs).

## Counts (balanced set)

| | Physical | Semantic | New verbs | Total | Train / val / test |
|---|---|---|---|---|---|
| Explicit | 9 | 115 | 86 | 210 | 125 / 42 / 43 |
| Implicit | 9 | 115 | 86 | 210 | 130 / 41 / 39 |

Explicit families: push_off 35, destroy 34, stove_obj 34, hurt_hand 32, candle_obj 22, micro_obj 22, pour_onto 16, cut_on 12, heavy_on 3.

The physical stage is now small (9 + 9).

## Report and rebuild

`report.html` shows every pair with its pictures; pairs outside the balanced set are tagged "not in balanced set". If pictures do
not show, run `python3 -m http.server 8765 --directory libero_safety_suites` and open http://localhost:8765/v1_4/report.html.

Rebuild: `python libero_safety_suites/make_v1_4.py && python libero_safety_suites/make_report.py v1_4`.
