# v1.5 held-out test set

84 pairs whose objects or verbs appear **nowhere in v1.5** (nor in v1–v1.3), for measuring object- and verb-identity shortcuts.
Train on v1.5 (`../pairs.jsonl`, splits train/val), then report accuracy on the v1.5 test split (in-distribution) and on this set; the
gap is the shortcut. Every row has `split: "heldout"` and `heldout: "objects"` or `"verbs"`. Built by `../../make_heldout.py`.

| Pair type | Pairs | Held out | Families |
|---|---|---|---|
| Implicit (same sentence, one object swapped) | 42 | new objects on both sides | flip_scene 14, heavy_scene 14, pour_scene 14 |
| Explicit, object differs | 21 | new objects on both sides | pour_onto 8, micro_obj 6, heavy_on 7 |
| Explicit, verb differs | 21 | new verbs on both sides (+ a new object, same on both sides) | destroy 11, push_off 10 |

- **New objects:** simulator objects outside LIBERO-Safety that passed the stability screen (`libero_safety_ood/`), high-confidence
  labels only. Harmful: soy sauce / gin / sesame oil bottles, alarm clock, box of tissues, card terminal, comic book, cardstock,
  corkscrew, sardine and tomato tins. Harmless: cucumber, apricot, carrot, bell pepper, cupcake, slice of bread, sardine and tomato tins.
- **New verbs:** harmful pulverize, demolish, wreck, bash, fling, topple, flick; harmless raise, polish, inspect, steady, position,
  stand, arrange.
- Implicit and explicit-object sentences use benchmark verbs (flip, place, pour, put … turn it on), so only the objects are new there.
- The 4 fragile objects (3 glass bottles, alarm clock) carry most of the harmful side (57 of 84 pairs): they are the only screened
  new objects that are fragile. Report per-object results or a group bootstrap over objects.

## Checks

Same as v1.5 (placement, nothing falls off the table, implicit pairs from one seed with the swapped object lowered onto its target and
shared objects within 3 cm), plus a **visibility check** that v1–v1.5 do not have: each scene state is re-created and MuJoCo's
segmentation of the table view must show every object the sentence names (>= 120 px, centre >= 16 px inside the frame), and in an
implicit pair the two swapped objects must appear within 25 px of each other. Pairs failing it were replaced; pour-onto scenes were
re-rendered with other seeds until both objects were in view. "Push off" pairs say "off the table" (the added object stands on the
table, not on a cabinet). All 84 pairs' pictures were also inspected by eye.

## Files

`pairs.jsonl` (paths relative to this folder), `images/` (table view, wrist, front), `scenes/`, `manifest.json` (every rendered
candidate, its checks and pixel counts; images of rejected candidates were deleted), `report.html`. Implicit scenes are reproduced as
in v1.3: load the scene file, `reset` with the stored `seed`, then set `object_qpos`. To rebuild, delete `manifest.json` first.
