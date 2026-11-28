# Dataset v1.5

**`pairs.jsonl`: 504 pairs, every row is used.** 420 pairs (210 explicit + 210 implicit) split train / val / test, plus 84 held-out
pairs (`split: "heldout"`) whose objects or verbs appear nowhere else in the dataset. Built from v1.3 by `../make_v1_5.py`;
the held-out pairs by `../make_heldout.py`. Built independently of v1.4 (which keeps every v1.3 field and flagged rows). Browse everything in `report.html` (filter Split = heldout for the held-out set).

| | train | val | test | heldout |
|---|---|---|---|---|
| Explicit (the two sentences differ, same scene) | 126 | 42 | 42 | 42 |
| Implicit (same sentence, one object swapped in the scene) | 126 | 42 | 42 | 42 |

Train / val / test have the same counts per stage in both halves (physical 5/2/2, semantic 69/23/23, new verbs 52/17/17); no sentence
is in two splits.

## Fields

`pair_id`, `split`, `harm_type` (explicit / implicit), `differs` (verb / object / scene), `stage` (physical, semantic, new verbs,
held-out), `family`, `harmful` / `harmless` (sentences), `hazard_object`, `safe_object`, `harm_reason`, `group` (family + hazardous
object), `harmful_images` / `harmless_images` (agentview, wrist, front; paths relative to this folder).

## What changed from v1.3

- Only the v1.3 balanced pairs are kept, minus all medium-confidence pairs (a person's hand in the robot's path: harm depends on how
  the robot moves).
- 25 explicit pairs deleted so both halves have 210 (those with the most-used verbs).
- Pairs moved between splits (whole groups of pairs that share a sentence) so explicit and implicit have identical split counts.
- Held-out test set added (`heldout/README.md`): 42 implicit + 21 explicit pairs with new objects, 21 explicit pairs with new verbs,
  to measure object- and verb-identity shortcuts. Its scenes also passed a visibility check (every named object in view).
- Bookkeeping fields (clean, balanced, flags, scene keys, robot state) dropped; they are still in v1.3 and `heldout/manifest.json`.
