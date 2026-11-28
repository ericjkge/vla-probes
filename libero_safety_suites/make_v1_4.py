"""Dataset v1.4 = v1.3 without the 25 medium-confidence hand-in-path implicit pairs in the balanced set.

    python libero_safety_suites/make_v1_4.py && python libero_safety_suites/make_report.py v1_4

Nothing in v1_3 is changed. Rows are copied; only `balanced` (and `balance_note`) change, and paths are rewritten so they still
point at the v1.3 (or older) files. To keep explicit = implicit in every stage, the same number of explicit pairs is taken out
of the balanced set, in the same splits as the removed implicit pairs: per (stage, split), the newest pair of the family that
has the most pairs there.
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
SRC, OUT = HERE / "v1_3", HERE / "v1_4"
rows = [json.loads(l) for l in open(SRC / "pairs.jsonl")]


def fix(v):
    if isinstance(v, str) and v.startswith(("images/", "scenes/")):
        return "../v1_3/" + v
    if isinstance(v, dict):
        return {k: fix(x) for k, x in v.items()}
    return v


rows = [{k: fix(v) for k, v in r.items()} for r in rows]

hand = [r for r in rows if r.get("balanced") and r["family"] == "human_in_path"]
assert len(hand) == 25 and all(r["label_confidence"] == "medium" for r in hand)
for r in hand:
    r["balanced"] = False
    r["balance_note"] = "v1.4: medium-confidence hand-in-path pair, taken out of the balanced set"

need = Counter((r["stage_name"], r["split"]) for r in hand)
dropped = []
for (stage, split), n in sorted(need.items()):
    for _ in range(n):
        pool = [r for r in rows if r.get("balanced") and r["harm_type"] == "explicit" and r["stage_name"] == stage and r["split"] == split]
        fam = Counter(r["family"] for r in pool).most_common(1)[0][0]
        r = max((r for r in pool if r["family"] == fam), key=lambda r: r["pair_id"])
        r["balanced"] = False
        r["balance_note"] = "v1.4: taken out so explicit matches implicit after the hand-in-path pairs were removed"
        dropped.append(r["pair_id"])

bal = [r for r in rows if r.get("balanced")]
c = Counter((r["stage_name"], r["harm_type"]) for r in bal)
assert all(c[(k, "explicit")] == c[(k, "implicit")] for k, _ in c), c
assert not any(r["family"] == "human_in_path" for r in bal)

OUT.mkdir(exist_ok=True)
with open(OUT / "pairs.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("explicit dropped:", " ".join(dropped))
print("balanced:", Counter(r["harm_type"] for r in bal), Counter((r["stage_name"], r["harm_type"]) for r in bal))
print("splits:", Counter((r["harm_type"], r["split"]) for r in bal))
print("explicit families:", Counter(r["family"] for r in bal if r["harm_type"] == "explicit"))
