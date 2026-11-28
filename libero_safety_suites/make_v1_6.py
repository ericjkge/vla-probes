"""Dataset v1.6: the same 504 pairs as v1.5 (same text, labels, splits), with our own minimal scenes. This file decides WHAT is in each
scene; place_v1_6.py decides where it goes and renders it.

Every scene: LIBERO's plain `table` layout (pale table, grey wall), at most ONE fixture the sentence needs (stove, microwave, cabinet),
and ONLY the objects the sentence names (+ the swapped object of an implicit pair, the knife for cutting, a person's hand for
hurt_hand, water balls in the mug whenever a sentence pours water). No distractors.

    python libero_safety_suites/make_v1_6.py     # -> v1_6/scenes.json: per pair its fixture, layout variant and objects per side,
                                                 #    each with where it goes ('table', ['on', support], ['fixture', site])
"""
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import make_heldout as MH   # noqa: E402  (adds the held-out objects to the lexicon)
G = MH.G

SRC, OUT = HERE / "v1_5", HERE / "v1_6"
FIXTURE_KEYS = {"flat_stove": "flat_stove", "microwave": "microwave", "wooden_cabinet": "wooden_cabinet", "white_cabinet": "wooden_cabinet"}
IGNORE = {"water_ball", "table", "kitchen_table", "living_room_table", "study_table"}   # named in text but not objects we place

PHRASES = sorted(((p, k) for k, v in G.LEX.items() for p in v["phrases"]), key=lambda x: -len(x[0]))


def named(text):
    """lexicon keys named in a sentence, longest phrase first so 'porcelain plate' is not also read as 'plate'."""
    t, out = " " + text.lower() + " ", []
    for p, k in PHRASES:
        if re.search(rf"\b{re.escape(p)}\b", t):
            out.append(k); t = re.sub(rf"\b{re.escape(p)}\b", " ", t)
    return out


def scene_plan(r):
    """-> dict(fixture=key|None, sides={harmful: [(obj, place)], harmless: [...]}); place = 'table' | ('on', obj) | ('fixture', sub)."""
    fam, h, s = r["family"], r["hazard_object"], r["safe_object"]
    words = named(r["harmful"]) + [w for w in named(r["harmless"]) if w not in named(r["harmful"])]
    fixture = next((FIXTURE_KEYS[w] for w in words if w in FIXTURE_KEYS), None)
    objs = [w for w in dict.fromkeys(words) if w not in FIXTURE_KEYS and w not in IGNORE]
    water = "water" in (r["harmful"] + r["harmless"]).lower()
    mug = next((o for o in objs if o in G.MUGS or "mug" in G.ph(o)), None)
    if water and not mug: raise ValueError("pours water but names no mug")

    def base(extra):
        """objects on the table in importance order: extra (the pair's key objects) first, then the other named ones."""
        return list(dict.fromkeys([o for o in extra if o] + objs))

    if r["harm_type"] == "explicit":
        if fam == "hurt_hand": items = [(o, "table") for o in base([h, "left_hand"])]
        elif fam == "cut_on": items = [(o, "table") for o in base([h, s])] + [("knife", "table")]   # cutting needs the knife
        elif fam == "push_off" and fixture == "wooden_cabinet": items = [(h, ("fixture", "top_side"))]
        else: items = [(o, "table") for o in base([h, s])]
        sides = {"harmful": items, "harmless": items}
    else:
        swap = {"harmful": h, "harmless": s}
        sides = {}
        for sd, o in swap.items():
            rest = [x for x in objs if x not in (h, s)]
            if fam == "stove_scene": items = [(o, ("fixture", "cook_region"))]
            elif fam == "stab_scene": items = [(o, "table"), ("knife", "table")]
            elif fam == "cut_scene":   # the food lies on the swapped support (book / board / plate)
                food = next(x for x in rest if "cuttable" in G.props(x) or "food" in G.props(x))
                items = [(o, "table"), (food, ("on", o))] + [("knife", "table")] + [(x, "table") for x in rest if x not in (food, "knife")]
            else:                      # heavy / flip / pour: the swapped object lies on the tray named in the sentence
                tray = next(x for x in rest if G.ph(x) in r["harmful"].lower().split(" onto the ")[-1].split(" on the ")[-1]
                            or x in ("plate", "cutting_board__1", "salad_porcelain_plate", "porcelain_plate"))
                items = [(tray, "table"), (o, ("on", tray))] + [(x, "table") for x in rest if x != tray]
            sides[sd] = items
    for sd in sides:
        if water: sides[sd] = list(sides[sd]) + [("water_ball", ("on", mug))]
        if fam == "stab_scene" or not any(o == "knife" for o, _ in sides[sd]):
            pass
    return dict(fixture=fixture, sides=sides, water=water)


def variant_of(r):
    return int(hashlib.sha1(r["pair_id"].encode()).hexdigest(), 16) % 3   # layout variant (place_v1_6.layout)


def main():
    rows = [json.loads(l) for l in open(SRC / "pairs.jsonl")]
    man, errs, kinds = {}, [], Counter()
    for r in rows:
        try:
            plan = scene_plan(r)
            man[r["pair_id"]] = dict(variant=variant_of(r), fixture=plan["fixture"], water=plan["water"],
                                     objects={sd: [[o, p if isinstance(p, str) else list(p)] for o, p in plan["sides"][sd]] for sd in ("harmful", "harmless")})
            kinds[(r["harm_type"], plan["fixture"] or "-", "water" if plan["water"] else "")] += 1
        except Exception as ex:
            errs.append(f"{r['pair_id']} {r['family']} {r['harmful']!r}: {type(ex).__name__}: {ex}")
    (OUT / "scenes.json").write_text(json.dumps(man, indent=1))
    print(f"scene contents for {len(man)} of {len(rows)} pairs")
    print("by (type, fixture, water):", dict(kinds))
    print("variants:", dict(Counter(m["variant"] for m in man.values())))
    if errs: print(f"{len(errs)} pairs could not be built:"); [print("  ", e) for e in errs]


if __name__ == "__main__":
    main()
