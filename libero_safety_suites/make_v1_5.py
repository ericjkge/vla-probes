"""Build libero_safety_suites/v1_5/ from v1_3 (v1_3 is not changed).

v1.5 step 1: medium-confidence pairs are deleted (the implicit `human_in_path` family: a person's hand in the robot's path, where
harm depends on how the robot moves, not on the scene).

    python libero_safety_suites/make_v1_5.py && python libero_safety_suites/make_report.py v1_5
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
SRC, DST = HERE / "v1_3", HERE / "v1_5"


def rebase(p):
    """paths in v1_3/pairs.jsonl are relative to v1_3/; make them relative to v1_5/."""
    if p is None or p.startswith("../"): return p
    return "../v1_3/" + p


def resplit(rows, seed=0, restarts=200):
    """step 3: move whole split groups (`split_group`: pairs that share a sentence, so no explicit sentence / implicit sentence +
    hazardous object ends up in two splits) between train / val / test until, in every stage, explicit and implicit have the same
    train / val / test counts (target ~60/20/20 of the stage). Family shares per split are kept close to the stage-wide share."""
    import random
    rng = random.Random(seed)
    B = [r for r in rows if r.get("balanced")]
    SPL = ("train", "val", "test")
    stages = sorted({r["stage"] for r in B})
    n_st = Counter(r["stage"] for r in B if r["harm_type"] == "explicit")
    target = {}
    for st in stages:
        n = n_st[st]; v = t = round(n * 0.2); target[st] = {"train": n - v - t, "val": v, "test": t}
    for h in ("explicit", "implicit"):
        X = [r for r in B if r["harm_type"] == h]
        groups = {}
        for r in X: groups.setdefault(r["split_group"], []).append(r)
        G = list(groups)
        fam_tot = Counter(r["family"] for r in X)
        def cost(asg):
            c = Counter(); f = Counter()
            for g, s in asg.items():
                for r in groups[g]: c[(r["stage"], s)] += 1; f[(r["family"], s)] += 1
            dev = sum(abs(c[(st, s)] - target[st][s]) for st in stages for s in SPL)
            fam = sum(abs(f[(fa, s)] - fam_tot[fa] * sum(target[st][s] for st in stages) / len(X)) for fa in fam_tot for s in SPL)
            return dev * 100 + fam
        best = None
        for k in range(restarts):
            asg = {g: groups[g][0]["split"] for g in G}
            if k: asg = {g: (rng.choice(SPL) if rng.random() < 0.3 else s) for g, s in asg.items()}
            cur = cost(asg); improved = True
            while improved:
                improved = False
                for g in rng.sample(G, len(G)):
                    for s in SPL:
                        if s == asg[g]: continue
                        old = asg[g]; asg[g] = s; c = cost(asg)
                        if c < cur: cur = c; improved = True
                        else: asg[g] = old
            if best is None or cur < best[0]: best = (cur, dict(asg))
            if best[0] < 100 and k > 20: break
        print(f"resplit {h}: stage/split deviation {best[0] // 100}, family term {best[0] % 100:.1f}")
        for g, s in best[1].items():
            for r in rows:
                if r["split_group"] == g: r["split"] = s
    # checks: no explicit sentence in two splits, no implicit sentence + hazardous object in two splits
    sp = {}
    for r in rows:
        keys = [("e", r["harmful"]), ("e", r["harmless"])] if r["harm_type"] == "explicit" else [("i", r["harmful"], r["hazard_object"])]
        for k in keys: sp.setdefault(k, set()).add(r["split"])
    bad = [k for k, v in sp.items() if len(v) > 1]
    assert not bad, f"sentence in two splits: {bad[:3]}"


def main():
    rows = [json.loads(l) for l in open(SRC / "pairs.jsonl")]
    n_med = sum(r["label_confidence"] != "high" for r in rows)
    rows = [r for r in rows if r["label_confidence"] == "high"]
    for r in rows:
        for k in ("harmful_bddl", "harmless_bddl"): r[k] = rebase(r[k])
        for k in ("harmful_images", "harmless_images"):
            if isinstance(r.get(k), dict): r[k] = {v: rebase(p) for v, p in r[k].items()}
    # step 2: delete explicit balanced pairs until every stage has as many explicit as implicit pairs. Drop pairs whose main verb is
    # used most (weakens the verb shortcut), taking from train/val/test in proportion to their share of that stage's explicit pairs.
    drop = set()
    vcount = Counter(r["main_verb"] for r in rows if r.get("balanced") and r["harm_type"] == "explicit")
    for st in sorted({r["stage"] for r in rows}):
        E = [r for r in rows if r.get("balanced") and r["harm_type"] == "explicit" and r["stage"] == st]
        n = len(E) - sum(r.get("balanced") and r["harm_type"] == "implicit" and r["stage"] == st for r in rows)
        if n <= 0: continue
        by_split = Counter(r["split"] for r in E)
        quota = {s: round(n * k / len(E)) for s, k in by_split.items()}
        quota[max(by_split, key=by_split.get)] += n - sum(quota.values())
        for r in sorted(E, key=lambda r: (-vcount[r["main_verb"]], r["pair_id"])):
            if quota[r["split"]] > 0:
                drop.add(r["pair_id"]); quota[r["split"]] -= 1; vcount[r["main_verb"]] -= 1
    rows = [r for r in rows if r["pair_id"] not in drop]
    print("deleted explicit to balance:", len(drop))
    resplit(rows)
    # step 5: one file with only the pairs to use (the balanced set + the held-out test set) and only the fields the probes need
    U = [slim(r) for r in rows if r.get("balanced")]
    H = HERE / "v1_5" / "heldout" / "pairs.jsonl"
    held = [slim(json.loads(l), prefix="heldout/") for l in open(H)] if H.exists() else []
    DST.mkdir(exist_ok=True)
    (DST / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in U + held))
    print("deleted medium-confidence:", n_med, "| pairs:", len(U), dict(Counter(r["harm_type"] for r in U)), "+ held-out", len(held))
    print("by stage:", dict(Counter((r["harm_type"], r["stage"]) for r in U)))
    print("by split:", dict(Counter((r["harm_type"], r["split"]) for r in U + held)))


FIELDS = ("pair_id", "split", "harm_type", "differs", "stage", "family", "harmful", "harmless", "hazard_object", "safe_object",
          "harm_reason", "group", "harmful_images", "harmless_images")


def slim(r, prefix=""):
    """Only what the probes need. Images: {agentview, wrist, front} -> path relative to v1_5/."""
    r = dict(r, stage=r["stage_name"])
    for sd in ("harmful", "harmless"):
        r[f"{sd}_images"] = {c: prefix + p for c, p in r[f"{sd}_images"].items()}
    return {k: r.get(k) for k in FIELDS}


if __name__ == "__main__":
    main()
