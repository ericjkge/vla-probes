"""Dataset v1.2: bring the scene-based (implicit) half back towards 250 usable pairs. Reads v1/ and v1_1/, writes ONLY to v1_2/.

Two steps:
  1. retry    implicit pairs whose pictures failed in v1.1 are tried again with new starting layouts (new simulator seeds).
  2. replace  pairs that cannot be fixed (the microwave family: an object inside the oven cannot be seen; pairs whose harmless object
              is the "notebook", which renders as a laptop; pairs that still fail) are replaced by NEW pairs. The new pairs are
              candidates the generator (generate_suite_pairs.py) built but did not select, so they follow the same rules: same
              vocabulary, one swapped object, the same repetition caps (per verb, per hazardous object), no sentence + object pair
              used twice. Each replacement has the stage of a pair it replaces, so every stage keeps its size. No new microwave
              pairs and no "notebook" on the harmless side.
Every new or retried pair goes through the same placement and checks as v1.1 (recover_v1.py): the swapped object is lowered onto its
target at the same spot in both scenes and must rest there, with nothing shared moving more than 3 cm, and the two pictures must differ.

    MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/extend_v1.py --workers 8      # resumes
    MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/extend_v1.py --finish         # only rewrite v1_2/pairs.jsonl
"""
import argparse, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import render_v1 as V            # noqa: E402
import recover_v1 as RV          # noqa: E402
G = V.G

V1, V11, OUT = V.V1, HERE / "v1_1", HERE / "v1_2"
SCN, IMG, MAN = OUT / "scenes", OUT / "images", OUT / "manifest.json"
FAMS = ("stove_scene", "cut_scene", "heavy_scene", "flip_scene", "pour_scene")   # families the placement method can build
SIDES = ("harmful", "harmless")
cap1 = lambda t: t[0].upper() + t[1:]


def job(j):
    """Worker: recover_v1's placement, writing its pictures into v1_2/."""
    RV.OUT, RV.IMG = OUT, IMG
    return RV.recover(j)


def sig(c):
    return json.dumps([c["harmful"], c["harmless"], c["scene"], list(c["additions"]), c["hazard_object"], c["safe_object"]])


def diff_ok(es):
    import numpy as np, imageio.v2 as iio
    A, B = (iio.imread(OUT / es[sd]["images"]["agentview"]).astype(int) for sd in SIDES)
    d = round(float((np.abs(A - B).max(-1) > 25).mean()), 4)
    return d, 0.002 <= d < 0.5


def run(jobs, workers, tag):
    import multiprocessing as mp
    out = {}
    if not jobs: return out
    print(f"{tag}: {len(jobs)} pairs with {workers} workers", flush=True)
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=3) as pool:
        for n, (pid, ret, err) in enumerate(pool.imap_unordered(job, jobs), 1):
            out[pid] = (ret, err)
            print(f"[{n}/{len(jobs)}] {pid} {'ok' if ret else 'no: ' + str(err)[:150]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--finish", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for d in (SCN, IMG): d.mkdir(parents=True, exist_ok=True)
    man1 = json.loads((V1 / "manifest.json").read_text())
    base_fell = man1.get("_base_fell", {})
    rec11 = json.loads((V11 / "manifest_recovered.json").read_text())
    rows = [json.loads(l) for l in open(V11 / "pairs.jsonl")]
    st = json.loads(MAN.read_text()) if MAN.exists() else {"retry": {}, "retry_failed": {}, "new": {}, "tried": {}, "scenes": {}}

    # ------------------------------------------------------------------ 1. retry with new seeds
    if not a.finish:
        todo = []
        for p in rows:
            pid = p["pair_id"]
            if p["harm_type"] != "implicit" or p["scene_ok"] or p["family"] not in FAMS or pid in st["retry"] or pid in st["retry_failed"]: continue
            if pid not in rec11["failed"] or "no visible difference" in rec11["failed"][pid]: continue
            ms = {sd: (p[f"{sd}_scene_key"], man1[p[f"{sd}_scene_key"]]) for sd in SIDES}
            if any(not m.get("checks") or "bddl" not in m for _, m in ms.values()): continue
            todo.append((pid, p["family"], ms, list(range(5, 5 + 2 * a.seeds)), base_fell))
        if a.limit: todo = todo[: a.limit]
        for pid, (ret, err) in run(todo, a.workers, "retry").items():
            if ret:
                es = {sd: e for sd, (k, e) in ret.items()}
                d, ok = diff_ok(es)
                if ok:
                    st["retry"][pid] = {sd: k for sd, (k, e) in ret.items()}
                    for sd, (k, e) in ret.items(): st["scenes"][k] = dict(e, image_diff_frac=d)
                    continue
                err = "placed correctly, but no visible difference between the two pictures"
            st["retry_failed"][pid] = err
        MAN.write_text(json.dumps(st, indent=1))

    def current():
        """v1.1 rows with this run's retries applied; paths relative to v1_2/."""
        out = []
        for p in rows:
            p = dict(p)
            for sd in SIDES:
                im = p.get(f"{sd}_images")
                if im: p[f"{sd}_images"] = {c: (f if f.startswith("../") else "../v1_1/" + f) for c, f in im.items()}
            ks = st["retry"].get(p["pair_id"])
            if ks:
                for sd, k in ks.items():
                    e = st["scenes"][k]
                    p[f"{sd}_scene_key"] = k; p[f"{sd}_bddl"] = e["bddl"]; p[f"{sd}_images"] = e["images"]; p[f"{sd}_robot_state"] = e["robot_state"]
                p["image_diff_frac"] = st["scenes"][ks["harmful"]]["image_diff_frac"]
                p["scene_ok"] = True; p.pop("scene_problem", None); p.pop("recovery_attempt", None); p["recovered_in"] = "v1.2"
                p["clean"] = not p.get("label_flag") and not p.get("text_flag")
            elif p["pair_id"] in st["retry_failed"]:
                p["recovery_attempt"] = st["retry_failed"][p["pair_id"]]
            out.append(p)
        return out

    # ------------------------------------------------------------------ 2. replacements from the generator's unused candidates
    base_rows, _, staged = G.build()
    used = {sig(c) for c in base_rows}
    pool = [c for (pt, fam), cs in sorted(staged.items()) if pt == "implied_scene" and fam in FAMS for c in cs
            if sig(c) not in used and c["safe_object"] != "notebook"]
    random.Random(0).shuffle(pool)
    group_split = {r["split_group"]: r["split"] for r in rows}
    group_split.update({v["row"]["split_group"]: v["row"]["split"] for v in st["new"].values()})   # earlier runs' new pairs too

    def counters(cur, new_rows):
        verb_n, obj_n, haz_n, trip = Counter(), Counter(), Counter(), set()
        for r in [r for r in cur if r["clean"]] + new_rows:
            verb_n[r["main_verb"]] += 1; obj_n[r["hazard_object"]] += 1
            if r["differs"] != "verb": haz_n[r["hazard_object"]] += 1
            if r["harm_type"] == "implicit": trip.add((r["harmful"], r["hazard_object"], r["safe_object"]))
        return verb_n, obj_n, haz_n, trip

    def allowed(c, ctr):
        verb_n, obj_n, haz_n, trip = ctr
        return not (verb_n[G.main_verb(c["harmful"])] >= G.VERB_CAP or obj_n[c["hazard_object"]] >= G.OBJ_CAP
                    or (c["hazard_object"] in G.SEM_ONLY and haz_n[c["hazard_object"]] >= G.SEM_HAZ_CAP)
                    or (cap1(c["harmful"]), c["hazard_object"], c["safe_object"]) in trip)

    def bump(c, ctr):
        ctr[0][G.main_verb(c["harmful"])] += 1; ctr[1][c["hazard_object"]] += 1; ctr[2][c["hazard_object"]] += 1
        ctr[3].add((cap1(c["harmful"]), c["hazard_object"], c["safe_object"]))

    def new_rows():
        return [st["new"][k]["row"] for k in sorted(st["new"], key=lambda k: st["new"][k]["row"]["pair_id"])]

    def need_by_stage(cur):
        """How many clean implicit pairs each stage is short of its original size."""
        tot = Counter(r["stage"] for r in rows if r["harm_type"] == "implicit")
        have = Counter(r["stage"] for r in cur + new_rows() if r["harm_type"] == "implicit" and r["clean"])
        return {s: max(0, tot[s] - have[s]) for s in (1, 2, 3)}

    if not a.finish:
        for rnd in range(a.rounds):
            cur = current(); need = need_by_stage(cur)
            if not sum(need.values()): break
            # choose a batch: per stage, rotate over families and prefer the hazardous object used least so far
            ctr = counters(cur, new_rows()); batch, fam_n = [], Counter(r["family"] for r in cur + new_rows() if r["harm_type"] == "implicit" and r["clean"])
            for s in (1, 2, 3):
                want = int(need[s] * 1.7) + (2 if need[s] else 0)
                cands = [c for c in pool if c["stage"] == s and sig(c) not in st["tried"]]
                while want > 0:
                    ok = [c for c in cands if allowed(c, ctr)]
                    if not ok: break
                    c = min(ok, key=lambda c: (fam_n[c["family"]], ctr[1][c["hazard_object"]]))
                    cands.remove(c); bump(c, ctr); fam_n[c["family"]] += 1; batch.append(c); want -= 1
            if a.limit: batch = batch[: a.limit]
            if not batch: print("no candidates left for", need); break
            jobs, info = [], {}
            for i, c in enumerate(batch):
                tid = f"r{rnd}_{i:03d}"
                try:
                    ms = {}
                    for sd, spec in V.scene_specs(c).items():
                        text, notes, checks = V.build_bddl(spec); k = V.key_of(spec)
                        (SCN / f"{k}.bddl").write_text(text)
                        ms[sd] = (k, dict(spec=spec, bddl=f"../v1_2/scenes/{k}.bddl", notes=notes, checks=checks))
                    if any(not m["checks"] for _, m in ms.values()): raise ValueError("no placement to check")
                except Exception as ex:
                    st["tried"][sig(c)] = f"could not build the scene: {type(ex).__name__}: {str(ex)[:120]}"; continue
                info[tid] = c; jobs.append((tid, c["family"], ms, list(range(a.seeds)), base_fell))
            res = run(jobs, a.workers, f"replacements round {rnd + 1} (still needed per stage: {need})")
            cur = current(); ctr = counters(cur, new_rows()); need = need_by_stage(cur)
            split_need = Counter(r["split"] for r in cur if r["harm_type"] == "implicit" and not r["clean"])
            split_need.subtract(Counter(r["split"] for r in new_rows()))
            for tid in sorted(res):
                ret, err = res[tid]; c = info[tid]
                if not ret: st["tried"][sig(c)] = err; continue
                es = {sd: dict(e, bddl=e["bddl"].replace("../v1/../v1_2/", "")) for sd, (k, e) in ret.items()}
                d, ok = diff_ok(es)
                if not ok: st["tried"][sig(c)] = "placed correctly, but no visible difference between the two pictures"; continue
                if not need[c["stage"]] or not allowed(c, ctr): st["tried"][sig(c)] = "rendered fine but not needed (stage full or a cap reached)"; continue
                row = G.to_row(c, 501 + len(st["new"]))
                sg = f"{row['family']}|{('i', row['harmful'], row['hazard_object'])}"
                row["split"] = group_split.get(sg) or max(("train", "val", "test"), key=lambda k: split_need[k])
                row["split_group"] = sg; group_split[sg] = row["split"]; split_need[row["split"]] -= 1
                for sd, (k, _) in ret.items():
                    row[f"{sd}_scene_key"] = k; row[f"{sd}_bddl"] = es[sd]["bddl"]; row[f"{sd}_images"] = es[sd]["images"]; row[f"{sd}_robot_state"] = es[sd]["robot_state"]
                    st["scenes"][k] = dict(es[sd], image_diff_frac=d)
                row.update(scene_ok=True, image_diff_frac=d, clean=True, added_in="v1.2")
                st["new"][sig(c)] = {"row": row}; st["tried"][sig(c)] = "accepted"
                bump(c, ctr); need[c["stage"]] -= 1
            MAN.write_text(json.dumps(st, indent=1))

    out = current() + new_rows()
    (OUT / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    MAN.write_text(json.dumps(st, indent=1))
    c = lambda f: sum(1 for r in out if f(r))
    print(f"retried ok {len(st['retry'])}, new pairs {len(st['new'])}; clean {c(lambda r: r['clean'])} "
          f"(explicit {c(lambda r: r['clean'] and r['harm_type'] == 'explicit')}, implicit {c(lambda r: r['clean'] and r['harm_type'] == 'implicit')}); "
          f"still short per stage {need_by_stage(current())}")


if __name__ == "__main__":
    main()
