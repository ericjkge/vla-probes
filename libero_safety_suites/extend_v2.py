"""Dataset v1.3: the same number of usable explicit and implicit pairs, stage by stage. Reads v1/ and v1_2/, writes ONLY to v1_3/.

v1.2 had 218 usable explicit and 231 usable implicit pairs (target 250 each). Steps:
  1. retry     implicit pairs that still fail are tried again with more starting layouts (seeds 13-24).
  2. implicit  new implicit pairs from the generator's unused candidates, with v1.2's rules (extend_v1.py): same caps per verb and per
               hazardous object, no microwave, no "notebook" on the harmless side, same placement and picture checks. The semantic stage
               has no candidates left under the caps, so its shortfall is filled in the new-verb stage instead.
  3. explicit  new explicit pairs from the generator's unused candidates, same caps, none of the two weak-text kinds flagged in v1.1
               (an object on a vase, a milk carton on electronics) and no "notebook" on the harmless side; an explicit sentence never
               lands in two splits. Their scene is rendered once (or reused from v1 when the same scene was already rendered).
  4. balance   each stage gets the same number of usable pairs in both halves. If one half still has more in a stage, its extra pairs
               keep clean=true but get balanced=false. Use balanced == true for the equal set.

    MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/extend_v2.py --workers 8      # resumes
    MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/extend_v2.py --finish
"""
import argparse, json, random, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import render_v1 as V            # noqa: E402
import recover_v1 as RV          # noqa: E402
import extend_v1 as X            # noqa: E402
G = V.G

V1, V12, OUT = V.V1, HERE / "v1_2", HERE / "v1_3"
SCN, IMG, MAN = OUT / "scenes", OUT / "images", OUT / "manifest.json"
SIDES = ("harmful", "harmless")
TARGET = 250
SPLIT_FRAC = {"train": 0.6, "val": 0.2, "test": 0.2}
cap1 = X.cap1


def job_implicit(j):
    RV.OUT, RV.IMG = OUT, IMG
    return RV.recover(j)


def job_explicit(j):
    """Render one explicit scene into v1_3/ and record the robot state after the same settle."""
    key, path, checks, base_fell = j
    V.V1, V.IMG = OUT, IMG
    V.BASE_FELL.update(base_fell)
    k, e = V.render((key, path, checks, 3, 0))
    if e.get("status") == "ok":
        _, rs, _ = V.probe((key, path, e["seed"]))
        e["robot_state"] = rs
    return k, e, None


def pool_run(fn, jobs, workers, tag):
    import multiprocessing as mp
    out = {}
    if not jobs: return out
    print(f"{tag}: {len(jobs)} with {workers} workers", flush=True)
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=3) as pool:
        for n, (pid, ret, err) in enumerate(pool.imap_unordered(fn, jobs), 1):
            out[pid] = (ret, err)
            print(f"[{n}/{len(jobs)}] {pid} {'ok' if ret and (not isinstance(ret, dict) or ret.get('status') == 'ok') else 'no: ' + str(err or (ret or {}).get('error'))[:150]}", flush=True)
    return out


def rel12(p):
    """Paths in a v1.2 row are relative to v1_2/; make them relative to v1_3/."""
    p = dict(p)
    for sd in SIDES:
        im = p.get(f"{sd}_images")
        if im: p[f"{sd}_images"] = {c: (f if f.startswith("../") else "../v1_2/" + f) for c, f in im.items()}
        b = p.get(f"{sd}_bddl")
        if b and not b.startswith("../"): p[f"{sd}_bddl"] = "../v1_2/" + b
    return p


# found by eye (v1.3 contact sheets): the candle lies on its side or is out of view, so "the burning candle" is not in the picture.
# The automatic checks only catch objects that leave the table. New candle pairs are therefore not added.
EYE_FLAG = {pid: "the candle is knocked over or not visible in the picture (found by eye in v1.3)"
            for pid in ("sp_0085", "sp_0347", "sp_0348", "sp_0567", "sp_0568", "sp_0569")}
EYE_FLAG.update({"sp_0582": "the wine bottle named in the sentence is not visible in the picture (found by eye in v1.3)",
                 "sp_0584": "the frypan named in the harmless sentence is not visible in the picture (found by eye in v1.3)"})


def weak_text(c):
    return c["family"] == "liquid_on" or (c["family"] == "heavy_on" and c["hazard_object"] not in ("egg", "eyeglasses_1"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--finish", action="store_true")
    a = ap.parse_args()
    for d in (SCN, IMG): d.mkdir(parents=True, exist_ok=True)
    man1 = json.loads((V1 / "manifest.json").read_text())
    base_fell = man1.get("_base_fell", {}); V.BASE_FELL.update(base_fell)
    man12 = json.loads((V12 / "manifest.json").read_text())
    rows12 = [rel12(json.loads(l)) for l in open(V12 / "pairs.jsonl")]
    st = json.loads(MAN.read_text()) if MAN.exists() else {"retry": {}, "retry_failed": {}, "new": {}, "tried": {}, "scenes": {}}

    # ------------------------------------------------------------------ 1. retry failed implicit pairs with more seeds
    if not a.finish:
        todo = []
        for p in rows12:
            pid = p["pair_id"]
            if p["harm_type"] != "implicit" or p["scene_ok"] or p.get("label_flag") or p["family"] not in X.FAMS: continue
            if pid in st["retry"] or pid in st["retry_failed"]: continue
            ms = {sd: (p[f"{sd}_scene_key"], man1.get(p[f"{sd}_scene_key"], {})) for sd in SIDES}
            if any(not m.get("checks") or "bddl" not in m for _, m in ms.values()): continue
            todo.append((pid, p["family"], ms, list(range(13, 25)), base_fell))
        for pid, (ret, err) in pool_run(job_implicit, todo, a.workers, "retry").items():
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
        out = []
        for p in rows12:
            p = dict(p); ks = st["retry"].get(p["pair_id"])
            if ks:
                for sd, k in ks.items():
                    e = st["scenes"][k]
                    p[f"{sd}_scene_key"] = k; p[f"{sd}_bddl"] = e["bddl"]; p[f"{sd}_images"] = e["images"]; p[f"{sd}_robot_state"] = e["robot_state"]
                p["image_diff_frac"] = st["scenes"][ks["harmful"]]["image_diff_frac"]
                p["scene_ok"] = True; p.pop("scene_problem", None); p.pop("recovery_attempt", None); p["recovered_in"] = "v1.3"
                p["clean"] = not p.get("label_flag") and not p.get("text_flag")
            elif p["pair_id"] in st["retry_failed"]:
                p["recovery_attempt"] = st["retry_failed"][p["pair_id"]]
            out.append(p)
        out += [dict(st["new"][k]["row"]) for k in sorted(st["new"], key=lambda k: st["new"][k]["row"]["pair_id"])]
        for p in out:
            if p["pair_id"] in EYE_FLAG:
                p["scene_ok"] = False; p["clean"] = False; p["scene_problem"] = list(p.get("scene_problem") or []) + [EYE_FLAG[p["pair_id"]]]
        return out

    # ------------------------------------------------------------------ shared bookkeeping
    base_rows, _, staged = G.build()
    used = {X.sig(c) for c in base_rows} | set(man12["new"])            # v1 pairs and v1.2's new pairs
    next_id = lambda cur: 1 + max(int(r["pair_id"][3:]) for r in cur)

    def counters(cur):
        verb_n, obj_n, haz_n, trip, text_n = Counter(), Counter(), Counter(), set(), Counter()
        for r in cur:
            if not r["clean"]: continue
            verb_n[r["main_verb"]] += 1; obj_n[r["hazard_object"]] += 1
            if r["differs"] != "verb": haz_n[r["hazard_object"]] += 1
            if r["harm_type"] == "implicit": trip.add((r["harmful"], r["hazard_object"], r["safe_object"]))
            else: text_n[r["harmful"]] += 1; text_n[("h", r["harmless"])] += 1
        return verb_n, obj_n, haz_n, trip, text_n

    def allowed(c, ctr):
        verb_n, obj_n, haz_n, trip, text_n = ctr
        if verb_n[G.main_verb(c["harmful"])] >= G.VERB_CAP or obj_n[c["hazard_object"]] >= G.OBJ_CAP: return False
        if c["pair_type"] != "explicit" and c["hazard_object"] in G.SEM_ONLY and haz_n[c["hazard_object"]] >= G.SEM_HAZ_CAP: return False
        if c["pair_type"] == "implied_scene": return (cap1(c["harmful"]), c["hazard_object"], c["safe_object"]) not in trip
        return text_n[cap1(c["harmful"])] < G.TEXT_REUSE and text_n[("h", cap1(c["harmless"]))] < G.TEXT_REUSE

    def bump(c, ctr):
        ctr[0][G.main_verb(c["harmful"])] += 1; ctr[1][c["hazard_object"]] += 1
        if c["pair_type"] != "explicit": ctr[2][c["hazard_object"]] += 1
        if c["pair_type"] == "implied_scene": ctr[3].add((cap1(c["harmful"]), c["hazard_object"], c["safe_object"]))
        else: ctr[4][cap1(c["harmful"])] += 1; ctr[4][("h", cap1(c["harmless"]))] += 1

    def have(cur, ht):
        return Counter(r["stage"] for r in cur if r["harm_type"] == ht and r["clean"])

    def split_for(cur, ht, groups):
        """Split of an existing group if any (None if the groups disagree), else the split furthest below 60/20/20 for this half."""
        got = {r["split"] for r in cur if r["split_group"] in groups}
        if len(got) > 1: return None
        if got: return got.pop()
        n = Counter(r["split"] for r in cur if r["harm_type"] == ht and r["clean"])
        tot = sum(n.values()) + 1
        return max(SPLIT_FRAC, key=lambda s: SPLIT_FRAC[s] * tot - n[s])

    # ------------------------------------------------------------------ 2. implicit: fill every stage, overflow into the new-verb stage
    ipool = [c for (pt, fam), cs in sorted(staged.items()) if pt == "implied_scene" and fam in X.FAMS for c in cs
             if X.sig(c) not in used and c["safe_object"] != "notebook"]
    random.Random(1).shuffle(ipool)
    tried12 = set(man12["tried"])

    def i_need(cur):
        h = have(cur, "implicit"); short = TARGET - sum(h.values())
        return short, h

    if not a.finish:
        for rnd in range(a.rounds):
            cur = current(); short, h = i_need(cur)
            if short <= 0: break
            ctr = counters(cur); fam_n = Counter(r["family"] for r in cur if r["harm_type"] == "implicit" and r["clean"])
            cands = [c for c in ipool if X.sig(c) not in st["tried"] and X.sig(c) not in tried12]
            batch, want = [], int(short * 1.7) + 2
            for s in (1, 2, 3):   # earlier stages first, as the generator does
                while want > 0:
                    ok = [c for c in cands if c["stage"] == s and allowed(c, ctr)]
                    if not ok: break
                    c = min(ok, key=lambda c: (fam_n[c["family"]], ctr[1][c["hazard_object"]]))
                    cands.remove(c); bump(c, ctr); fam_n[c["family"]] += 1; batch.append(c); want -= 1
            if not batch: print("implicit: no candidates left under the caps; short by", short); break
            jobs, info = [], {}
            for i, c in enumerate(batch):
                tid = f"i{rnd}_{i:03d}"
                try:
                    ms = {}
                    for sd, spec in V.scene_specs(c).items():
                        text, notes, checks = V.build_bddl(spec); k = V.key_of(spec)
                        (SCN / f"{k}.bddl").write_text(text)
                        ms[sd] = (k, dict(spec=spec, bddl=f"../v1_3/scenes/{k}.bddl", notes=notes, checks=checks))
                    if any(not m["checks"] for _, m in ms.values()): raise ValueError("no placement to check")
                except Exception as ex:
                    st["tried"][X.sig(c)] = f"could not build the scene: {type(ex).__name__}: {str(ex)[:120]}"; continue
                info[tid] = c; jobs.append((tid, c["family"], ms, list(range(4)), base_fell))
            res = pool_run(job_implicit, jobs, a.workers, f"implicit round {rnd + 1} (short {short})")
            cur = current(); ctr = counters(cur); short, _ = i_need(cur)
            for tid in sorted(res):
                ret, err = res[tid]; c = info[tid]
                if not ret: st["tried"][X.sig(c)] = err; continue
                es = {sd: dict(e, bddl=e["bddl"].replace("../v1/../v1_3/", "")) for sd, (k, e) in ret.items()}
                d, ok = diff_ok(es)
                if not ok: st["tried"][X.sig(c)] = "placed correctly, but no visible difference between the two pictures"; continue
                if short <= 0 or not allowed(c, ctr): st["tried"][X.sig(c)] = "rendered fine but not needed (half full or a cap reached)"; continue
                row = G.to_row(c, next_id(cur))
                sg = f"{row['family']}|{('i', row['harmful'], row['hazard_object'])}"
                row["split"] = split_for(cur, "implicit", {sg}); row["split_group"] = sg
                for sd, (k, _) in ret.items():
                    row[f"{sd}_scene_key"] = k; row[f"{sd}_bddl"] = es[sd]["bddl"]; row[f"{sd}_images"] = es[sd]["images"]; row[f"{sd}_robot_state"] = es[sd]["robot_state"]
                    st["scenes"][k] = dict(es[sd], image_diff_frac=d)
                row.update(scene_ok=True, image_diff_frac=d, clean=True, added_in="v1.3")
                st["new"][X.sig(c)] = {"row": row}; st["tried"][X.sig(c)] = "accepted"
                bump(c, ctr); short -= 1; cur.append(row)
            MAN.write_text(json.dumps(st, indent=1))

    # ------------------------------------------------------------------ 3. explicit: match the implicit count in every stage
    epool = [c for (pt, fam), cs in sorted(staged.items()) if pt in ("explicit", "implied_prompt") for c in cs
             if X.sig(c) not in used and c["safe_object"] != "notebook" and not weak_text(c) and c["family"] != "candle_obj"]
    random.Random(2).shuffle(epool)

    def e_need(cur):
        hi, he = have(cur, "implicit"), have(cur, "explicit")
        return {s: max(0, hi[s] - he[s]) for s in (1, 2, 3)}

    if not a.finish:
        for rnd in range(a.rounds):
            cur = current(); need = e_need(cur)
            if not sum(need.values()): break
            ctr = counters(cur); fam_n = Counter(r["family"] for r in cur if r["harm_type"] == "explicit" and r["clean"])
            cands = [c for c in epool if X.sig(c) not in st["tried"]]
            batch = []
            for s in (1, 2, 3):
                want = int(need[s] * 1.5) + (2 if need[s] else 0)
                while want > 0:
                    ok = [c for c in cands if c["stage"] == s and allowed(c, ctr)]
                    if not ok: break
                    c = min(ok, key=lambda c: (fam_n[c["family"]], ctr[1][c["hazard_object"]]))
                    cands.remove(c); bump(c, ctr); fam_n[c["family"]] += 1; batch.append(c); want -= 1
            if not batch: print("explicit: no candidates left for", need); break
            scenes, jobs = {}, []
            for c in batch:
                spec = V.scene_specs(c)["harmful"]; k = V.key_of(spec)
                m1 = man1.get(k)
                if m1 and V.status(dict(m1, spec=m1.get("spec", spec)))[0] and m1.get("robot_state", {}).get("pi05_state"):
                    scenes[k] = dict(m1, bddl="../v1/" + m1["bddl"], images={cc: "../v1/" + f for cc, f in m1["images"].items()}); continue
                if k in st["scenes"] or k in scenes or any(j[0] == k for j in jobs): continue
                try:
                    text, notes, checks = V.build_bddl(spec)
                except Exception as ex:
                    st["tried"][X.sig(c)] = f"could not build the scene: {type(ex).__name__}: {str(ex)[:120]}"; continue
                p = SCN / f"{k}.bddl"; p.write_text(text)
                scenes.setdefault("_new", {})[k] = dict(spec=spec, bddl=f"scenes/{k}.bddl", notes=notes, checks=checks)
                jobs.append((k, p, checks, base_fell))
            for k, (ret, err) in pool_run(job_explicit, jobs, a.workers, f"explicit scenes round {rnd + 1} (needed per stage {need})").items():
                e = dict(scenes["_new"][k], **(ret or {"status": "failed", "error": err}))
                ok, why = V.status(e)
                if ok and "error" in e.get("robot_state", {"error": 1}): ok, why = False, ["no robot state"]
                st["scenes"][k] = dict(e, ok=ok, why=why)
            scenes.pop("_new", None)
            cur = current(); ctr = counters(cur); need = e_need(cur)
            for c in batch:
                if X.sig(c) in st["tried"]: continue
                spec = V.scene_specs(c)["harmful"]; k = V.key_of(spec)
                e = scenes.get(k) or st["scenes"].get(k)
                if not e or (k in st["scenes"] and not st["scenes"][k].get("ok")):
                    st["tried"][X.sig(c)] = "scene did not render cleanly: " + "; ".join((e or {}).get("why", ["not rendered"])); continue
                if not need[c["stage"]] or not allowed(c, ctr): st["tried"][X.sig(c)] = "not needed (stage matched or a cap reached)"; continue
                row = G.to_row(c, next_id(cur))
                groups = {r["split_group"] for r in cur if r["harm_type"] == "explicit" and {r["harmful"], r["harmless"]} & {row["harmful"], row["harmless"]}}
                sp = split_for(cur, "explicit", groups)
                if sp is None: st["tried"][X.sig(c)] = "its sentences are already in two different splits"; continue
                row["split"] = sp; row["split_group"] = min(groups) if groups else f"{row['family']}|{('s', row['harmful'])}"
                for sd in SIDES:
                    row[f"{sd}_scene_key"] = k; row[f"{sd}_bddl"] = e["bddl"]; row[f"{sd}_images"] = e["images"]; row[f"{sd}_robot_state"] = e["robot_state"]
                row.update(scene_ok=True, clean=True, added_in="v1.3")
                st["new"][X.sig(c)] = {"row": row}; st["tried"][X.sig(c)] = "accepted"
                bump(c, ctr); need[c["stage"]] -= 1; cur.append(row)
            MAN.write_text(json.dumps(st, indent=1))

    # ------------------------------------------------------------------ 4. balance: same usable count per stage in both halves
    out = current()
    hi, he = have(out, "implicit"), have(out, "explicit")
    keep = {s: min(hi[s], he[s]) for s in (1, 2, 3)}
    for r in out:
        r["balanced"] = bool(r["clean"]); r.pop("balance_note", None)
    for ht in ("explicit", "implicit"):
        for s in (1, 2, 3):
            rs = [r for r in out if r["harm_type"] == ht and r["clean"] and r["stage"] == s]
            extra = len(rs) - keep[s]
            # drop the newest pairs first, from the split with the most pairs relative to 60/20/20
            n = Counter(r["split"] for r in rs)
            while extra > 0:
                sp = max(SPLIT_FRAC, key=lambda k: n[k] / SPLIT_FRAC[k])
                r = max((r for r in rs if r["split"] == sp and r["balanced"]), key=lambda r: r["pair_id"])
                r["balanced"] = False; r["balance_note"] = f"usable, but left out of the balanced set: the other half has only {keep[s]} pairs in this stage"
                n[sp] -= 1; extra -= 1
    (OUT / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    MAN.write_text(json.dumps(st, indent=1))
    b = [r for r in out if r.get("balanced")]
    print(f"retried ok {len(st['retry'])}, new pairs {len(st['new'])}; clean explicit {sum(he.values())} implicit {sum(hi.values())}; "
          f"balanced set {len(b)} = per stage {dict(sorted(keep.items()))} per half; splits "
          f"{ {ht: dict(Counter(r['split'] for r in b if r['harm_type'] == ht)) for ht in ('explicit', 'implicit')} }")


def diff_ok(es):
    import numpy as np, imageio.v2 as iio
    A, B = (iio.imread(OUT / es[sd]["images"]["agentview"]).astype(int) for sd in SIDES)
    d = round(float((np.abs(A - B).max(-1) > 25).mean()), 4)
    return d, 0.002 <= d < 0.5


if __name__ == "__main__":
    main()
