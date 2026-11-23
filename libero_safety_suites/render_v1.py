"""Dataset v1 = the 500 suite pairs (suite_pairs.jsonl, unchanged) + one rendered scene per side.

Nothing existing is modified: everything is written under libero_safety_suites/v1/
  scenes/<key>.bddl      scene files (benchmark scene + the pair's added objects / the one swapped object)
  images/<key>__{agentview,wrist,front}.png
  manifest.json          per scene: status, seed, placement checks, objects that fell off
  pairs.jsonl            the 500 pairs + harmful_scene_key / harmless_scene_key, image paths, scene_ok

Explicit pairs: one scene for both sides (base scene + added objects on free table spots).
Implicit pairs: two scenes that differ by ONE object in the same place (on the burner, in the microwave, under the food, on the tray,
or the obstacle in the robot's path). The swapped object is always a new instance, so both sides get the identical edit.

    MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/render_v1.py --build-only      # write scene files, report what cannot be built
    MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/render_v1.py --workers 6       # render (resumes: skips scenes already clean)
    MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/render_v1.py --finish          # no rendering: write pairs.jsonl from the manifest
"""
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "libero_safety"))
import render_scenes as R            # noqa: E402
import generate_suite_pairs as G     # noqa: E402

V1 = HERE / "v1"
SCN, IMG, MAN = V1 / "scenes", V1 / "images", V1 / "manifest.json"
TASK_BDDL = {t["task_id"]: t["bddl"] for t in G.COMP["tasks"]}


def swap_type(text, old, new):
    """Replace every instance of object type `old` by one of type `new`, in place (same region, motion and constraints)."""
    names = [n for n, ty in R.declared(text, "objects") if ty == old]
    assert names, f"no {old} in scene"
    for i, n in enumerate(names, 1):
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9])", f"{new}_{i}", text)
    return re.sub(rf"- {re.escape(old)}(?![A-Za-z0-9_])", f"- {new}", text)


def scene_specs(c):
    """{side: spec}; spec = dict(base=task id, additions=[type], arrangement=[...], swap=(old, new) or None)."""
    base = c["extra"].get("base", c["scene"])
    text = (R.LS / TASK_BDDL[base]).read_text()
    have = {}   # lexicon key -> the type name this scene uses for it (e.g. classic_blue_mug -> rotated_classic_blue_mug)
    for _, ty in R.declared(text, "fixtures") + R.declared(text, "objects"):
        k = G.inv_key(ty)
        if k: have.setdefault(k, ty)
    ty = lambda k: have.get(k, k)
    add = list(c["additions"])
    if c["pair_type"] != "implied_scene":
        s = dict(base=base, additions=add, arrangement=[], swap=None)
        return {"harmful": s, "harmless": s}
    out = {}
    for side, o in (("harmful", c["hazard_object"]), ("harmless", c["safe_object"])):
        fam = c["family"]
        if fam in ("human_in_path", "stab_scene"):
            orig = c["safe_object"] if fam == "human_in_path" else c["hazard_object"]   # the object the base scene ships with
            out[side] = dict(base=base, additions=[], arrangement=[], swap=None if o == orig else (orig, o))
            continue
        if fam == "stove_scene": arr = [{"object": o, "on": "flat_stove"}]
        elif fam == "micro_scene": arr = [{"object": o, "in": "microwave"}]
        elif fam == "cut_scene":
            food = c["need"][0]
            arr = [{"object": ty(food), "on": o}]
        else:   # heavy_scene / flip_scene / pour_scene: the object lies on the tray named in the sentence
            tray = next(t for t in c["need"] if G.ph(t) == c["fmt"]["t"])
            arr = [{"object": o, "on": ty(tray)}]
        out[side] = dict(base=base, additions=add + [o], arrangement=arr, swap=None)
    return out


def key_of(spec):
    blob = json.dumps(spec, sort_keys=True)
    return f"{spec['base'].replace('/', '__')[:70]}__{hashlib.sha1(blob.encode()).hexdigest()[:8]}"


def merge_decls(text):
    """One declaration line per object type: the BDDL parser keeps only the LAST line of a type, so 'a_1 - a' + 'a_2 - a' loses a_1."""
    s, e = R.block_span(text, "objects")
    by = {}
    for n, ty in R.declared(text, "objects"): by.setdefault(ty, []).append(n)
    return text[:s] + "(:objects\n" + "".join(f"    {' '.join(ns)} - {ty}\n" for ty, ns in by.items()) + "  )" + text[e:]


def prefetch(folder):
    """Assets render_scenes.fetch_from_zip cannot locate (it assumes new_objects*/<name>/<variant>/, this one is one level shallower)."""
    dest0 = R.ASSETS / folder
    if dest0.exists() and any(dest0.iterdir()): return
    from remotezip import RemoteZip
    z = RemoteZip(R.ZIP_URL)
    for i in z.infolist():
        if i.filename.startswith(f"assets/{folder}/") and not i.is_dir():
            d = R.ASSETS / i.filename[len("assets/"):]; d.parent.mkdir(parents=True, exist_ok=True)
            if not d.exists(): d.write_bytes(z.read(i.filename))
    print("prefetched", folder)


def build_bddl(spec):
    text = (R.LS / TASK_BDDL[spec["base"]]).read_text()
    notes, checks = [], []
    if spec["swap"]:
        text = swap_type(text, *spec["swap"]); notes.append(f"{spec['swap'][0]} swapped in place for {spec['swap'][1]}")
    if spec["additions"] or spec["arrangement"]:
        sc = dict(additions=[{"object": o} for o in spec["additions"]], arrangement=spec["arrangement"])
        text, n2, checks = R.compose_bddl(text, sc)
        notes += n2
        text = merge_decls(text)
    return text, notes, checks


BASE_FELL = {}   # base task id -> objects that end up off the table in the benchmark's own, unmodified scene (not our doing)


def status(m):
    """(ok, reasons). Stricter than render_scenes.scene_status: an arranged object must also end up no more than 15 cm above its
    target (otherwise it is resting on the robot's gripper or another object), and objects that already leave the table in the
    unmodified benchmark scene are not counted against the scene."""
    if not m or m.get("status") != "ok": return False, [m.get("error", "render failed") if m else "not rendered"]
    why = []
    for c in m.get("placement_check", []):
        if c.get("ok") is False: why.append(f"placement failed {c['object']}->{c['target']}")
        elif c.get("dz") is not None and c["dz"] > 0.15: why.append(f"{c['object']} is {c['dz']:.2f} m above {c['target']} (resting on something else)")
    fell = [o for o in m.get("fell_off", []) if o not in BASE_FELL.get(m["spec"]["base"], ())]
    if fell: why.append(f"fell off {fell}")
    return not why, why


def probe(job):
    """Robot state of a scene after the same reset + settle as the render (no image is kept), and what fell off."""
    key, path, seed = job
    try:
        R.setup_env_vars()
        import numpy as np
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=64, camera_widths=64, camera_names=["agentview"])
        try:
            for sd in range(seed, seed + 6):
                try:
                    env.seed(sd); obs = env.reset(); break
                except Exception as e:
                    if "place all objects" not in str(e) or sd == seed + 5: raise
            for _ in range(20): obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
            q = np.array(obs["robot0_eef_quat"], dtype=float); q[3] = np.clip(q[3], -1, 1); den = np.sqrt(1 - q[3] ** 2)
            aa = np.zeros(3) if np.isclose(den, 0) else q[:3] * 2 * np.arccos(q[3]) / den
            zs = {n: float(x.get_geom_state()["pos"][2]) for n, x in env.env.object_states_dict.items()}
            zmed = sorted(zs.values())[len(zs) // 2]
            r = lambda v: [round(float(x), 6) for x in v]
            return key, dict(eef_pos=r(obs["robot0_eef_pos"]), eef_quat_xyzw=r(obs["robot0_eef_quat"]), gripper_qpos=r(obs["robot0_gripper_qpos"]),
                             joint_pos=r(obs["robot0_joint_pos"]),
                             pi05_state=r(np.concatenate([obs["robot0_eef_pos"], aa, obs["robot0_gripper_qpos"]]))), [n for n, z in zs.items() if z < zmed - 0.25]
        finally:
            env.close()
    except Exception as e:
        return key, {"error": f"{type(e).__name__}: {str(e)[:200]}"}, None


def match_pair(job):
    """Render BOTH scenes of an implicit pair from the same simulator seed, so the layout of everything except the swapped object is
    the same. Tries seeds until both sides are clean; returns {side: (new key, manifest entry)} or None."""
    pid, sides, seeds, base_fell = job
    try:
        R.setup_env_vars()
        import numpy as np, imageio
        from libero.libero.envs import OffScreenRenderEnv
        BASE_FELL.update(base_fell)
        envs = {sd: OffScreenRenderEnv(bddl_file_name=str(V1 / m["bddl"]), camera_heights=256, camera_widths=256,
                                       camera_names=["agentview", "robot0_eye_in_hand", "frontview"]) for sd, (k, m) in sides.items()}
        try:
            for seed in seeds:
                out, start = {}, None
                for sd, (k, m) in sides.items():
                    env = envs[sd]
                    try:
                        env.seed(seed); obs = env.reset()
                    except Exception:
                        break
                    # the second scene starts with every shared object exactly where the first scene's reset put it (the simulator
                    # otherwise reshuffles the layout when the swapped object has a different size)
                    sim, ee = env.env.sim, env.env
                    def joints():
                        d = {}
                        for n in ee.obj_body_id:
                            try: j = ee.get_object(n).joints
                            except Exception: j = []
                            if j: d[n] = j[0]
                        return d
                    js = joints()
                    if start is None: start = {n: np.array(sim.data.get_joint_qpos(j)) for n, j in js.items()}
                    else:
                        for n, j in js.items():
                            if n in start and len(start[n]) == len(sim.data.get_joint_qpos(j)): sim.data.set_joint_qpos(j, start[n])
                        ua, ub = [n for n in start if n not in js], [n for n in js if n not in start]
                        if len(ua) == 1 and len(ub) == 1 and len(start[ua[0]]) == 7:   # the swapped object goes to the same spot on the table
                            q = np.array(sim.data.get_joint_qpos(js[ub[0]]))
                            if len(q) == 7: q[:2] = start[ua[0]][:2]; sim.data.set_joint_qpos(js[ub[0]], q)
                        sim.forward()
                    for _ in range(20): obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
                    st = env.env.object_states_dict
                    res = []
                    for o, t in m.get("checks", []):
                        po, pt = st[o].get_geom_state()["pos"], st[t].get_geom_state()["pos"]
                        d = float(((po[:2] - pt[:2]) ** 2).sum() ** .5)
                        res.append({"object": o, "target": t, "ok": bool(d < 0.1 and po[2] > pt[2] - 0.06), "xy_dist": round(d, 3), "dz": round(float(po[2] - pt[2]), 3)})
                    zs = {n: float(x.get_geom_state()["pos"][2]) for n, x in st.items()}
                    zmed = sorted(zs.values())[len(zs) // 2]
                    q = np.array(obs["robot0_eef_quat"], dtype=float); q[3] = np.clip(q[3], -1, 1); den = np.sqrt(1 - q[3] ** 2)
                    aa = np.zeros(3) if np.isclose(den, 0) else q[:3] * 2 * np.arccos(q[3]) / den
                    r = lambda v: [round(float(x), 6) for x in v]
                    e = dict(spec=m["spec"], bddl=m["bddl"], notes=m.get("notes", []), checks=m.get("checks", []), status="ok", seed=seed,
                             placement_check=res, fell_off=[n for n, z in zs.items() if z < zmed - 0.25],
                             robot_state=dict(eef_pos=r(obs["robot0_eef_pos"]), eef_quat_xyzw=r(obs["robot0_eef_quat"]), gripper_qpos=r(obs["robot0_gripper_qpos"]),
                                              joint_pos=r(obs["robot0_joint_pos"]), pi05_state=r(np.concatenate([obs["robot0_eef_pos"], aa, obs["robot0_gripper_qpos"]]))))
                    if not status(e)[0]: break
                    pos = {n: np.array(x.get_geom_state()["pos"], dtype=float) for n, x in st.items()}
                    out[sd] = (f"{k}__s{seed}", e, {c: obs[kk][::-1].copy() for c, kk in R.CAMERAS.items()}, pos)
                if len(out) == len(sides):
                    pa, pb = (out[sd][3] for sd in out)   # everything the two scenes share must sit in the same place
                    shift = max([float(np.linalg.norm(pa[n] - pb[n])) for n in set(pa) & set(pb)] or [0.0])
                    if shift > 0.03: continue
                    ret = {}
                    for sd, (k2, e, imgs, _) in out.items():
                        e["layout_max_shift"] = round(shift, 4)
                        e["images"] = {}
                        for cam, im in imgs.items():
                            f = IMG / f"{k2}__{cam}.png"; imageio.imwrite(f, im); e["images"][cam] = str(f.relative_to(V1))
                        ret[sd] = (k2, e)
                    return pid, ret, None
            return pid, None, "no seed gives two clean scenes with the same layout"
        finally:
            for env in envs.values(): env.close()
    except Exception as ex:
        return pid, None, f"{type(ex).__name__}: {str(ex)[:200]}"


def render(job):
    key, path, checks, tries, seed0 = job
    try:
        R.setup_env_vars()
        import imageio
        outs, seed = R.render_one(path, 256, seed0, 20, True, checks, (0,), tries)
        _, imgs, res, fell, _ = outs[0]
        files = {}
        for cam, im in imgs.items():
            f = IMG / f"{key}__{cam}.png"; imageio.imwrite(f, im); files[cam] = str(f.relative_to(V1))
        return key, dict(status="ok", images=files, seed=seed, placement_check=res, fell_off=fell)
    except Exception as e:
        return key, dict(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true"); ap.add_argument("--finish", action="store_true")
    ap.add_argument("--match-seeds", action="store_true", help="re-render implicit pairs whose two scenes were rendered from different seeds (or are not clean) from one shared seed")
    ap.add_argument("--states", action="store_true", help="no rendering: record robot state per rendered scene and what falls in the unmodified base scenes")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--tries", type=int, default=3); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="first simulator seed to try (use a higher one to retry flagged scenes with new object positions)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(HERE / "suite_pairs.jsonl")]
    cands, _, _ = G.build()
    cands = sorted(cands, key=lambda c: (c["stage"], c["pair_type"], c["family"], c["harmful"]))
    assert len(cands) == len(rows)
    for r, c in zip(rows, cands):   # the regenerated candidates must be exactly the saved pairs
        t = G.to_row(c, 0)
        assert (t["harmful"], t["harmless"], t["harmful_scene"], t["harmless_scene"]) == (r["harmful"], r["harmless"], r["harmful_scene"], r["harmless_scene"]), r["pair_id"]
    for d in (SCN, IMG): d.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MAN.read_text()) if MAN.exists() else {}
    pair_keys, jobs, build_err = {}, {}, {}
    for r, c in zip(rows, cands):
        try:
            specs = scene_specs(c)
        except Exception as e:
            build_err[r["pair_id"]] = f"{type(e).__name__}: {e}"; continue
        pair_keys[r["pair_id"]] = {}
        for side, spec in specs.items():
            k = key_of(spec); pair_keys[r["pair_id"]][side] = k
            if k in jobs or k in build_err: continue
            try:
                text, notes, checks = build_bddl(spec)
            except Exception as e:
                build_err[k] = f"{type(e).__name__}: {str(e)[:200]}"; continue
            p = SCN / f"{k}.bddl"; p.write_text(text)
            manifest[k] = {**manifest.get(k, {}), "spec": spec, "bddl": str(p.relative_to(V1)), "notes": notes, "checks": checks}
            jobs[k] = (k, p, checks, a.tries, a.seed)
    for k, e in build_err.items(): manifest.setdefault(k, {}).update(status="failed", error="build: " + e)
    print(f"{len(rows)} pairs, {len(jobs)} scenes to build, {len(build_err)} could not be built")
    for k, e in list(build_err.items())[:15]: print("  BUILD FAIL", k, e)
    MAN.write_text(json.dumps(manifest, indent=1))

    BASE_FELL.update(manifest.get("_base_fell", {}))
    if a.states:
        import multiprocessing as mp
        todo = [("_base_" + b, R.LS / TASK_BDDL[b], 0) for b in sorted({m["spec"]["base"] for k, m in manifest.items() if not k.startswith("_") and "spec" in m}) if b not in BASE_FELL]
        todo += [(k, jobs[k][1], manifest[k].get("seed", 0)) for k in jobs if manifest[k].get("status") == "ok" and "pi05_state" not in manifest[k].get("robot_state", {})]
        print(f"probing {len(todo)} scenes", flush=True)
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=8) as pool:
            for n, (k, st, fell) in enumerate(pool.imap_unordered(probe, todo), 1):
                if k.startswith("_base_"):
                    if fell is not None: BASE_FELL[k[6:]] = fell
                else: manifest[k]["robot_state"] = st
                if "error" in st: print("  PROBE FAIL", k, st["error"], flush=True)
                if n % 25 == 0:
                    print(f"[{n}/{len(todo)}]", flush=True); manifest["_base_fell"] = BASE_FELL; MAN.write_text(json.dumps(manifest, indent=1))
        manifest["_base_fell"] = BASE_FELL; MAN.write_text(json.dumps(manifest, indent=1))

    override = manifest.setdefault("_pair_override", {})   # pair id -> {side: manifest key of the same-seed render}
    if a.match_seeds:
        import multiprocessing as mp
        todo = []
        for r in rows:
            ks = pair_keys.get(r["pair_id"])
            if r["harm_type"] != "implicit" or not ks: continue
            ov = override.get(r["pair_id"])
            if ov and all("layout_max_shift" in manifest.get(k, {}) for k in ov.values()): continue   # already verified
            override.pop(r["pair_id"], None)
            ms = {sd: manifest[k] for sd, k in ks.items()}
            if any("bddl" not in m for m in ms.values()): continue
            first = [m["seed"] for m in ms.values() if m.get("status") == "ok"]
            seeds = list(dict.fromkeys(first + list(range(8))))
            todo.append((r["pair_id"], {sd: (k, manifest[k]) for sd, k in ks.items()}, seeds, BASE_FELL))
        print(f"matching seeds for {len(todo)} implicit pairs", flush=True)
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=4) as pool:
            for n, (pid, ret, err) in enumerate(pool.imap_unordered(match_pair, todo), 1):
                if ret:
                    override[pid] = {sd: k2 for sd, (k2, e) in ret.items()}
                    for sd, (k2, e) in ret.items(): manifest[k2] = e
                print(f"[{n}/{len(todo)}] {pid} {'ok seed ' + str(next(iter(ret.values()))[1]['seed']) if ret else 'BAD ' + err}", flush=True)
                if n % 10 == 0: MAN.write_text(json.dumps(manifest, indent=1))
        MAN.write_text(json.dumps(manifest, indent=1))

    if not (a.build_only or a.finish or a.states or a.match_seeds):
        todo = [j for k, j in jobs.items() if not status(manifest[k])[0]]
        if a.limit: todo = todo[:: max(1, len(todo) // a.limit)][: a.limit]
        prefetch("new_objects_with_hand/plate")
        print(f"rendering {len(todo)} scenes with {a.workers} workers", flush=True)
        import multiprocessing as mp
        t0 = time.time()
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=8) as pool:
            for n, (k, res) in enumerate(pool.imap_unordered(render, todo), 1):
                old = manifest[k]
                if status({**old, **res, "robot_state": None})[0] or old.get("status") != "ok":   # keep the earlier render unless the new one is clean
                    old.pop("error", None); old.pop("robot_state", None); old.update(res)
                ok, why = status(manifest[k])
                print(f"[{n}/{len(todo)}] {'ok  ' if ok else 'BAD '} {k} {'' if ok else res.get('error') or why}  ({time.time() - t0:.0f}s)", flush=True)
                if n % 10 == 0: MAN.write_text(json.dumps(manifest, indent=1))
        MAN.write_text(json.dumps(manifest, indent=1))

    out, n_ok = [], 0
    for r in rows:
        ks = override.get(r["pair_id"]) or pair_keys.get(r["pair_id"])
        r = dict(r)
        if ks:
            oks = [status(manifest.get(k))[0] for k in ks.values()]
            for side, k in ks.items():
                m = manifest.get(k, {})
                r[f"{side}_scene_key"] = k; r[f"{side}_bddl"] = m.get("bddl"); r[f"{side}_images"] = m.get("images")
                r[f"{side}_robot_state"] = m.get("robot_state")
            r["scene_ok"] = all(oks)
            prob = [w for k in ks.values() for w in status(manifest.get(k))[1]]
            if r["harm_type"] == "implicit" and all(oks):
                a_, b_ = (manifest[ks[sd]] for sd in ("harmful", "harmless"))
                if "layout_max_shift" not in a_: prob.append("no seed gives two clean scenes with the same layout for everything except the swapped object")
                import numpy as np, imageio.v2 as iio
                A, B = (iio.imread(V1 / m["images"]["agentview"]).astype(int) for m in (a_, b_))
                r["image_diff_frac"] = round(float((np.abs(A - B).max(-1) > 25).mean()), 4)   # share of table-view pixels that differ
                if r["image_diff_frac"] < 0.002: prob.append("no visible difference between the two scenes in the table view")
            r["scene_ok"] = not prob
            if prob: r["scene_problem"] = prob
        else:
            r["scene_ok"] = False; r["scene_problem"] = [build_err.get(r["pair_id"], "not built")]
        # found by looking at the renders: the simulator's "notebook" is a laptop computer, not a paper notebook. It is still a valid
        # hazardous object (fire, water, cutting), but it is not a harmless thing to put weight on or tip off a tray.
        if r["safe_object"] == "notebook":
            r["label_flag"] = "harmless side uses the 'notebook', which renders as a laptop: putting weight on it / flipping it off is not clearly harmless"
        r["clean"] = bool(r["scene_ok"] and not r.get("label_flag"))
        n_ok += r["scene_ok"]; out.append(r)
    (V1 / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"pairs with clean scenes on both sides: {n_ok} of {len(out)} -> {V1 / 'pairs.jsonl'}")
    print(f"clean pairs (scenes ok and no label flag): {sum(r['clean'] for r in out)}")
    import shutil, openpyxl
    src = ROOT / "results" / "dataset_suites.xlsx"
    if src.exists():
        shutil.copy(src, V1 / "dataset_v1.xlsx")
        wb = openpyxl.load_workbook(V1 / "dataset_v1.xlsx")
        if "Render status" in wb.sheetnames: del wb["Render status"]
        ws = wb.create_sheet("Render status")
        ws.append(["Pair", "Harm type", "Stage", "Family", "Split", "Clean", "Scenes OK", "Scene problem", "Label flag", "Harmful image", "Harmless image"])
        for r in out:
            ws.append([r["pair_id"], r["harm_type"], r["stage_name"], r["family"], r["split"], "yes" if r["clean"] else "no", "yes" if r["scene_ok"] else "no",
                       "; ".join(r.get("scene_problem", [])), r.get("label_flag", ""),
                       (r.get("harmful_images") or {}).get("agentview", ""), (r.get("harmless_images") or {}).get("agentview", "")])
        for col, w in zip("ABCDEFGHIJK", (9, 10, 11, 14, 7, 7, 10, 60, 50, 60, 60)): ws.column_dimensions[col].width = w
        ws.freeze_panes = "A2"
        wb.save(V1 / "dataset_v1.xlsx")


if __name__ == "__main__":
    main()
