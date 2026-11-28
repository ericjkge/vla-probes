"""Dataset v1.6 scene files and images, by direct placement (no random placement, no retries).

Per pair (scene contents from make_v1_6.scene_plan / v1_6/scenes.json):
  1. build each side's scene (LIBERO plain `table`, the fixture at the back centre) and measure every object's exact extent from its
     mesh vertices;
  2. table objects go in one row across the table (two rows if it is too wide), spaced by their real widths + GAP; an implicit
     pair's two sides use the same positions (each slot is as wide as its wider side);
  3. stacked objects are set directly on their support (DROP above it), objects on a fixture directly on its site, the water balls
     in a column inside the mug;
  4. settle SETTLE steps, render once (agentview, wrist, front).
One check only: a support that is completely hidden in the table view (hidden_support in the result). Everything else is reviewed by
eye; re-render chosen pairs with --pairs.

    MUJOCO_GL=cgl ../.venv/bin/python place_v1_6.py --pairs sp_0140,sp_0260      # a few pairs
    MUJOCO_GL=cgl ../.venv/bin/python place_v1_6.py --all                        # every pair not done yet (--redo: all again)
    python place_v1_6.py --finish                                                # -> v1_6/pairs.jsonl + hidden-support list
"""
import argparse, json, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "libero_safety"))
OUT = HERE / "v1_6"
SIDES = ("harmful", "harmless")
TABLE_Z = 0.9          # top of LIBERO's `table`
GAP = 0.03             # between neighbouring objects in a row
DROP = 0.005           # objects are set this far above what they rest on
ROW_X = -0.10           # centre of the row(s) in depth (negative = back, towards the robot); also keeps them in the wrist view
ROW_MAX = 0.46         # widest single row that stays in the table view
FIXTURE_Y = [0.27, -0.27, 0.27]   # fixture side per variant (in the middle it stands behind the robot arm)
# fixture depth position: its back edge at the back of the table (x -0.47), so it reaches as little as possible into the object row
FIXTURE_X = {"flat_stove": -0.427, "microwave": -0.307, "wooden_cabinet": -0.348}
FRONT = 0.10          # rows end at most here (front of the table view); further back than BACK they hit the robot base
BACK = -0.43
HALF = 0.25            # objects stay within |y| <= HALF (table view)
LAY_FLAT = {"black_book"}   # LIBERO stands these on their edge; lay them flat (things are put on them)
N_WATER = 4            # water balls per mug: a mug's inside is ~5 cm across, a ball 3 cm, so more do not fit
SETTLE = 30
HIDDEN_PX = 10         # a support with fewer visible pixels than this counts as completely hidden
# hand-set offsets (dx back<0 / front>0, dy) of a covering object on its support, so the support shows: {pair: {side: {object: (dx, dy)}}}
OFFSET = {
    "sp_0260": {"harmful": {"notebook": (-0.15, 0.08)}},
    "sp_0262": {"harmful": {"notebook": (-0.08, 0.10)}},
    "sp_0266": {"harmful": {"game_console": (-0.04, -0.09)}},
    "sp_0267": {"harmful": {"notebook": (-0.08, 0.10)}},
    "sp_0269": {"harmful": {"game_console": (-0.04, 0.09)}},
    "sp_0270": {"harmful": {"notebook": (-0.15, 0.08)}},
    "ho_040": {"harmful": {"cardstock": (-0.07, 0.07)}},
}
# parking spots for LIBERO's initial placement (everything is moved afterwards); far apart so the sampler never fails
PARK = [(0.30, -0.40), (0.30, 0.00), (0.30, 0.40), (-0.05, -0.45), (-0.05, 0.45), (0.05, 0.0)]


def inst(o): return f"{o}_1"


def renamed(r):
    """v1.6 text fixes: LIBERO's `notebook` object is a laptop, so the sentences call it a laptop."""
    import re
    fix = lambda t: re.sub(r"\bnotebook\b", "laptop", t.replace("notebook (paper)", "notebook")) if isinstance(t, str) else t
    r = {k: fix(v) for k, v in r.items()}
    # push_off: the one scene shows the object already on the cabinet / table, so the harmless sentence must not ask to place it
    # there; use a verb that fits an object already in place (held-out pairs: held-out verbs, so their verbs stay unseen)
    m = re.match(r"(Place|Put|Rest|Set|Position|Arrange|Stand) (the .+) (on the (?:cabinet|table))$", r["harmless"])
    if r["family"] == "push_off" and m:
        v, o, where = m.groups()
        new = {"Place": "Leave", "Rest": "Leave", "Put": "Keep", "Position": "Steady", "Arrange": "Inspect", "Stand": "Polish"}
        r["harmless"] = f"Let {o} stay {where}" if v == "Set" else f"{new[v]} {o} {where}"
    return r


FIXTURE_SUB = {"flat_stove": ["cook_region"], "microwave": ["heating_region", "top_side"], "wooden_cabinet": ["top_side", "top_region"]}
FIXTURE_YAW = {"wooden_cabinet": 3.141592653589793, "microwave": 0.0, "flat_stove": 0.0}


def write_scene(path, language, fixture, items, variant):
    """LIBERO scene file. Table objects start at the PARK spots (they are moved by arrange()); the fixture's region fixes where it
    stands; stacked objects / water start on their support."""
    region = lambda name, xy, yaw: (f"      ({name}\n          (:target main_table)\n          (:ranges (\n              "
                                    f"({xy[0] - 0.01:.3f} {xy[1] - 0.01:.3f} {xy[0] + 0.01:.3f} {xy[1] + 0.01:.3f})\n            )\n          )\n"
                                    f"          (:yaw_rotation (\n              ({yaw} {yaw})\n            )\n          )\n      )")
    regions, fixtures, objects, init, names = [], ["    main_table - table"], [], [], {}
    if fixture:
        fx = f"{fixture}_1"; fixtures.append(f"    {fx} - {fixture}")
        regions.append(region(f"{fixture}_region", (FIXTURE_X[fixture], FIXTURE_Y[variant]), FIXTURE_YAW[fixture]))
        regions += [f"      ({sub}\n          (:target {fx})\n      )" for sub in FIXTURE_SUB[fixture]]
        init.append(f"    (On {fx} main_table_{fixture}_region)")
    spot = 0
    for o, place in items:
        if o == "water_ball":
            balls = [f"water_ball_{i}" for i in range(1, N_WATER + 1)]
            objects.append(f"    {' '.join(balls)} - water_ball")
            init.append(f"    (On water_ball_1 {names[place[1]]})")
            init += [f"    (On water_ball_{i} water_ball_{i - 1})" for i in range(2, N_WATER + 1)]
            continue
        n = inst(o); names[o] = n; objects.append(f"    {n} - {o}")
        if place == "table":
            regions.append(region(f"spot_{spot}_region", PARK[spot], 0.0)); init.append(f"    (On {n} main_table_spot_{spot}_region)"); spot += 1
        elif place[0] == "on": init.append(f"    (On {n} {names[place[1]]})")
        else: init.append(f"    (On {n} {fixture}_1_{place[1]})")
    first = objects[0].split()[0]
    goal = next(l.strip() for l in init if l.split()[1] == first)   # LIBERO needs a goal; unused
    path.write_text(f"(define (problem LIBERO_Tabletop_Manipulation)\n  (:domain robosuite)\n  (:language {language})\n    (:regions\n"
                    + "\n".join(regions) + "\n    )\n\n  (:fixtures\n" + "\n".join(fixtures) + "\n  )\n\n  (:objects\n" + "\n".join(objects)
                    + f"\n  )\n\n  (:obj_of_interest\n    {first}\n  )\n\n  (:init\n" + "\n".join(init)
                    + f"\n  )\n\n  (:goal\n    (And {goal})\n  )\n\n)\n")


def open_env(path):
    import render_scenes as R
    from libero.libero.envs import OffScreenRenderEnv
    import re, os
    for _ in range(12):
        try:
            env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=256, camera_widths=256,
                                     camera_names=["agentview", "robot0_eye_in_hand", "frontview"])
            break
        except (FileNotFoundError, ValueError) as e:   # a mesh / texture not fetched yet
            m = re.search(r"opening file '([^']+)'", str(e))
            f = e.filename if isinstance(e, FileNotFoundError) else (m.group(1) if m else None)
            if not (f and os.path.realpath(f).startswith(str(R.ASSETS.resolve())) and R.fetch_from_zip(os.path.realpath(f))): raise
    for sd in range(20):
        try: env.seed(sd); env.reset(); break
        except Exception as ex:
            if "place all objects" not in str(ex) or sd == 19: raise
    ee = env.env; ee.ignore_done = True; ee.horizon = 10 ** 9
    lay_flat(env)
    return env


def lay_flat(env):
    """turn LAY_FLAT objects by 90 degrees so their thinnest side is vertical (same result every time: deterministic)."""
    import numpy as np
    ee = env.env
    for n in ee.obj_body_id:
        if n[:-2] not in LAY_FLAT: continue
        lo, hi = extents(env, [n])[n]; size = hi - lo
        k = int(np.argmin(size[:2]))
        if size[2] <= size[k]: continue   # already flat
        axis = np.zeros(3); axis[1 - k] = 1.0
        r = np.r_[np.cos(np.pi / 4), np.sin(np.pi / 4) * axis]
        j = joint_of(ee, n); q = np.array(ee.sim.data.get_joint_qpos(j))
        w1, v1, w2, v2 = r[0], r[1:], q[3], q[4:7]
        q[3:] = np.r_[w1 * w2 - v1 @ v2, w1 * v2 + w2 * v1 + np.cross(v1, v2)]
        ee.sim.data.set_joint_qpos(j, q); ee.sim.forward()


def extents(env, only=None):
    """{instance: (lo, hi)}: the object's extent relative to its body position (world axes, current orientation), from mesh
    vertices (primitive geoms: their box corners)."""
    import numpy as np
    ee = env.env; m, d = ee.sim.model, ee.sim.data
    mm = m._model if hasattr(m, "_model") else m
    out = {}
    for n, bid in ee.obj_body_id.items():
        if only and n not in only: continue
        root = m.body_rootid[bid]; pts = []
        for g in range(m.ngeom):
            if m.body_rootid[m.geom_bodyid[g]] != root: continue
            R_, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            if m.geom_type[g] == 7:   # mesh
                mid = m.geom_dataid[g]; a, k = mm.mesh_vertadr[mid], mm.mesh_vertnum[mid]
                v = np.asarray(mm.mesh_vert[a:a + k])
            else:
                c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                v = c + np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3) * h
            pts.append(v @ R_.T + p)
        pts = np.concatenate(pts); b = d.body_xpos[bid]
        out[n] = (pts.min(0) - b, pts.max(0) - b)
    return out


def cup_centre(ee, n):
    """xy of a mug's opening: the middle of its tall (wall) collision geoms; the handle's geoms are shorter and left out."""
    import numpy as np
    m, d = ee.sim.model, ee.sim.data; root = m.body_rootid[ee.obj_body_id[n]]
    gs = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root and m.geom_contype[g]]
    h = {g: (m.geom_aabb[g][5] if m.geom_type[g] == 7 else m.geom_size[g][2]) for g in gs}
    walls = [g for g in gs if h[g] >= 0.75 * max(h.values())]
    xy = np.array([d.geom_xpos[g][:2] for g in walls])
    return (xy.min(0) + xy.max(0)) / 2


def joint_of(ee, n):
    import numpy as np
    return next(j for j in ee.get_object(n).joints if np.shape(ee.sim.data.get_joint_qpos(j)) == (7,))


def lower_until_contact(ee, n, pos, depth=0.08):
    """from pos, move n down in 1 mm steps until it first touches something, then 1 mm back up (no overlap, no drop)."""
    import numpy as np
    m, d = ee.sim.model, ee.sim.data; root = m.body_rootid[ee.obj_body_id[n]]
    mine = lambda g: m.body_rootid[m.geom_bodyid[g]] == root
    for k in range(int(depth / 0.001)):
        put(ee, n, pos - [0, 0, 0.001 * k]); ee.sim.forward()
        if any(mine(d.contact[i].geom1) != mine(d.contact[i].geom2) for i in range(d.ncon)):
            put(ee, n, pos - [0, 0, 0.001 * (k - 1)]); ee.sim.forward(); return pos[2] - 0.001 * (k - 1)
    return pos[2] - depth


def put(ee, n, pos):
    import numpy as np
    j = joint_of(ee, n)
    q = np.array(ee.sim.data.get_joint_qpos(j)); q[:3] = pos
    ee.sim.data.set_joint_qpos(j, q); ee.sim.data.set_joint_qvel(j, np.zeros(6))


def slots(m, ext):
    """table objects of each side -> [(width_y, depth_x, height)] per slot, the max over both sides of the object and what sits on it."""
    out = []
    for sd in SIDES:
        items = m["objects"][sd]
        table = [o for o, p in items if p == "table"]
        for i, o in enumerate(table):
            group = [o] + [x for x, p in items if not isinstance(p, str) and p[0] == "on" and p[1] == o and x != "water_ball"]
            lo = [ext[sd][inst(x)][0] for x in group]; hi = [ext[sd][inst(x)][1] for x in group]
            # width of the group: stacked objects are centred on the support, so use half-widths around the centre
            half = [max((h[k] - l[k]) / 2 for l, h in zip(lo, hi)) for k in (0, 1)]
            height = sum(h[2] - l[2] for l, h in zip(lo, hi))
            if len(out) <= i: out.append([0.0, 0.0, 0.0])
            out[i] = [max(out[i][0], 2 * half[1]), max(out[i][1], 2 * half[0]), max(out[i][2], height)]
    return out


def layout(sl, variant, fx=None):
    """slot sizes -> centre (x, y) per slot. variant 0: importance order, 1: reversed, 2: importance order, rows 5 cm nearer.
    fx: the fixture's box (x0, x1, y0, y1) on the table: a row goes beside it if it fits there, else in front of it."""
    if not sl: return {}   # nothing on the table (everything is on the fixture)
    order = list(range(len(sl)))
    if variant == 1: order = order[::-1]
    total = sum(s[0] for s in sl) + GAP * max(0, len(sl) - 1)
    rows = [order] if total <= ROW_MAX or len(sl) < 2 else [order[: (len(order) + 1) // 2], order[(len(order) + 1) // 2:]]
    rows.sort(key=lambda row: -max(sl[i][2] for i in row))   # tallest row at the back, so it hides nothing
    depth = [max(sl[i][1] for i in row) for row in rows]
    back = ROW_X + (0.05 if variant == 2 else 0.0) - (sum(depth) + GAP * (len(rows) - 1)) / 2   # back edge of the first row
    end = back + sum(depth) + GAP * (len(rows) - 1)
    if end > FRONT: back = max(BACK, back - (end - FRONT))   # too far forward: move the rows back
    if fx: band = (-HALF, fx[2] - GAP) if fx[2] > 0 else (fx[3] + GAP, HALF)   # table width beside the fixture
    where = {}
    for row, d in zip(rows, depth):
        w = sum(sl[i][0] for i in row) + GAP * (len(row) - 1)
        y = -w / 2
        if fx and back < fx[1] + GAP:   # the row would reach the fixture's depth
            if w <= band[1] - band[0]: y = (band[0] + band[1]) / 2 - w / 2
            else: back = fx[1] + GAP
        x = back + d / 2
        for i in row:
            where[i] = (x, y + sl[i][0] / 2); y += sl[i][0] + GAP
        back += d + GAP
    return where


def arrange(env, items, fixture, where, ext, offset=None):
    import numpy as np
    ee = env.env
    top, centre = {}, {}
    table = [o for o, p in items if p == "table"]
    for o, p in items:
        if o == "water_ball": continue
        n = inst(o); lo, hi = ext[n]
        if p == "table":
            x, y = where[table.index(o)]
            pos = np.array([x - (lo[0] + hi[0]) / 2, y - (lo[1] + hi[1]) / 2, TABLE_Z + DROP - lo[2]])
        elif p[0] == "on":
            s = inst(p[1]); cx, cy = centre[s]
            pos = np.array([cx - (lo[0] + hi[0]) / 2, cy - (lo[1] + hi[1]) / 2, top[s] + DROP - lo[2]])
            if offset and o in offset:   # hand-set: moved off-centre, then set down where it first touches
                pos[:2] += offset[o]; pos[2] = lower_until_contact(ee, n, pos, depth=0.15)
        else:   # on a fixture's site (stove burner, cabinet top)
            site = np.array(ee.object_states_dict[f"{fixture}_1_{p[1]}"].get_geom_state()["pos"], dtype=float)
            # x/y: by its own centre, not its box (a pan's box includes the handle: centring that tips it off the burner);
            # z: above the fixture's top, since a site can sit below the surface (the stove's burner site does)
            fb = ee.sim.data.body_xpos[ee.obj_body_id[f"{fixture}_1"]][2] + ext[f"{fixture}_1"][1][2]
            pos = np.array([site[0], site[1], max(site[2], fb) + DROP - lo[2]])
            pos[2] = lower_until_contact(ee, n, pos)
        put(ee, n, pos)
        top[n] = pos[2] + hi[2]; centre[n] = (pos[0] + (lo[0] + hi[0]) / 2, pos[1] + (lo[1] + hi[1]) / 2)
    water = [(o, p) for o, p in items if o == "water_ball"]
    if water:
        mug = inst(water[0][1][1]); ee.sim.forward()
        c = cup_centre(ee, mug); z = ee.sim.data.body_xpos[ee.obj_body_id[mug]][2] + ext[mug][0][2] + 0.03
        balls = sorted((n for n in ee.obj_body_id if n.startswith("water_ball")), key=lambda s: int(s.split("_")[-1]))
        for i, n in enumerate(balls):   # a column in the cup, alternating 5 mm to either side
            put(ee, n, [c[0] + (0.005 if i % 2 else -0.005), c[1], z + 0.032 * i])
    ee.sim.forward()


def settle_and_render(env):
    import numpy as np
    import render_scenes as R
    noop = np.array([0, 0, 0, 0, 0, 0, -1.0])
    for _ in range(SETTLE): obs, *_ = env.step(noop)
    return {c: obs[k][::-1].copy() for c, k in R.CAMERAS.items()}


def visible_px(env):
    import numpy as np
    ee = env.env; m = ee.sim.model
    seg = ee.sim.render(camera_name="agentview", height=256, width=256, segmentation=True)[::-1]
    geom = np.where(seg[..., 0] == 5, seg[..., 1], -1)
    body = np.where(geom >= 0, m.body_rootid[m.geom_bodyid[np.clip(geom, 0, None)]], -1)
    return {n: int((body == m.body_rootid[bid]).sum()) for n, bid in ee.obj_body_id.items() if not n.startswith("water_ball")}


def do_pair(pid, m, r, out, variant=None):
    import imageio
    v = m["variant"] if variant is None else variant
    scn = out / "scenes"; img = out / "images"; scn.mkdir(parents=True, exist_ok=True); img.mkdir(parents=True, exist_ok=True)
    r = renamed(r)
    keys = {sd: pid if r["harm_type"] == "explicit" else f"{pid}__{sd}" for sd in SIDES}
    files_ = {}
    for sd in SIDES:
        if sd == "harmless" and keys[sd] == keys["harmful"]: continue
        f = scn / f"{keys[sd]}.bddl"; write_scene(f, r[sd], m["fixture"], m["objects"][sd], v); files_[sd] = f
    ext = {}
    for sd, f in files_.items():   # one simulator at a time: two open at once garble each other's images
        env = open_env(f)
        try:
            ext[sd] = extents(env)
            if m["fixture"]:
                n = f"{m['fixture']}_1"; b = env.env.sim.data.body_xpos[env.env.obj_body_id[n]]; lo, hi = ext[sd][n]
                fx = (b[0] + lo[0], b[0] + hi[0], b[1] + lo[1], b[1] + hi[1])
        finally: env.close()
    if "harmless" not in ext: ext["harmless"] = ext["harmful"]
    where = layout(slots(m, ext), v, fx if m["fixture"] else None)
    res = dict(ok=True, variant=v, images={}, px={}, hidden_support=[])
    for sd, f in files_.items():
        items = m["objects"][sd]
        env = open_env(f)
        try:
            arrange(env, items, m["fixture"], where, ext[sd], OFFSET.get(pid, {}).get(sd))
            imgs = settle_and_render(env)
            px = visible_px(env)
        finally: env.close()
        files = {}
        for cam, im in imgs.items():
            fn = img / f"{keys[sd]}__{cam}.png"; imageio.imwrite(fn, im); files[cam] = str(fn.relative_to(out))
        res["px"][sd] = px
        sup = {inst(p[1]) for o, p in items if not isinstance(p, str) and p[0] == "on" and o != "water_ball"}
        res["hidden_support"] += [f"{sd}: {s} ({px.get(s, 0)} px)" for s in sorted(sup) if px.get(s, 0) < HIDDEN_PX]
        res["images"][sd] = files
    if "harmless" not in res["images"]: res["images"]["harmless"] = res["images"]["harmful"]
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs"); ap.add_argument("--all", action="store_true"); ap.add_argument("--redo", action="store_true")
    ap.add_argument("--variant", type=int); ap.add_argument("--out", default=str(OUT)); ap.add_argument("--finish", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); res_dir = out / "placed"; res_dir.mkdir(parents=True, exist_ok=True)
    man = json.loads((OUT / "scenes.json").read_text())
    rows = {json.loads(l)["pair_id"]: json.loads(l) for l in open(HERE / "v1_5" / "pairs.jsonl")}
    if a.finish:
        done, hidden, failed, missing = [], [], [], []
        for pid, r in rows.items():
            f = res_dir / f"{pid}.json"
            if not f.exists(): missing.append(pid); continue
            res = json.loads(f.read_text())
            if not res.get("ok"): failed.append((pid, res.get("error", "")[-200:])); continue
            if res["hidden_support"]: hidden.append((pid, r["family"], res["hidden_support"]))
            done.append(dict(renamed(r), harmful_images=res["images"]["harmful"], harmless_images=res["images"]["harmless"]))
        (out / "pairs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in done))
        print(f"{len(done)} pairs rendered; {len(failed)} errors; {len(missing)} not run; {len(hidden)} with a completely hidden support")
        for x in hidden: print("  HIDDEN", *x)
        for x in failed: print("  ERROR", *x)
        return
    import render_scenes as R
    R.setup_env_vars()
    pids = a.pairs.split(",") if a.pairs else [p for p in rows if a.redo or a.pairs or not (res_dir / f"{p}.json").exists()]
    if not (a.pairs or a.all): ap.error("--pairs, --all or --finish")
    t0 = time.time()
    for i, pid in enumerate(pids):
        try: res = do_pair(pid, man[pid], rows[pid], out, a.variant)
        except Exception: res = dict(ok=False, error=traceback.format_exc())
        (res_dir / f"{pid}.json").write_text(json.dumps(res, indent=1))
        el = time.time() - t0
        print(f"[{i + 1}/{len(pids)}] {pid} {'ok' if res['ok'] else 'ERROR'}"
              + (f" hidden: {res['hidden_support']}" if res.get("hidden_support") else "")
              + f"  ({el / (i + 1):.0f}s/pair, ~{el / (i + 1) * (len(pids) - i - 1) / 60:.0f} min left)", flush=True)
        if not res["ok"]: print(res["error"].splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
