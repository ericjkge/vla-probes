"""Run a demo scene live: a scripted robot (hand-written waypoints, NOT pi05) carries out the instruction in the simulator
while a window shows it moving, with the simulator's own success check and safety cost on screen.

    MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/run_demo.py --demo 1 --scene harmful
    MUJOCO_GL=cgl .venv/bin/python libero_safety_suites/run_demo.py --demo 1 --scene harmless
    ... --fast      rush: the robot moves before the obstacle is out of the way, so the simulator records a safety violation
    ... --wait      wait for the obstacle to leave before moving (the safe way to do it)
    ... --no-window headless (prints the result only)      ... --save out.mp4  also write a video
Keys in the window: q quits, space pauses.
"""
import argparse, glob, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "libero_safety"))
import render_scenes as R          # noqa: E402
import check_executable as C       # noqa: E402

GAIN, VMAX = 0.8, 0.012   # eef target moves this fraction of the remaining distance per step, at most VMAX metres
# demo -> list of (object to carry, place target, grasp offset from the object's centre, drop offset from the target's centre)
PLANS = {
    1: [("akita_black_bowl_1", "plate_1", (0.0, 0.047, 0.03), (0.0, -0.047, 0.09))],
    2: [("black_book_1", "desk_caddy_1_back_contain_region", (0.0, 0.0, 0.08), (0.0, 0.0, 0.14))],
    3: None,   # moka pots: needs a rotated grasp on the handle, not scripted yet (robot stays still)
    4: [("alphabet_soup_1", "basket_1", (0.0, 0.0, 0.03), (0.03, 0.0, 0.16)),
        ("cream_cheese_1", "basket_1", (0.0, 0.0, 0.0), (-0.03, 0.0, 0.16))],
    5: None,   # mug into microwave + close door: not scripted yet (robot stays still)
}
CLEAR = {1: 1.16, 2: 1.20, 4: 0.30}   # height the gripper travels at


class Runner:
    def __init__(self, bddl, window, save, size=448):
        from libero.libero.envs import OffScreenRenderEnv
        C.make_env(bddl).close()
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=size, camera_widths=size, camera_names=["agentview", "frontview"])
        self.env.seed(0); self.obs = self.env.reset()
        self.e = self.env.env; self.e.ignore_done = True; self.e.horizon = 10 ** 9
        self.window, self.save, self.writer, self.title = window, save, None, ""
        self.cost, self.t, self.phase, self.success = {}, 0, "", False
        self.out_max = np.asarray(self.e.robots[0].controller.output_max[:3], dtype=float)   # metres per unit of action
        self.gain, self.vmax = GAIN, VMAX
        self.obstacle = next(o for o in self.e.obj_body_id if "with_hand" in o or o.split("_")[0] in ("toy", "billiard"))

    def pos(self, name):
        if name in self.e.obj_body_id: return np.array(self.e.sim.data.body_xpos[self.e.obj_body_id[name]])
        return np.array(self.e.object_states_dict[name].get_geom_state()["pos"], dtype=float)   # a region of a fixture (burner, compartment...)

    def step(self, a):
        self.obs, _, _, info = self.env.step(np.asarray(a, dtype=float))
        self.t += 1
        for k, v in (info.get("cost") or {}).items(): self.cost[k] = max(self.cost.get(k, 0), int(v))
        self.success = self.success or bool(self.e._check_success())
        if self.window or self.save: self.show()

    def show(self):
        import cv2
        im = np.concatenate([self.obs["frontview_image"][::-1], self.obs["agentview_image"][::-1]], axis=1)[:, :, ::-1].copy()
        viol = sum(self.cost.values())
        for i, (txt, col) in enumerate(((self.title, (0, 255, 255)), (f"step {self.t}  {self.phase}", (255, 255, 255)),
                                        (f"task success: {self.success}", (80, 220, 80) if self.success else (255, 255, 255)),
                                        (f"safety violations: {self.cost}", (60, 60, 255) if viol else (80, 220, 80)))):
            cv2.putText(im, txt, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(im, txt, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        if self.save:
            if self.writer is None: self.writer = cv2.VideoWriter(self.save, cv2.VideoWriter_fourcc(*"avc1"), 20, (im.shape[1], im.shape[0]))
            self.writer.write(im)
        if self.window:
            cv2.imshow("LIBERO-Safety demo (q quit, space pause)", im)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "): k = cv2.waitKey(0) & 0xFF
            if k == ord("q"): raise KeyboardInterrupt

    def goto(self, target, grip, phase, tol=0.012, max_steps=400):
        self.phase = phase
        last, still = None, 0
        for _ in range(max_steps):
            p = self.obs["robot0_eef_pos"]; d = np.asarray(target) - p
            still = still + 1 if last is not None and np.linalg.norm(p - last) < 5e-4 else 0   # the arm has stopped getting closer
            if np.linalg.norm(d) < tol or still > 12: break
            last = p.copy()
            self.step(np.r_[np.clip(d * self.gain, -self.vmax, self.vmax) / self.out_max, 0, 0, 0, grip])

    def hold(self, grip, n, phase):
        self.phase = phase
        for _ in range(n): self.step([0, 0, 0, 0, 0, 0, grip])

    def carry(self, obj, tgt, g_off, d_off, clear):
        p = self.pos(obj) + np.array(g_off)
        self.goto([p[0], p[1], clear], -1, f"move above {obj}", tol=0.004)
        self.goto([p[0], p[1], p[2] + 0.05], -1, "reach down", tol=0.004)
        self.goto(p, -1, "reach down", tol=0.004)
        self.hold(1, 18, "close gripper")
        self.goto([p[0], p[1], clear], 1, "lift")
        q = self.pos(tgt) + np.array(d_off)
        self.goto([q[0], q[1], clear], 1, f"carry to {tgt}")
        self.goto(q, 1, "lower")
        self.hold(-1, 18, "open gripper")
        self.goto([q[0], q[1], clear], -1, "retreat")

    def wait_for_obstacle(self, pts, margin=0.22, max_steps=600):
        self.phase = "waiting for the obstacle to move away"
        for _ in range(max_steps):
            o = self.pos(self.obstacle)[:2]
            if all(np.linalg.norm(o - np.asarray(p)[:2]) > margin for p in pts): return
            self.step([0, 0, 0, 0, 0, 0, -1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=1, choices=sorted(PLANS))
    ap.add_argument("--scene", default="harmful", choices=["harmful", "harmless"])
    ap.add_argument("--wait", action="store_true"); ap.add_argument("--fast", action="store_true"); ap.add_argument("--no-window", action="store_true"); ap.add_argument("--save")
    a = ap.parse_args()
    R.setup_env_vars()
    bddl = glob.glob(str(C.OUT / f"demo{a.demo:02d}_{a.scene}__*.bddl"))[0]
    r = Runner(bddl, not a.no_window, a.save)
    if a.fast: r.vmax = 0.04
    print("instruction:", re.search(r"\(:language ([^)]+)\)", Path(bddl).read_text()).group(1).strip(), "| obstacle:", r.obstacle, flush=True)
    instr = re.search(r"\(:language ([^)]+)\)", Path(bddl).read_text()).group(1).strip()
    r.title = f"Demo {a.demo} {a.scene.upper()} ({'hand' if a.scene == 'harmful' else 'toy'} in path): {instr}"
    try:
        r.hold(-1, 10, "start")
        if PLANS[a.demo] is None: r.hold(-1, 200, "robot idle (no script for this demo yet)")
        for obj, tgt, g, d in PLANS[a.demo] or []:
            if a.wait: r.wait_for_obstacle([r.pos(obj), r.pos(tgt), r.obs["robot0_eef_pos"]])
            r.carry(obj, tgt, g, d, CLEAR[a.demo])
        r.hold(-1, 40, "done")
    except KeyboardInterrupt:
        pass
    print(f"RESULT demo {a.demo} {a.scene}: success={r.success} safety_violations={r.cost} steps={r.t}", flush=True)
    if r.writer is not None: r.writer.release()
    r.env.close()


if __name__ == "__main__":
    main()
