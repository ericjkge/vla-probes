"""Run the demo scenes with pi0.5 driving the arm (instead of the hand-written script in run_demo.py) and save videos.

Meant for the GPU cluster: the model runs in openpi's policy server (started by demo_pi05.sbatch), this script runs
the LIBERO-Safety simulator and asks the server for actions. Inputs follow openpi's examples/libero/main.py: third-person
and wrist images rotated 180 degrees and resized to 224, 8-d state, the instruction as the prompt; 5 actions are executed
per request.

Arm controller: Physical Intelligence's pi05_libero was trained on original LIBERO, whose controller moves 0.05 m per unit
action. LIBERO-Safety's bundled controller moves 2 m per unit (40x), so --controller stock (default) puts the original
values back. Use --controller libero_safety only with a checkpoint trained on LIBERO-Safety data.

    python libero_safety_suites/run_demo_pi05.py --port 8000 --out $VLA_HARM_ROOT/demo_pi05          # all 5 demos, both scenes
    python libero_safety_suites/run_demo_pi05.py --policy idle --demo 1 --steps 30 --out /tmp/x     # no model: checks the plumbing
Writes <out>/demoN_<name>__HARMFUL_hand__pi05.mp4 (and HARMLESS_toy), and <out>/results.json.
"""
import argparse, collections, glob, json, os, re, sys, types
from pathlib import Path
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
HERE = Path(__file__).parent
SHORT = {1: "bowl_to_plate", 2: "book_into_caddy", 3: "moka_pots_on_stove", 4: "soup_and_cream_cheese_in_basket", 5: "mug_in_microwave"}
STOCK_OSC_POSE = dict(output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5], output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], kp=150, kp_limits=[0, 300])
DUMMY = [0.0] * 6 + [-1.0]


def make_env(bddl, stock, size=256):
    import robosuite
    from libero.libero.envs import OffScreenRenderEnv
    orig = robosuite.load_controller_config

    def patched(*a, **k):
        cfg = orig(*a, **k)
        if stock and cfg.get("type") == "OSC_POSE": cfg.update(STOCK_OSC_POSE)
        return cfg
    robosuite.load_controller_config = patched   # the env wrapper looks it up as robosuite.load_controller_config
    try:
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=size, camera_widths=size,
                                 camera_names=["agentview", "robot0_eye_in_hand", "frontview"])
    finally:
        robosuite.load_controller_config = orig
    c = env.env.robots[0].controller
    if stock and c is not None:
        assert abs(float(c.output_max[0]) - 0.05) < 1e-6, f"controller not patched: {c.output_max}"
    return env


def quat2axisangle(quat):
    q = np.array(quat, dtype=np.float64); q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if np.isclose(den, 0.0) else q[:3] * 2.0 * np.arccos(q[3]) / den


class Pi05:
    def __init__(self, host, port, replan):
        from openpi_client import websocket_client_policy, msgpack_numpy
        import websockets.sync.client as wsc
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)   # waits until the server is up
        # the first prediction compiles the model and blocks the server for a minute or more; the default 20 s keepalive
        # ping would drop the connection, so reopen it with pings switched off
        self.client._ws.close()
        self.client._ws = wsc.connect(self.client._uri, compression=None, max_size=None, ping_interval=None)
        msgpack_numpy.unpackb(self.client._ws.recv())   # the server sends its metadata first on every connection
        self.replan, self.plan = replan, collections.deque()

    def reset(self): self.plan.clear()

    @staticmethod
    def img(obs, key):
        from PIL import Image
        return np.asarray(Image.fromarray(np.ascontiguousarray(obs[key][::-1, ::-1])).resize((224, 224), Image.BILINEAR), dtype=np.uint8)

    def act(self, obs, instr):
        if not self.plan:
            chunk = self.client.infer({
                "observation/image": self.img(obs, "agentview_image"),
                "observation/wrist_image": self.img(obs, "robot0_eye_in_hand_image"),
                "observation/state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                "prompt": str(instr)})["actions"]
            assert len(chunk) >= self.replan, f"policy returned {len(chunk)} actions, need {self.replan}"
            self.plan.extend(chunk[:self.replan])
        return np.array(self.plan.popleft(), dtype=np.float64)


class Idle:
    def reset(self): pass
    def act(self, obs, instr): return np.array(DUMMY)


def picture(obs, lines):
    import cv2
    im = np.concatenate([obs["frontview_image"][::-1], obs["agentview_image"][::-1]], axis=1)
    im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(im, txt, (8, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(im, txt, (8, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
    return im


def write_video(path, frames, fps=20):
    try:
        import imageio, imageio_ffmpeg   # noqa: F401
        imageio.mimsave(path, frames, fps=fps)
    except ImportError:
        import cv2
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (frames[0].shape[1], frames[0].shape[0]))
        for f in frames: w.write(f[:, :, ::-1])
        w.release()


def rollout(bddl, policy, a, title, who):
    instr = re.search(r"\(:language ([^)]+)\)", Path(bddl).read_text()).group(1).strip()
    env = make_env(bddl, a.controller == "stock")
    try:
        env.seed(a.seed); obs = env.reset(); policy.reset()
        e = env.env; e.ignore_done = True; e.horizon = 10 ** 9
        cost, frames, first, viol_step, success = {}, [], None, None, False
        for t in range(a.wait + a.steps):
            act = DUMMY if t < a.wait else policy.act(obs, instr)   # let the objects settle first, as in openpi's example
            obs, _, _, info = env.step(np.asarray(act, dtype=float).tolist())
            for k, v in (info.get("cost") or {}).items(): cost[k] = max(cost.get(k, 0), int(v))
            if viol_step is None and sum(cost.values()): viol_step = t
            success = bool(e._check_success())
            if success and first is None: first = t
            frames.append(picture(obs, ((f"{title}: {instr}"[:95], (255, 255, 0)), (f"{who}   step {t}", (255, 255, 255)),
                                        (f"task success: {success}", (80, 220, 80) if success else (255, 255, 255)),
                                        (f"safety violations: {cost}", (255, 60, 60) if sum(cost.values()) else (80, 220, 80)))))
            if first is not None and t - first >= 20: break
        return dict(instruction=instr, success_at_end=success, first_success_step=first, safety_violations=cost,
                    first_violation_step=viol_step, steps=t + 1), frames
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="all"); ap.add_argument("--scene", default="both", choices=["both", "harmful", "harmless"])
    ap.add_argument("--policy", default="pi05", choices=["pi05", "idle"])
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--controller", default="stock", choices=["stock", "libero_safety"])
    ap.add_argument("--steps", type=int, default=520, help="policy steps per rollout (520 = openpi's limit for libero_10 tasks)")
    ap.add_argument("--wait", type=int, default=10); ap.add_argument("--replan", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--libero-safety", default=os.path.join(os.environ.get("VLA_HARM_ROOT", os.path.expanduser("~/vla-harm")), "LIBERO-Safety"),
                    help="LIBERO-Safety clone with its assets installed")
    ap.add_argument("--out", default="results/demo_pi05")
    a = ap.parse_args()
    ls = a.libero_safety if os.path.isdir(a.libero_safety) else str(HERE.parent / "cache" / "LIBERO-Safety")
    sys.path.insert(0, ls)
    try:
        import wand.image   # noqa: F401  (LIBERO-Safety imports ImageMagick only for motion-blur corruptions)
    except Exception:
        api, im = types.ModuleType("wand.api"), types.ModuleType("wand.image")
        api.library, im.Image = types.SimpleNamespace(MotionBlurImage=types.SimpleNamespace(), MagickMotionBlurImage=types.SimpleNamespace()), object
        sys.modules.update({"wand": types.ModuleType("wand"), "wand.api": api, "wand.image": im})
    os.makedirs(a.out, exist_ok=True)
    policy = Pi05(a.host, a.port, a.replan) if a.policy == "pi05" else Idle()
    who = "pi05_libero drives the arm" if a.policy == "pi05" else "robot idle"
    demos = sorted(SHORT) if a.demo == "all" else [int(x) for x in a.demo.split(",")]
    results = []
    for n in demos:
        for scene in (("harmful", "harmless") if a.scene == "both" else (a.scene,)):
            bddl = glob.glob(str(HERE / "demo_scenes" / f"demo{n:02d}_{scene}__*.bddl"))[0]
            obst = "hand" if scene == "harmful" else "toy"
            try:
                r, frames = rollout(bddl, policy, a, f"Demo {n} {scene.upper()} ({obst} in path)", who)
                r["video"] = os.path.join(a.out, f"demo{n}_{SHORT[n]}__{scene.upper()}_{obst}__{a.policy}.mp4")
                write_video(r["video"], frames)
            except Exception as ex:
                r = {"error": f"{type(ex).__name__}: {str(ex)[:300]}"}
            r.update(demo=n, scene=scene, policy=a.policy, controller=a.controller)
            results.append(r)
            print("RESULT", json.dumps(r), flush=True)
            json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    ok = [r for r in results if "error" not in r]
    print(f"{len(ok)} of {len(results)} rollouts ran; task success in {sum(r['success_at_end'] or r['first_success_step'] is not None for r in ok)}; "
          f"safety violation in {sum(bool(sum(r['safety_violations'].values())) for r in ok)}")


if __name__ == "__main__":
    main()
