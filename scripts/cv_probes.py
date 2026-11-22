"""Grouped k-fold cross-validated probes + paired significance test vs the bag-of-words baseline.

Why: the fixed 300/100/100 split tests on ~1/5 of the data, so CIs are wide. Here every row is tested exactly once.
CV units: hazard groups, merged when they share an exact (scene, instruction) input or a pair (union-find), so no input
is ever in both train and test of a fold. Folds balance row counts.
Per outer fold k: test = fold k, inner val = fold k+1, train = rest. C is picked on inner-val AUROC, then the probe is refit
on train+val. "selected" = the layer picked per fold on inner-val AUROC (never looks at test) -> the headline number.
Baseline "bow" = probe on mean_instr at layer 0 (mean of raw token embeddings, no context = bag of words), same procedure.
Significance: paired cluster bootstrap of (probe correct - bow correct) over CV units; 95% CI and one-sided p = P(diff <= 0).

  OMP_NUM_THREADS=1 python scripts/cv_probes.py --acts pi05_base=cache/acts/pi05_base.npz --feature last_instr --out results/probes_cv
Outputs: {out}/{name}_{feature}.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_probes import fit_logreg, auroc, CS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def cv_units(pid, label, group):
    """union-find over group, exact input and pair -> unit id per row."""
    tab = {(r["pair_id"], r["label"]): r for r in map(json.loads, open(ROOT / "results/dataset_review/probe_table.jsonl"))}
    par = {}
    def f(x):
        while par.setdefault(x, x) != x: x = par[x]
        return x
    keys = []
    for p, l, g in zip(pid, label, group):
        m = tab[(p, int(l))]
        k = "k:" + m["scene_key"] + "||" + m["instruction"].lower()
        par[f("g:" + g)] = f(k); par[f("p:" + p)] = f(k)
        keys.append(k)
    roots = [f(k) for k in keys]
    u = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    return np.array([u[r] for r in roots])


def make_folds(units, k):
    sizes = np.bincount(units)
    fold_of, load = np.zeros(len(sizes), int), np.zeros(k)
    for u in np.argsort(-sizes, kind="stable"):  # largest first into the lightest fold
        j = int(np.argmin(load)); fold_of[u] = j; load[j] += sizes[u]
    return fold_of[units]


def fit_eval(Xtr, ytr, Xva, yva, Xte):
    """pick C on val AUROC, refit on train+val; -> (val_auroc, test scores)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    best = None
    for C in CS:
        w = fit_logreg((Xtr - mu) / sd, ytr, C)
        a = auroc(((Xva - mu) / sd) @ w[:-1] + w[-1], yva)
        if best is None or a > best[0]: best = (a, C)
    Xa, ya = np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva])
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    w = fit_logreg((Xa - mu) / sd, ya, best[1])
    return best[0], ((Xte - mu) / sd) @ w[:-1] + w[-1]


def boot_diff(a, b, units, rng, B=10000):
    """paired cluster bootstrap of mean(a) - mean(b)."""
    ug = np.unique(units)
    S = np.array([[a[units == u].sum(), b[units == u].sum(), (units == u).sum()] for u in ug], float)
    idx = rng.integers(0, len(ug), (B, len(ug)))
    s = S[idx].sum(1)
    d = (s[:, 0] - s[:, 1]) / s[:, 2]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


def boot_acc(a, units, rng, B=10000):
    lo, hi, _ = boot_diff(a, np.zeros_like(a), units, rng, B)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True, help="name=path.npz")
    ap.add_argument("--feature", default="last_instr", choices=["last_instr", "mean_instr", "mean_img", "last_tok"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results/probes_cv"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    name, path = a.acts.split("=", 1)
    z = np.load(path, allow_pickle=False)
    X = z[a.feature].astype(np.float32)
    bow_X = z["mean_instr"][:, 0].astype(np.float32)
    y, pid, group = z["label"].astype(int), z["pair_id"], z["group"]
    N, nL, _ = X.shape
    units = cv_units(pid, y, group)
    fold = make_folds(units, a.folds)
    print(f"{name} {a.feature}: N={N} units={units.max() + 1} fold sizes={np.bincount(fold).tolist()}", flush=True)

    score = np.full((nL, N), np.nan); vauc = np.zeros((a.folds, nL)); bow = np.full(N, np.nan); sel = np.full(N, np.nan)
    sel_layers = []
    t0 = time.time()
    for k in range(a.folds):
        te, va = fold == k, fold == (k + 1) % a.folds
        tr = ~te & ~va
        _, bow[te] = fit_eval(bow_X[tr], y[tr], bow_X[va], y[va], bow_X[te])
        for l in range(nL):
            vauc[k, l], score[l, te] = fit_eval(X[tr, l], y[tr], X[va, l], y[va], X[te, l])
        b = int(np.argmax(vauc[k])); sel_layers.append(b); sel[te] = score[b, te]
        print(f"  fold {k}: selected layer {b}  ({time.time() - t0:.0f}s)", flush=True)

    ok_bow = ((bow > 0).astype(int) == y).astype(float)
    def summary(s, m=None):
        m = np.ones(N, bool) if m is None else m
        ok = ((s > 0).astype(int) == y).astype(float)
        lo, hi = boot_acc(ok[m], units[m], rng)
        dlo, dhi, p = boot_diff(ok[m], ok_bow[m], units[m], rng)
        return {"acc": float(ok[m].mean()), "acc_ci": [lo, hi], "auroc": float(auroc(s[m], y[m])),
                "minus_bow": float(ok[m].mean() - ok_bow[m].mean()), "minus_bow_ci": [dlo, dhi], "p_one_sided": p}
    res = {"model": name, "feature": a.feature, "folds": a.folds, "n": int(N), "n_units": int(units.max() + 1),
           "bow_acc": float(ok_bow.mean()), "bow_acc_ci": list(boot_acc(ok_bow, units, rng)),
           "selected_layers_per_fold": sel_layers, "selected": summary(sel),
           "layers": {l: summary(score[l]) for l in range(nL)}}
    # subsets (probes are still trained on everything; only evaluation is restricted)
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / "libero_safety/minimal_pairs_500.jsonl")}
    ht = np.array([pairs[p]["harm_type"] for p in pid])
    res["subsets"] = {}
    for h in sorted(set(ht)):
        m = ht == h
        res["subsets"][h] = {"n": int(m.sum()), "n_units": int(len(np.unique(units[m]))), "bow_acc": float(ok_bow[m].mean()),
                             "selected": summary(sel, m), "layers": {l: summary(score[l], m) for l in range(nL)}}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"{name}_{a.feature}.json", "w"), indent=1)
    np.savez_compressed(out / f"{name}_{a.feature}_scores.npz", score=score, sel=sel, bow=bow, y=y, units=units, fold=fold, pair_id=pid)
    s = res["selected"]
    print(f"== {name} {a.feature}: bow {res['bow_acc']:.3f} | selected (layers {sel_layers}) acc {s['acc']:.3f} "
          f"diff {s['minus_bow']:+.3f} CI [{s['minus_bow_ci'][0]:+.3f}, {s['minus_bow_ci'][1]:+.3f}] p={s['p_one_sided']:.4f}")
    for l in range(nL):
        r = res["layers"][l]
        print(f"  L{l:2d} acc {r['acc']:.3f}  diff {r['minus_bow']:+.3f} [{r['minus_bow_ci'][0]:+.3f}, {r['minus_bow_ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
