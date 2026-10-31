"""exp029: ATLAS-8B trained with external real-model passages against ATLAS-8B on the official rows only, out of fold.

Kernels ntl-atlas-cv-off and ntl-atlas-cv-ext ([`build_atlas_cv.py`](../kaggle/atlas/build_atlas_cv.py), training
script [`atlas_train.py`](../kaggle/atlas/atlas_train.py)) train Joseph Adamski's ATLAS adapters (his ntl_atlas code and
training_config) with the same seed on the official train minus prompt fold 0 of ntl.cv, -ext also on 500 passages per
label from crtproduction/name-that-llm-open-pool, and score fold 0 (30 unseen prompts x 18). This script compares them
alone, and the way infer.py uses ATLAS: mixed at weight w into the ensemble's probabilities on rows whose top-2 margin
is below 0.3, then V31's soft Sinkhorn (alpha 0.45 towards 0.65 x predicted + 0.35 x train marginals). The ensemble is
exp015's --plain replica (core + DeBERTa stand-in), out of fold on every row.

ATLAS's scores are nearly flat (prod readout over 18 label adapters: top ~0.067, top-2 margin ~0.002), so the linear
mix shrinks those rows towards uniform and hardly ever moves a pick. The second part multiplies the mixed rows by
ATLAS**k (a log-linear mix, k = temperature x weight), alone and after the 20% linear mix the 0.663 code runs, with a
prompt bootstrap and a cross-fitted choice of k (k picked on 24 of the 30 prompts, scored on the other 6, 5 rotations).

    PYTHONPATH=src python experiments/exp029_atlas_ext.py [off_dir ext_dir]
    PYTHONPATH=src python experiments/exp029_atlas_ext.py pool ext_dir [ext_dir ...]   # power mix on several held-out folds

Default dirs: outputs/exp029_atlas_ext/{off,ext}/atlas_out/, each kernel's output (holdout_scores.npz).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.argv.append("--plain")
import exp015_nlm_core_oof as E  # noqa: E402  (reads --plain at import)
from ntl.cv import prompt_folds  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402
from ntl.paths import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "outputs" / "exp029_atlas_ext"
WEIGHTS = (0.10, 0.20, 0.30)


def soft_sinkhorn(P, alpha=0.45, rho=0.35, iters=100):
    """V31's balancing (exp025); the train marginals are even."""
    marg = np.clip(P.mean(0), 1e-6, None); marg /= marg.sum()
    target = np.clip((1 - rho) * marg + rho / P.shape[1], 1e-6, None); target /= target.sum()
    Q = P.copy()
    for _ in range(iters):
        Q *= len(Q) * target[None, :] / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
        Q /= np.clip(Q.sum(1, keepdims=True), 1e-12, None)
    return (1 - alpha) * np.log(P + 1e-12) + alpha * np.log(Q + 1e-12)


def load_scores(d, ids, fold=0):
    z = np.load(d / "holdout_scores.npz", allow_pickle=False)
    pos = {i: k for k, i in enumerate(z["ids"].tolist())}
    assert set(pos) <= set(ids), f"held-out ids outside fold {fold}"
    cols = [list(z["labels"]).index(l) for l in E.NLM_LABELS]
    return z["scores"][:, cols].astype(np.float64), [pos[i] for i in ids if i in pos], [i in pos for i in ids]


def replica(train):
    """exp015's --plain replica (core + DeBERTa stand-in), out of fold on every train row."""
    y_all = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    z = np.load(E.NLM_OOF, allow_pickle=False)   # its ids are an object array, so only the numeric fields are read
    members = {k: z[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    assert np.array_equal(z["y"], y_all)
    members["svc4"] = np.load(E.OUT / "svc4_oof.npy").astype(np.float64)
    d = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz", allow_pickle=False)
    assert np.array_equal(d["ids"], train.id.to_numpy(dtype=str))
    members["deb"] = E.lsm(d["logits"][:, [list(d["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    return E.blend(members, {**E.CORE, "deb": 1.0})


def pooled(dirs):
    """The power mix on the held-out folds of several -ext runs together (each dir: one kernel's atlas_out)."""
    train = load_train()
    labels = np.array(E.NLM_LABELS)
    y_all = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    folds = prompt_folds(train.prompt_id.to_numpy())
    ids_all = train.id.astype(str).to_numpy()
    rows, A = [], []
    for d in map(Path, dirs):
        f = json.load(open(d / "holdout_metrics.json"))["holdout_fold"]
        fr = np.flatnonzero(folds == f)
        a, o, keep = load_scores(d, ids_all[fr].tolist(), f)
        rows.append(fr[np.array(keep)]); A.append(a[o])
        print(f"{d}: fold {f}, {int(np.sum(keep))} rows")
    rows, A = np.concatenate(rows), np.concatenate(A)
    P = replica(train)[rows]
    y, prompts, truth, fold = y_all[rows], train.prompt_id.to_numpy()[rows], train.model.to_numpy()[rows], folds[rows]
    score = lambda pred, m=slice(None): balanced_accuracy(truth[m], labels[pred[m]])
    s = np.sort(P, 1); low = (s[:, -1] - s[:, -2]) < 0.3

    def power_mix(lin, k):
        Pm = P.copy()
        Pm[low] = (1 - lin) * P[low] + lin * A[low]
        Pm[low] *= A[low] ** k
        return soft_sinkhorn(Pm / Pm.sum(1, keepdims=True)).argmax(1)

    cur = power_mix(0.2, 0)
    R = {"rows": int(len(rows)), "folds": sorted(set(fold.tolist())), "linear20": score(cur)}
    print(f"{len(rows)} rows on folds {R['folds']}: ATLAS alone {score(A.argmax(1)):.4f}, the 20% linear mix {score(cur):.4f}")
    grid = (0, 2, 3, 4, 5, 6, 8)
    preds = {k: power_mix(0.2, k) for k in grid}
    for k in grid[1:]:
        g = paired_bootstrap(y, preds[k], cur, prompts)
        per = {int(f): score(preds[k], fold == f) - score(cur, fold == f) for f in R["folds"]}
        R[f"k{k}"] = {"score": score(preds[k]), "vs_linear20": g, "per_fold": per}
        print(f"  k {k}: {score(preds[k]):.4f} ({g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]); per fold "
              + ", ".join(f"{f}: {v:+.4f}" for f, v in per.items()))
    json.dump(R, open(OUT / "metrics_pooled.json", "w"), indent=1)


def main():
    args = [a for a in sys.argv[1:] if a != "--plain"]
    if args[:1] == ["pool"]:
        return pooled(args[1:])
    off_dir, ext_dir = (args + [None, None])[:2]
    off_dir = OUT / "off" / "atlas_out" if off_dir is None else Path(off_dir)
    ext_dir = OUT / "ext" / "atlas_out" if ext_dir is None else Path(ext_dir)
    train = load_train()
    labels = np.array(E.NLM_LABELS)
    y_all = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    f0 = np.flatnonzero(prompt_folds(train.prompt_id.to_numpy()) == 0)
    ids = train.id.astype(str).to_numpy()[f0].tolist()
    A_off, o_off, keep = load_scores(off_dir, ids)
    A_ext, o_ext, keep2 = load_scores(ext_dir, ids)
    assert keep == keep2, "the two kernels scored different rows"
    rows = f0[np.array(keep)]
    A_off, A_ext = A_off[o_off], A_ext[o_ext]
    y, prompts, truth = y_all[rows], train.prompt_id.to_numpy()[rows], train.model.to_numpy()[rows]
    score = lambda pred: balanced_accuracy(truth, labels[pred])
    R = {"rows": int(len(rows)), "prompts": int(len(set(prompts)))}
    print(f"fold 0: {len(rows)} rows, {len(set(prompts))} prompts")
    A_avg = (A_off + A_ext) / 2
    for name, A in (("off", A_off), ("ext", A_ext), ("avg", A_avg)):
        s = np.sort(A, 1)
        R[f"alone_{name}"] = score(A.argmax(1))
        print(f"ATLAS {name} alone: {R[f'alone_{name}']:.4f} (mean top prob {s[:, -1].mean():.3f}, "
              f"top-2 margin {(s[:, -1] - s[:, -2]).mean():.4f})")
    g = paired_bootstrap(y, A_ext.argmax(1), A_off.argmax(1), prompts)
    R["alone_ext_minus_off"] = g
    print(f"ext - off alone: {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]")

    z = np.load(E.NLM_OOF, allow_pickle=False)   # its ids are an object array, so only the numeric fields are read
    members = {k: z[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    assert np.array_equal(z["y"], y_all)
    members["svc4"] = np.load(E.OUT / "svc4_oof.npy").astype(np.float64)
    d = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz", allow_pickle=False)
    assert np.array_equal(d["ids"], train.id.to_numpy(dtype=str))
    members["deb"] = E.lsm(d["logits"][:, [list(d["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    P = E.blend(members, {**E.CORE, "deb": 1.0})[rows]
    s = np.sort(P, 1); low = (s[:, -1] - s[:, -2]) < 0.3
    base = soft_sinkhorn(P).argmax(1)
    R["replica"] = score(base); R["low_margin_share"] = float(low.mean())
    print(f"\nreplica (core + deb, V31 Sinkhorn) on these rows: {R['replica']:.4f}; top-2 margin < 0.3 on {low.mean():.3f}")
    for w in WEIGHTS:
        preds = {}
        for name, A in (("off", A_off), ("ext", A_ext), ("avg", A_avg)):
            Pm = P.copy(); Pm[low] = (1 - w) * P[low] + w * A[low]
            preds[name] = soft_sinkhorn(Pm).argmax(1)
            R[f"mix{int(w * 100)}_{name}"] = score(preds[name])
        g_off = paired_bootstrap(y, preds["off"], base, prompts)
        g_ext = paired_bootstrap(y, preds["ext"], base, prompts)
        g_d = paired_bootstrap(y, preds["ext"], preds["off"], prompts)
        R[f"mix{int(w * 100)}_gains"] = {"off_vs_replica": g_off, "ext_vs_replica": g_ext, "ext_vs_off": g_d}
        print(f"w {w:.2f}: off {R[f'mix{int(w * 100)}_off']:.4f} ({g_off[0]:+.4f}), ext {R[f'mix{int(w * 100)}_ext']:.4f} "
              f"({g_ext[0]:+.4f}), avg {R[f'mix{int(w * 100)}_avg']:.4f}; ext - off {g_d[0]:+.4f} [{g_d[1]:+.4f}, {g_d[2]:+.4f}]")

    def power_mix(A, lin, k):
        Pm = P.copy()
        Pm[low] = (1 - lin) * P[low] + lin * A[low]
        Pm[low] *= A[low] ** k
        return soft_sinkhorn(Pm / Pm.sum(1, keepdims=True)).argmax(1)

    cur = power_mix(A_off, 0.2, 0)   # the 0.663 code's 20% linear mix (identical picks with either bundle)
    print(f"\npower mix on the top-2 margin < 0.3 rows (x ATLAS**k), against the 20% linear mix ({score(cur):.4f})")
    grid = (0, 1, 2, 3, 4, 5, 6, 8, 10)
    for lin in (0.0, 0.2):
        for name, A in (("off", A_off), ("ext", A_ext)):
            preds = {k: power_mix(A, lin, k) for k in grid}
            line = []
            for k in grid[1:]:
                g = paired_bootstrap(y, preds[k], cur, prompts)
                R[f"pow_lin{int(lin * 100)}_{name}_k{k}"] = {"score": score(preds[k]), "vs_linear20": g}
                line.append(f"k{k} {score(preds[k]):.4f} ({g[0]:+.4f})")
            up = np.unique(prompts); np.random.default_rng(0).shuffle(up)
            cross = np.empty(len(P), int); picks = []
            for r in range(5):
                te = np.isin(prompts, up[r::5])
                kb = max(grid, key=lambda k: (balanced_accuracy(truth[~te], labels[preds[k][~te]]), -k))
                picks.append(kb); cross[te] = preds[kb][te]
            g = paired_bootstrap(y, cross, cur, prompts)
            R[f"pow_lin{int(lin * 100)}_{name}_crossfit"] = {"picks": picks, "score": score(cross), "vs_linear20": g}
            print(f"lin {lin:.1f} {name}: " + ", ".join(line))
            print(f"  cross-fitted k {picks}: {score(cross):.4f}, {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}] vs the linear mix")
    k4 = power_mix(A_ext, 0.2, 4)
    R["k4_rows_changed"] = {"changed": int((k4 != cur).sum()), "fixed": int(((k4 == y) & (cur != y)).sum()),
                            "broken": int(((k4 != y) & (cur == y)).sum())}
    print(f"ext, 20% linear + k 4 against the linear mix: {R['k4_rows_changed']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
