"""exp032: other readouts of ATLAS-8B's token log-probabilities than its argmax-prod, out of fold.

Kernels ntl-atlas-rescore-f0/-f1 ([`build_atlas_rescore.py`](../kaggle/atlas/build_atlas_rescore.py)) reload exp029's
500-per-label hold-out bundles (folds 0 and 1) the way infer.py loads ATLAS and keep every token's log-probability
under each of the 18 label adapters. ATLAS's own readout averages per-token soft wins, exp(-sum_j max(lp_j - lp_k, 0)/T),
over the tokens and T = 1, 2, 4, 8, which leaves the 18 scores nearly flat (top 0.067, top-2 margin 0.002); exp029's
power mix (rows x ATLAS^k) sharpens them afterwards. Here the evidence is read out directly instead: the passage
log-likelihood under each adapter, S_k = sum_t lp_k(t), as R = softmax((S - n mu) / (tau n^gamma)) with n the scored
tokens (gamma 0: the Bayes sum, 1: the mean) and mu each adapter's average offset (0, or estimated label-free on the
scored rows themselves, as infer.py could on the test); also token wins (the share of tokens each adapter predicts
best). Every readout is scored alone and mixed the way the 0.664 code mixes ATLAS: replica rows with top-2 margin < m
become ((1 - 0.2) P + 0.2 R) x R^k, then V31's soft Sinkhorn over both folds. tau (and k) are cross-fitted: chosen on
one fold, scored on the other.

    PYTHONPATH=src:experiments python experiments/exp032_atlas_readout.py [f0_npz f1_npz]

Default inputs: outputs/exp032_atlas_readout/{f0,f1}/rescore/holdout_tokens.npz (the kernels' outputs).
"""

import json
import sys
from pathlib import Path

import numpy as np

import exp029_atlas_ext as X   # noqa: E402  (imports exp015 with --plain)
from ntl.cv import prompt_folds
from ntl.data import load_train
from ntl.metrics import balanced_accuracy
from ntl.oofbank import paired_bootstrap
from ntl.paths import REPO_ROOT

E = X.E
OUT = REPO_ROOT / "outputs" / "exp032_atlas_readout"
TAUS = np.geomspace(1e-3, 1e3, 37)


def load_tokens(path):
    """Per row, in our label order: scored tokens n, token-win shares W, the saved argmax-prod readout, and C, the
    summed token log-probabilities centred per token across the adapters (the passage log-likelihoods S_k up to a
    per-row constant, which no softmax or argmax sees)."""
    z = np.load(path, allow_pickle=False)
    cols = [list(z["labels"]).index(l) for l in E.NLM_LABELS]
    lens = z["lens"].astype(np.int64); assert (lens > 0).all()
    lp = z["lp"][:, cols].astype(np.float64)
    start = np.r_[0, np.cumsum(lens)[:-1]]
    wins = (lp == lp.max(1, keepdims=True)).astype(np.float64)
    centred = lp - lp.mean(1, keepdims=True)
    return dict(ids=z["ids"].tolist(), n=lens.astype(np.float64),
                W=np.add.reduceat(wins, start) / lens[:, None], C=np.add.reduceat(centred, start),
                prod=z["prod"][:, cols].astype(np.float64))


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True); e = np.exp(Z)
    return e / e.sum(1, keepdims=True)


def main():
    paths = [Path(p) for p in sys.argv[1:] if p != "--plain"] or [OUT / f / "rescore" / "holdout_tokens.npz" for f in ("f0", "f1")]
    train = load_train()
    labels = np.array(E.NLM_LABELS)
    y_all = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    folds = prompt_folds(train.prompt_id.to_numpy())
    pos = {i: k for k, i in enumerate(train.id.astype(str))}
    parts = [load_tokens(p) for p in paths]
    rows = np.concatenate([[pos[i] for i in d["ids"]] for d in parts])
    T = {k: np.concatenate([d[k] for d in parts]) for k in ("n", "W", "C", "prod")}
    P = X.replica(train)[rows]
    y, prompts, truth, fold = y_all[rows], train.prompt_id.to_numpy()[rows], train.model.to_numpy()[rows], folds[rows]
    fl = sorted(set(fold.tolist()))
    score = lambda pred, m=slice(None): balanced_accuracy(truth[m], labels[pred[m]])
    s = np.sort(P, 1); margin = s[:, -1] - s[:, -2]
    R = {"rows": int(len(rows)), "folds": fl}
    print(f"{len(rows)} rows on folds {fl}; replica alone {score(X.soft_sinkhorn(P).argmax(1)):.4f}")

    def evidence(gamma, bias, m):
        """log-scale evidence per row before 1/tau; mu estimated (label-free) on the rows the readout is applied to."""
        mu = T["C"][m].sum(0) / T["n"][m].sum() if bias else 0.0
        return (T["C"] - T["n"][:, None] * mu) / T["n"][:, None] ** gamma

    print("\nalone (argmax; tau and gamma do not move it):")
    for name, A in (("argmax-prod", T["prod"]), ("sum", T["C"]), ("sum - n mu", evidence(0, True, slice(None))),
                    ("token wins", T["W"])):
        R[f"alone|{name}"] = score(A.argmax(1))
        print(f"  {name:12s} {R[f'alone|{name}']:.4f}  " + "  ".join(f"fold {f} {score(A.argmax(1), fold == f):.4f}" for f in fl))

    def mixed(m, A_lin, A_pow, k):
        Pm = P.copy()
        Pm[m] = (1 - 0.2) * P[m] + 0.2 * A_lin[m]
        Pm[m] *= A_pow[m] ** k
        return X.soft_sinkhorn(Pm / Pm.sum(1, keepdims=True)).argmax(1)

    def crossfit(preds):
        """preds: {setting: prediction}; the setting is chosen on the other fold(s), ties to the first listed."""
        keys = list(preds); out = np.empty(len(rows), int); picks = {}
        for f in fl:
            te = fold == f
            kb = max(keys, key=lambda k: score(preds[k], ~te)); picks[int(f)] = kb; out[te] = preds[kb][te]
        return out, picks

    cur = mixed(margin < 0.3, T["prod"], T["prod"], 0)   # the 0.664 code: 20% linear on the margin < 0.3 rows
    R["linear20"] = score(cur)
    for thr in (0.3, 1.01):
        m = margin < thr
        print(f"\nmixed on rows with top-2 margin < {thr} ({m.mean():.3f} of rows), against the 0.664 code's 20% linear mix "
              f"on the margin < 0.3 rows ({score(cur):.4f})")
        fams = {"argmax-prod^k": {f"k{k}": mixed(m, T["prod"], T["prod"], k) for k in (2, 3, 4, 5, 6, 8)}}
        for gamma in (0.0, 0.5, 1.0):
            for bias in (False, True):
                Z = evidence(gamma, bias, m)
                nm = f"sum/(tau n^{gamma:g})" + (" - mu" if bias else "")
                fams[nm + ", lin+pow"] = {f"tau{t:.3g}": mixed(m, softmax(Z / t), softmax(Z / t), 1) for t in TAUS}
                fams[nm + ", prod lin, pow"] = {f"tau{t:.3g}": mixed(m, T["prod"], softmax(Z / t), 1) for t in TAUS}
        fams["token wins, prod lin, pow"] = {f"tau{t:.3g}": mixed(m, T["prod"], softmax(T["W"] / t), 1) for t in TAUS}
        for nm, preds in fams.items():
            best = max(preds, key=lambda k: score(preds[k]))
            cf, picks = crossfit(preds)
            g = paired_bootstrap(y, cf, cur, prompts)
            per = {int(f): score(cf, fold == f) - score(cur, fold == f) for f in fl}
            R[f"m{thr}|{nm}"] = {"best_in_sample": [best, score(preds[best])], "crossfit": score(cf), "picks": picks,
                                 "vs_linear20": g, "per_fold": per}
            print(f"  {nm:34s} best {best:>9s} {score(preds[best]):.4f} | cross-fitted {score(cf):.4f} "
                  f"({g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]) picks {list(picks.values())} per fold "
                  + ", ".join(f"{v:+.4f}" for v in per.values()))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
