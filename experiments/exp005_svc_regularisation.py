"""exp005_svc_regularisation: host and pool regularisation of the stack's n-gram SVC, separately.

A sample weight w on a pool row is the same as fitting that row with C * w, so exp004's best
(C 0.5, pool w 0.3) is host C 0.5 and pool C 0.15. This grid moves the two independently, and
tests class balancing on weighted counts: sklearn's class_weight="balanced" counts rows, not
weights, so once pool rows are down-weighted the classes with many pool rows end up with less
total weight than the others.

Variants (same grouped 5-fold protocol and pool as exp004):
  c{C}_w{W}     LinearSVC C, pool sample weight W, class_weight="balanced"  (c05_w03 = exp004 pool_w03)
  wbal_w{W}     C 0.5, pool weight W, then every class rescaled to equal total weight
Also prints a paired bootstrap over prompts (95% interval) of each variant against pool weight 1.

    python experiments/exp005_svc_regularisation.py c1_w015 c1_w03 c025_w06 wbal_w03
"""

import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC

from ntl.cv import iter_folds
from ntl.data import LABELS, load_train
from ntl.metrics import balanced_accuracy
from ntl.paths import OUTPUT_DIR

sys.path.insert(0, str(Path(__file__).parent))
from exp003_svc_pools import load_pool  # noqa: E402
from exp004_svc_variants import features  # noqa: E402

OUT = OUTPUT_DIR / "exp005_svc_regularisation"
BASE_OOF = OUTPUT_DIR / "exp003_svc_pools" / "oof_pool_v4.npy"  # C 0.5, pool weight 1


def number(s):  # "05" -> 0.5, "015" -> 0.15, "1" -> 1.0
    return float(s[0] + "." + s[1:]) if s.startswith("0") else float(s)


def parse(variant):
    a, b = variant.split("_")
    return dict(C=0.5 if a == "wbal" else number(a[1:]), w=number(b[1:]), wbal=a == "wbal")


def run_fold(variant, train, pool, tr, va):
    v = parse(variant)
    texts = train.text.iloc[tr].tolist() + pool.text.tolist()
    y = np.r_[train.model.iloc[tr].map(LABELS.index).values, pool.model.map(LABELS.index).values]
    X, Xv = features(texts, train.text.iloc[va].tolist())
    sw = np.r_[np.ones(len(tr)), np.full(len(pool), v["w"])]
    if v["wbal"]:
        mass = np.bincount(y, weights=sw, minlength=len(LABELS))
        sw = sw * (mass.sum() / len(LABELS)) / mass[y]
    svc = LinearSVC(C=v["C"], random_state=42, max_iter=5000, class_weight=None if v["wbal"] else "balanced")
    d = svc.fit(X, y, sample_weight=sw).decision_function(Xv)
    e = np.exp(d - d.max(1, keepdims=True))
    return va, e / e.sum(1, keepdims=True)


def paired_bootstrap(y, pred_a, pred_b, groups, n=2000, seed=0):
    """95% interval of BA(b) - BA(a), resampling whole prompts."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in uniq}
    diffs = []
    for _ in range(n):
        idx = np.concatenate([rows[g] for g in rng.choice(uniq, len(uniq))])
        diffs.append(balanced_accuracy(y[idx], pred_b[idx]) - balanced_accuracy(y[idx], pred_a[idx]))
    return np.percentile(diffs, [2.5, 97.5]).round(4).tolist()


def main(variants):
    train = load_train()
    pool = load_pool("pool_v4", train)
    OUT.mkdir(parents=True, exist_ok=True)
    y, groups = train.model.to_numpy(), train.prompt_id.to_numpy()
    base = np.array(LABELS)[np.load(BASE_OOF).argmax(1)]
    results = {}
    for v in variants:
        path = OUT / f"oof_{v}.npy"
        if path.exists():
            oof = np.load(path)
        else:
            oof = np.zeros((len(train), len(LABELS)))
            for va, p in Parallel(n_jobs=3)(delayed(run_fold)(v, train, pool, tr, va) for _, tr, va in iter_folds(train.prompt_id)):
                oof[va] = p
            np.save(path, oof)
        pred = np.array(LABELS)[oof.argmax(1)]
        results[v] = dict(cv=balanced_accuracy(y, pred), gain_vs_w1_ci95=paired_bootstrap(y, base, pred, groups))
        print(v, results[v], flush=True)
    (OUT / f"metrics_{'_'.join(variants)}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["c1_w015", "c1_w03", "c025_w06", "wbal_w03"])
