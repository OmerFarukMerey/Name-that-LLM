"""exp004_svc_variants: cheap changes to the stack's n-gram SVC, scored on host passages.

Same grouped 5-fold protocol as exp003 (pool_v4, train-like pool rows dropped).
Variants:
  base        the upstream member: LinearSVC C=0.5, balanced classes, pool weight 1
  pool_wNN    pool rows at weight 0.NN, e.g. pool_w05 = 0.5 (host rows matter more)
  domain_w    pool rows weighted by a host-vs-pool domain classifier, p(host)/p(pool),
              normalised to mean 1 per class: importance weighting toward host style
  c025 / c1   C = 0.25 / 1.0

    python experiments/exp004_svc_variants.py base domain_w
"""

import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from ntl.cv import iter_folds
from ntl.data import LABELS, load_train
from ntl.metrics import balanced_accuracy
from ntl.paths import OUTPUT_DIR

sys.path.insert(0, str(Path(__file__).parent))
from exp002_pseudo_label_sim import VIEWS, shape_text, small_words_text  # noqa: E402
from exp003_svc_pools import load_pool  # noqa: E402

OUT = OUTPUT_DIR / "exp004_svc_variants"


def features(fit_texts, test_texts):
    A, B = [], []
    for view, lo, hi, max_features, weight in VIEWS:
        prep = shape_text if view == "shape" else small_words_text if view == "small" else (lambda s: s)
        analyzer = "char" if view == "shape" else "word" if view == "small" else view
        opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                    max_features=max_features, dtype=np.float32)
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vec = TfidfVectorizer(**opts)
        A.append(vec.fit_transform([prep(t) for t in fit_texts]) * weight)
        B.append(vec.transform([prep(t) for t in test_texts]) * weight)
    return hstack(A, format="csr"), hstack(B, format="csr")


def domain_weights(X, n_host, y):
    """p(host)/p(pool) for the pool rows from a cross-fitted logistic domain classifier."""
    d = np.r_[np.ones(n_host), np.zeros(X.shape[0] - n_host)]
    pool_idx = np.arange(n_host, X.shape[0])
    p = np.zeros(len(pool_idx))
    halves = np.array_split(np.random.default_rng(0).permutation(len(pool_idx)), 2)
    for k in range(2):
        held = pool_idx[halves[k]]
        fit_rows = np.r_[np.arange(n_host), pool_idx[halves[1 - k]]]
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X[fit_rows], d[fit_rows])
        p[halves[k]] = clf.predict_proba(X[held])[:, 1]
    w = np.clip(p / (1 - p), 0.05, 20)
    yp = y[n_host:]
    for c in np.unique(yp):  # keep each class's total pool weight unchanged
        w[yp == c] /= w[yp == c].mean()
    return w


def run_fold(variant, train, pool, tr, va):
    texts = train.text.iloc[tr].tolist() + pool.text.tolist()
    y = np.r_[train.model.iloc[tr].map(LABELS.index).values, pool.model.map(LABELS.index).values]
    X, Xv = features(texts, train.text.iloc[va].tolist())
    n_host = len(tr)
    sw = np.ones(len(y))
    if variant.startswith("pool_w"):  # pool_w05 -> 0.5, pool_w03 -> 0.3
        sw[n_host:] = float("0." + variant[len("pool_w0"):])
    if variant == "domain_w":
        sw[n_host:] = domain_weights(X, n_host, y)
    C = {"c025": 0.25, "c1": 1.0}.get(variant, 0.5)
    svc = LinearSVC(C=C, random_state=42, max_iter=5000, class_weight="balanced").fit(X, y, sample_weight=sw)
    d = svc.decision_function(Xv)
    e = np.exp(d - d.max(1, keepdims=True))
    return va, e / e.sum(1, keepdims=True)


def main(variants):
    train = load_train()
    pool = load_pool("pool_v4", train)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for v in variants:
        oof = np.zeros((len(train), len(LABELS)))
        for va, p in Parallel(n_jobs=3)(delayed(run_fold)(v, train, pool, tr, va) for _, tr, va in iter_folds(train.prompt_id)):
            oof[va] = p
        results[v] = balanced_accuracy(train.model, np.array(LABELS)[oof.argmax(1)])
        print(v, round(results[v], 4), flush=True)
        np.save(OUT / f"oof_{v}.npy", oof)
    (OUT / f"metrics_{'_'.join(variants)}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["base", "pool_w05", "domain_w"])
