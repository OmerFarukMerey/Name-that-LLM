"""exp023: Roman Tamrazov's typography step (exp020) widened from 34 hand-picked markers to every punctuation / layout
pattern, out of fold.

Each passage is reduced to its "shape": letters -> a (A if upper case), digits -> 0, everything else (spaces, newlines,
punctuation, quotes, dashes, symbols) kept. Binary presence of shape n-grams that contain at least one non-letter,
non-digit character (n = 1..4; at least 20 train passages) then enters the same Bernoulli likelihood as his markers
((hits + 2) / (n + 4), clipped to [0.015, 0.985], mean log-likelihood centred per row), fitted on the other folds,
and is added like his step (gate (0.8 - margin) / 0.8, clip 0.35), alone or next to his markers. Scale is reported on a
grid and tuned on the other folds (cross-fitted). Base: exp015's `--plain` replica with no post step (exp020).

    PYTHONPATH=src python experiments/exp023_shape_ngrams.py [path/to/atlas4b/oof_all.parquet]
"""

import sys

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import CountVectorizer

import exp020_typobayes_oof as T  # noqa: E402  (appends --plain and imports exp015 as T.E)
import exp021_typobayes_tuning as U  # noqa: E402
from ntl.data import load_train
from ntl.metrics import balanced_accuracy
from ntl.oofbank import paired_bootstrap

E = T.E
SCALES = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)


def shape(t):
    return "".join("A" if c.isupper() else "a" if c.isalpha() else "0" if c.isdigit() else c for c in t)


def shape_matrix(texts, min_df=20, ngram=(1, 4)):
    """binary shape n-grams with at least one character that is not a letter or a digit"""
    shapes = [shape(t) for t in texts]
    cv = CountVectorizer(analyzer="char", ngram_range=ngram, lowercase=False, min_df=min_df, binary=True)
    X = cv.fit_transform(shapes)
    keep = np.array([any(ch not in "aA0" for ch in g) for g in cv.get_feature_names_out()])
    return X[:, keep].toarray().astype(np.float64), cv.get_feature_names_out()[keep]


def main():
    train = load_train()
    labels = np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts, texts = train.prompt_id.to_numpy(), train.text.to_numpy()
    nlm = np.load(E.NLM_OOF); folds = nlm["folds"]
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(E.OUT / "svc4_oof.npy").astype(np.float64)
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    assert np.array_equal(z["ids"], train.id.to_numpy(dtype=str))
    members["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    score = lambda pred: balanced_accuracy(train.model, labels[pred])
    P0 = E.blend(members, {**E.CORE, "deb": 1.0})
    bases = {"replica": P0}
    if len(sys.argv) > 1 and sys.argv[1].endswith(".parquet"):
        a = pd.read_parquet(sys.argv[1]).set_index("id").loc[train.id.astype(str)]
        LL = a[[f"LL_{l}" for l in E.NLM_LABELS]].to_numpy(np.float64)
        A = np.exp(14.820207057988586 * (LL - LL.max(1, keepdims=True))); A /= A.sum(1, keepdims=True)
        s = np.sort(P0, 1); rows = (s[:, -1] - s[:, -2]) < 0.3
        P = P0.copy(); P[rows] = 0.9 * P0[rows] + 0.1 * A[rows]
        bases["replica + atlas4b 10%"] = P

    # the vocabulary is fitted on all train texts (no labels, so no leak); rates come from the other folds
    X1, names1 = shape_matrix(texts, ngram=(1, 1), min_df=5)
    X4, names4 = shape_matrix(texts)
    print(f"shape unigrams (non-alphanumeric characters): {X1.shape[1]}; shape 1-4-grams: {X4.shape[1]}")
    evs = {"nb34": U.oof_evidence(U.markers(texts, T.MARKS), y, folds),
           "chars": U.oof_evidence(X1, y, folds), "shape1-4": U.oof_evidence(X4, y, folds)}
    for k, ev in evs.items():
        print(f"evidence alone {k:9s} {score(ev.argmax(1)):.4f}")

    for bname, P in bases.items():
        raw = np.log(E.sinkhorn(P) + 1e-12)
        q = softmax(raw, 1); qs = np.sort(q, 1); margin = qs[:, -1] - qs[:, -2]
        gate = np.clip((0.8 - margin) / 0.8, 0, 1)[:, None]
        ref = U.apply(raw, margin, evs["nb34"], *U.DEFAULT)
        print(f"\n== {bname}: no step {score(raw.argmax(1)):.4f}, v38a {score(ref):.4f}")
        cands = {"chars in place of nb34": lambda s: U.apply(raw, margin, evs["chars"], s, 0.35, 0.8),
                 "shape in place of nb34": lambda s: U.apply(raw, margin, evs["shape1-4"], s, 0.35, 0.8),
                 "v38a + chars": lambda s: (raw + gate * (np.clip(2.0 * evs["nb34"], -.35, .35)
                                                          + np.clip(s * evs["chars"], -.35, .35))).argmax(1),
                 "v38a + shape": lambda s: (raw + gate * (np.clip(2.0 * evs["nb34"], -.35, .35)
                                                          + np.clip(s * evs["shape1-4"], -.35, .35))).argmax(1)}
        for cname, fn in cands.items():
            preds = {s: fn(s) for s in SCALES}
            cf = np.empty_like(ref); picks = []
            for f in np.unique(folds):
                tr = folds != f
                pick = max(SCALES, key=lambda s: balanced_accuracy(y[tr], preds[s][tr]))
                cf[~tr] = preds[pick][~tr]; picks.append(pick)
            c = paired_bootstrap(y, cf, ref, prompts)
            print(f"  {cname:24s} " + " ".join(f"s{s:g} {score(p):.4f}" for s, p in preds.items())
                  + f" | cross-fitted {score(cf):.4f} vs v38a {c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}] picks {picks}")


if __name__ == "__main__":
    main()
