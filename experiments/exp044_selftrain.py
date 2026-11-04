"""exp044: test-time self-training. Nothing public uses the hidden test's own passages beyond clustering, balancing and
the twin/Gaussian heads. Here the replica's final predictions (K's replica, exp041's stylegram stand-in at x3, Sinkhorn
0.30) label the "test", and an n-gram LR fitted on those pseudo-labels alone (TF-IDF vocabulary from the test itself)
is mixed back in. Every train row is out of fold in the replica, so all 2,699 act as one test set. The LR is
cross-fitted: a row is never scored by a model that saw it or its group, grouped by the true prompt (oracle) or by
infer.py's own test clusters (capped_agglo, cap 18, threshold 0.15), which is all the hidden run can use.

    python -I experiments/exp044_selftrain.py

-I for consistency with the other replica scripts; the repo's src/ and experiments/ are put on the path below.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
if "--plain" not in sys.argv:
    sys.argv.append("--plain")
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import exp015_nlm_core_oof as E  # noqa: E402  (reads --plain at import)
import exp034_matched_extra as X  # noqa: E402
import exp035_comp_extra as CX  # noqa: E402
import exp036_svc_matched as M  # noqa: E402
from exp029_atlas_ext import soft_sinkhorn  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402


def capped_agglo(S, cap=18, thr=0.02):
    """infer.py's test clustering (average linkage, never past `cap` members, stop below `thr`)."""
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        Mx = np.where(ok, A, -np.inf); k = int(np.argmax(Mx)); i, j = divmod(k, n)
        if not np.isfinite(Mx[i, j]) or Mx[i, j] < thr:
            break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return lab


def replica(train):
    nlm = np.load(E.NLM_OOF)
    mem = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    mem["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    mem["svc4"] = np.load(M.OUT / "m2_svc4.npy").astype(np.float64)
    mem["distort"] = np.load(X.OUT / "m2_distort.npy")
    mem["tfidf"] = np.load(X.OUT / "m2_tfidf.npy")
    E.TERMS.update({"tfidf": 1, "distort": 1})
    P = E.blend(mem, {**E.CORE, "svc4": 12.0, "deb": 1.0, "tfidf": 0.25, "distort": 0.10})
    sty = np.load(CX.OUT / "ntl-v31ac-a20-sk30-osvc12-xm2-cx-p30_stylegram.npy")
    C = np.exp(3.0 * E.lsm(sty)); C /= C.sum(1, keepdims=True)
    P = 0.85 * P + 0.15 * C; P /= P.sum(1, keepdims=True)
    return P, nlm["folds"]


def features(texts):
    """infer.py's TF-IDF member views, vocabulary and idf from the test passages themselves."""
    mats = [TfidfVectorizer(sublinear_tf=True, lowercase=False, **kw).fit_transform(texts)
            for kw in [dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=60000),
                       dict(token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), min_df=3, max_features=30000)]]
    return sp.hstack(mats).tocsr()


def self_train(Xf, S, groups, nfold=5, C=10.0, keep=1.0, weighted=False):
    """Cross-fitted LR on pseudo-labels: groups are split over nfold folds; each fold is scored by an LR fitted on the
    other folds' rows (the top `keep` fraction by confidence, per predicted class)."""
    Q = np.exp(S - S.max(1, keepdims=True)); Q /= Q.sum(1, keepdims=True)
    pl, conf = Q.argmax(1), Q.max(1)
    sel = np.zeros(len(S), bool)
    for c in range(18):
        i = np.where(pl == c)[0]
        if len(i):
            sel[i[np.argsort(-conf[i])[:max(1, int(round(keep * len(i))))]]] = True
    ug = np.unique(groups); fold_of = {g: k % nfold for k, g in enumerate(np.random.RandomState(0).permutation(ug))}
    f = np.array([fold_of[g] for g in groups])
    out = np.zeros_like(S)
    for k in range(nfold):
        tr = (f != k) & sel
        lr = LogisticRegression(C=C, max_iter=3000).fit(Xf[tr], pl[tr], sample_weight=conf[tr] if weighted else None)
        lp = np.full((int((f == k).sum()), 18), np.log(1e-6))
        lp[:, lr.classes_] = lr.predict_log_proba(Xf[f == k])
        out[f == k] = lp
    return out


def main():
    train = load_train()
    labels = np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    P, folds = replica(train)
    S0 = soft_sinkhorn(P, alpha=0.30)
    score = lambda Z: balanced_accuracy(train.model, labels[Z.argmax(1)])
    print(f"replica + stylegram stand-in, Sinkhorn 0.30: {score(S0):.4f}", flush=True)
    texts = train.text.astype(str).tolist()
    Xf = features(texts)
    v = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.05)
    Xc = v.fit_transform(texts)
    clus = capped_agglo((Xc @ Xc.T).toarray(), 18, 0.15)
    from sklearn.metrics import adjusted_rand_score
    print(f"clusters {len(np.unique(clus))}, ARI vs prompts {adjusted_rand_score(prompts, clus):.3f}", flush=True)
    for gname, groups in (("prompt", prompts), ("cluster", clus)):
        for keep in (1.0, 0.6):
            for C in (3.0, 10.0, 30.0):
                ST = self_train(Xf, S0, groups, C=C, keep=keep)
                cells = [f"ST alone {score(ST):.4f}"]
                for w in (0.1, 0.25, 0.5, 1.0):
                    S1 = S0 + w * ST
                    d = paired_bootstrap(y, S1.argmax(1), S0.argmax(1), prompts)
                    cells.append(f"w{w}: {score(S1):.4f} ({d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}])")
                    # before Sinkhorn: P x exp(w ST), then the same balancing
                    P2 = P * np.exp(w * (ST - ST.max(1, keepdims=True))); P2 /= P2.sum(1, keepdims=True)
                    S2 = soft_sinkhorn(P2, alpha=0.30)
                    cells.append(f"pre {score(S2):.4f}")
                print(f"{gname:7s} keep {keep} C {C:4.0f} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
