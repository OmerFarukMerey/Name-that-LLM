"""exp049: ozertuu's 0.672 (his notebook's version 1) and his 0.670 load a private pre-fit SVM that his later public
notebooks describe in a code comment: "Super-SVC V5 (trained on 39.8k clean balanced pool + Claude extra, CV 0.3630)",
at weight 6.0 beside the in-notebook open-pool SVM at 4.0. Without it (his "decoy fallback": Nuri Dudak's v4n, a refit
of the in-notebook SVM, at 1.2 beside 10.0) the same code scores 0.668. This fits that SVM recipe (v4n's four views,
C 0.5, balanced class weights) on pool rows only, the same number per label (39.8k / 18 = 2,211), drawn at random or by
typography match_weight (exp034's draw), and scores it on every official row (out of sample: the pool holds no
train-prompt rows) alone and in K's replica (exp047's: TF-IDF, distort and the in-notebook SVM out of fold on K's rows,
the stylegram stand-in at 15%, Sinkhorn 0.30) at the weights his notebooks use. Control arm p39798: the same number of
pool rows drawn at random from the whole pool (the pool's label mix, 3.5k-22k per label).

    python -I experiments/exp049_balanced_pool_svc.py fit <ntl_open_pool.parquet> [arm ...]   # one arm at a time
    python -I experiments/exp049_balanced_pool_svc.py                                         # compare

-I because the pool is a download.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
if "--plain" not in sys.argv:
    sys.argv.append("--plain")
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp015_nlm_core_oof as E  # noqa: E402  (reads --plain at import)
import exp034_matched_extra as X  # noqa: E402
import exp035_comp_extra as CX  # noqa: E402
import exp036_svc_matched as M  # noqa: E402
from exp029_atlas_ext import soft_sinkhorn  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402
from ntl.paths import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "outputs" / "exp049_balanced_pool_svc"
ARMS = {"r2211": ("random", 2211), "m2211": ("match_weight", 2211), "m3541": ("match_weight", 3541),
        "p39798": ("pool mix", 39798), "t2211": ("top match_weight", 2211),
        "m2211c": ("match_weight, Claude labels 2x", 2211)}   # "+ Claude extra": the five Claude labels draw twice as many
VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5),
         ("function", 1, 3, 50000, 0.5)]   # v4n's metadata (shape / small there) = infer.py's svc4


def draw(pool, arm):
    how, n = ARMS[arm]
    if how == "pool mix":
        return pool.iloc[np.sort(np.random.RandomState(0).choice(len(pool), n, replace=False))]
    if how == "random":
        return pd.concat([g.iloc[np.sort(np.random.RandomState(0).choice(len(g), min(n, len(g)), replace=False))]
                          for _, g in pool.groupby("model")])
    if how == "top match_weight":
        return pd.concat([g.sort_values("match_weight", ascending=False, kind="stable").head(n) for _, g in pool.groupby("model")])
    claude = lambda m: 2 if how.endswith("2x") and m.startswith("claude-") else 1
    return pd.concat([X.weighted_draw(g, min(claude(m) * n, len(g))) for m, g in pool.groupby("model")])


def fit(pool_path, arms):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_train()
    official = train.text.fillna("").astype(str).to_numpy()
    pool = pd.read_parquet(pool_path, columns=["text", "model", "flag_train_prompt_regen", "match_weight"])
    pool = pool[~pool.flag_train_prompt_regen.astype(bool) & pool.model.isin(E.NLM_LABELS)].reset_index(drop=True)
    pool["text"] = pool.text.fillna("").astype(str)
    seen0 = {X.key(t) for t in official}
    for arm in arms:
        t0 = time.time()
        rows = draw(pool, arm)
        seen, texts, ys = set(seen0), [], []
        for t, m in zip(rows.text.values, rows.model.values):   # infer.py's exact de-duplication
            k = X.key(t)
            if k and k not in seen:
                seen.add(k); texts.append(t); ys.append(m)
        A, B = [], []
        for view, lo, hi, mf, wv in VIEWS:
            prep = M.shape if view == "shape" else M.func if view == "function" else M.ident
            an = "char" if view == "shape" else "word" if view == "function" else view
            o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf,
                     dtype=np.float32)
            if an == "word":
                o["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
            v = TfidfVectorizer(**o)
            A.append(v.fit_transform([prep(t) for t in texts]) * wv)
            B.append(v.transform([prep(t) for t in official]) * wv)
        svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").fit(hstack(A, format="csr"), np.array(ys))
        d = svc.decision_function(hstack(B, format="csr"))
        d = d[:, [list(svc.classes_).index(l) for l in E.NLM_LABELS]]
        np.save(OUT / f"{arm}_bsvc.npy", E.lsm(d).astype(np.float32))
        counts = pd.Series(ys).value_counts()
        print(f"{arm}: {len(texts)} fit rows ({counts.min()}-{counts.max()} per label), {time.time() - t0:.0f}s", flush=True)


def compare():
    train = load_train()
    nlm = np.load(E.NLM_OOF)
    labels = np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts, folds = train.prompt_id.to_numpy(), nlm["folds"]
    mem = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    mem["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    mem["distort"] = np.load(X.OUT / "m2_distort.npy")
    mem["tfidf"] = np.load(X.OUT / "m2_tfidf.npy")
    mem["svc4"] = np.load(M.OUT / "m2_svc4.npy").astype(np.float64)
    mem["v4n"] = mem["svc4"]   # v4n is the in-notebook SVM refit on all official rows: its out-of-fold stand-in
    E.TERMS.update({"tfidf": 1, "distort": 1, "v4n": 1, "bsvc": 1})
    sty = np.load(CX.OUT / "ntl-v31ac-a20-sk30-osvc12-xm2-cx-p30_stylegram.npy")
    C = np.exp(3.0 * E.lsm(sty)); C /= C.sum(1, keepdims=True)
    score = lambda S: balanced_accuracy(train.model, labels[S.argmax(1)])
    per_fold = lambda P: " ".join(f"{balanced_accuracy(train.model[folds == k], labels[P[folds == k].argmax(1)]):.3f}"
                                  for k in np.unique(folds))
    count = lambda S: f"3.7 {int((S.argmax(1) == 7).sum())} / 3.8 {int((S.argmax(1) == 8).sum())}"

    def run(w, rho=0.35):
        P = E.blend(mem, {**E.CORE, "deb": 1.0, "tfidf": 0.25, "distort": 0.10, **w})
        P = 0.85 * P + 0.15 * C; P /= P.sum(1, keepdims=True)
        return soft_sinkhorn(P, alpha=0.30, rho=rho)

    print(f"in-notebook SVM alone (K's rows, out of fold): {score(mem['svc4']):.4f}  {count(mem['svc4'])}")
    arms = [a for a in ARMS if (OUT / f"{a}_bsvc.npy").exists()]
    for a in arms:
        s = np.load(OUT / f"{a}_bsvc.npy").astype(np.float64)
        rec = pd.Series(s.argmax(1) == y).groupby(y).mean()
        print(f"{a} ({ARMS[a][0]}) alone: {score(s):.4f}  {count(s)}  recall 3.7 {rec[7]:.2f} 3.8 {rec[8]:.2f} kimi {rec[16]:.2f}")
    ref_w = {"svc4": 10.0, "v4n": 1.2}
    ref = run(ref_w)
    print(f"\nreplica, version 4's SVMs (in-notebook 10 + v4n 1.2): {score(ref):.4f}  {count(ref)}  folds {per_fold(ref)}")
    k12 = run({"svc4": 12.0})
    g = paired_bootstrap(y, k12.argmax(1), ref.argmax(1), prompts)
    print(f"  our 0.668's SVM (in-notebook 12): {score(k12):.4f} ({g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}])")
    for rho in (0.5, 0.75):
        P = run(ref_w, rho)
        g = paired_bootstrap(y, P.argmax(1), ref.argmax(1), prompts)
        print(f"  target mix {rho}: {score(P):.4f} ({g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}])")
    for a in arms:
        mem["bsvc"] = np.load(OUT / f"{a}_bsvc.npy").astype(np.float64)
        print(f"{a} in place of v4n:")
        for sw, bw in ((10.0, 1.2), (7.0, 3.0), (6.0, 4.0), (4.0, 6.0), (10.0, 6.0)):
            P = run({"svc4": sw, "bsvc": bw})
            g = paired_bootstrap(y, P.argmax(1), ref.argmax(1), prompts)
            print(f"  in-notebook {sw:g} + balanced {bw:g}: {score(P):.4f} ({g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}])"
                  f"  {count(P)}  folds {per_fold(P)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["fit"]:
        fit(sys.argv[2], [a for a in sys.argv[3:] if a in ARMS] or list(ARMS))
    else:
        compare()
