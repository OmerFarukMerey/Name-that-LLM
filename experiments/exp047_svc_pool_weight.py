"""exp047: infer.py's 4-view SVC (weight 12) reads the open pool's ~190k rows at a flat source weight 0.35. exp046 found
that weighting pool rows by match_weight (per label mean 1, clip 4) lifts a pool-only TF-IDF model on the official rows
by +0.017. This refits the SVC out of fold on exp015's folds with K's external rows (m2) and the pool rows at
0.35 x that weight instead, then scores it alone and in K's replica (TF-IDF and distort from m2, the stylegram stand-in
x3 at 15%, Sinkhorn 0.30) against exp036's m2 SVC.

    python -I experiments/exp047_svc_pool_weight.py fit <ntl_open_pool.parquet> <nlm_matched_clean.parquet>
    python -I experiments/exp047_svc_pool_weight.py                    # compare

EXP047_WORKERS (default 1) folds at a time: one fold on the whole pool peaks above ~10 GB, which on a 24 GB Mac
starves everything else. EXP047_POOL_N (default 0 = all) fits both arms on that many pool rows (fixed sample, seed 0):
"u<N>" (flat 0.35) and "pw<N>" (0.35 x match_weight), compared with each other. -I because both parquet files are
downloads.
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
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp015_nlm_core_oof as E  # noqa: E402  (reads --plain at import)
import exp034_matched_extra as X  # noqa: E402
import exp035_comp_extra as CX  # noqa: E402
import exp036_svc_matched as M  # noqa: E402
from exp029_atlas_ext import soft_sinkhorn  # noqa: E402
from exp046_pool_weighting import row_weights  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402
from ntl.paths import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "outputs" / "exp047_svc_pool_weight"
POWER, CLIP = 1.0, 4.0


def fit_fold(args):
    """exp036's fit_fold with per-row pool weights (0.35 x w) in place of the flat 0.35."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    tr_t, tr_y, ex_t, ex_y, extra_w, pool_t, pool_y, pool_w, va_t = args
    t0 = time.time()
    seen, texts, ys, sw = {X.key(t) for t in tr_t}, list(tr_t), list(tr_y), [1.0] * len(tr_t)
    for t, l in zip(ex_t, ex_y):
        k = X.key(t)
        if k and k not in seen:
            seen.add(k); texts.append(t); ys.append(l); sw.append(extra_w)
    for t, l, w in zip(pool_t, pool_y, pool_w):
        k = X.key(t)
        if k and k not in seen:
            seen.add(k); texts.append(t); ys.append(l); sw.append(0.35 * float(w))
    A, B = [], []
    for view, lo, hi, mf, wv in [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5),
                                 ("function", 1, 3, 50000, 0.5)]:
        prep = M.shape if view == "shape" else M.func if view == "function" else M.ident
        an = "char" if view == "shape" else "word" if view == "function" else view
        o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
        if an == "word":
            o["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**o)
        A.append(v.fit_transform([prep(t) for t in texts]) * wv)
        B.append(v.transform([prep(t) for t in va_t]) * wv)
    svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")
    d = svc.fit(hstack(A, format="csr"), np.array(ys), sample_weight=np.array(sw, np.float32)).decision_function(hstack(B, format="csr"))
    return E.lsm(d), len(texts), time.time() - t0


def fit(pool_path, mc_path, workers, pool_n=0):
    (OUT / "folds").mkdir(parents=True, exist_ok=True)
    train = load_train()
    folds = np.load(E.NLM_OOF)["folds"]
    l2i = {l: i for i, l in enumerate(E.NLM_LABELS)}
    texts, y = train.text.fillna("").astype(str).to_numpy(), train.model.map(l2i).to_numpy()
    pool = pd.read_parquet(pool_path, columns=["text", "model", "flag_train_prompt_regen", "match_weight"])
    pool = pool[~pool.flag_train_prompt_regen.astype(bool) & pool.model.isin(E.NLM_LABELS)].reset_index(drop=True)
    pool_w = row_weights(pool, POWER, CLIP)   # weights from the whole pool (as the kernel would), before any sample
    if pool_n:
        keep = np.sort(np.random.RandomState(0).choice(len(pool), pool_n, replace=False))
        pool, pool_w = pool.iloc[keep].reset_index(drop=True), pool_w[keep]
    print(f"pool {len(pool)} rows; source weight 0.35 x match_weight^{POWER} (clip {CLIP}): "
          f"min {0.35 * pool_w.min():.3f} max {0.35 * pool_w.max():.3f}", flush=True)
    pool_t, pool_y = pool.text.fillna("").astype(str).to_numpy(), pool.model.map(l2i).to_numpy()
    ex = X.extra_rows("m2", pool_path, mc_path, texts)
    ex_t, ex_y = ex.text.to_numpy(), ex.model.map(l2i).to_numpy()
    arms = {f"u{pool_n}": np.ones(len(pool)), f"pw{pool_n}": pool_w} if pool_n else {"m2pw": pool_w}
    part = lambda a, k: OUT / "folds" / f"{a}_svc4_{k}.npy"
    jobs, slots = [], []
    for a, w in arms.items():
        for k in np.unique(folds):
            if not part(a, k).exists():
                jobs.append((texts[folds != k], y[folds != k], ex_t, ex_y, X.VARIANTS["m2"][1], pool_t, pool_y, w, texts[folds == k]))
                slots.append((a, k))
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs) or 1))) as pe:
        for (a, k), (lp, n, secs) in zip(slots, pe.map(fit_fold, jobs)):
            np.save(part(a, k), lp.astype(np.float32))
            print(f"{a} svc4 fold {k}: {n} fit rows, {secs:.0f}s", flush=True)
    for a in arms:
        lp = np.zeros((len(train), 18), np.float32)
        for k in np.unique(folds):
            lp[folds == k] = np.load(part(a, k))
        np.save(OUT / f"{a}_svc4.npy", lp)


def compare(pool_n=0):
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
    E.TERMS.update({"tfidf": 1, "distort": 1})
    sty = np.load(CX.OUT / "ntl-v31ac-a20-sk30-osvc12-xm2-cx-p30_stylegram.npy")
    C = np.exp(3.0 * E.lsm(sty)); C /= C.sum(1, keepdims=True)
    score = lambda S: balanced_accuracy(train.model, labels[S.argmax(1)])
    flat, wtd = ("m2 (flat 0.35)", "m2pw (0.35 x match_weight)") if not pool_n else \
        (f"u{pool_n} (flat 0.35, {pool_n} pool rows)", f"pw{pool_n} (0.35 x match_weight, {pool_n} pool rows)")
    svc = {flat: np.load((M.OUT / "m2_svc4.npy") if not pool_n else (OUT / f"u{pool_n}_svc4.npy")).astype(np.float64),
           wtd: np.load(OUT / ("m2pw_svc4.npy" if not pool_n else f"pw{pool_n}_svc4.npy")).astype(np.float64)}
    for k, s in svc.items():
        rec = pd.Series(s.argmax(1) == y).groupby(y).mean()
        print(f"svc4 alone, {k}: {score(s):.4f}  predicted 3.7 {int((s.argmax(1) == 7).sum())} / 3.8 {int((s.argmax(1) == 8).sum())}"
              f"  recall 3.7 {rec[7]:.2f} kimi {rec[16]:.2f}")

    def run(s, svc_w=12.0, stack=True):
        P = E.blend({**mem, "svc4": s}, {**E.CORE, "svc4": svc_w, "deb": 1.0, "tfidf": 0.25, "distort": 0.10})
        if stack:
            P = 0.85 * P + 0.15 * C; P /= P.sum(1, keepdims=True)
        return soft_sinkhorn(P, alpha=0.30)

    per_fold = lambda P: " ".join(f"{balanced_accuracy(train.model[folds == k], labels[P[folds == k].argmax(1)]):.3f}"
                                  for k in np.unique(folds))
    for stack in (True, False):
        ref = run(svc[flat], stack=stack)
        print(f"\nreplica {'with' if stack else 'without'} the stylegram stand-in, {flat}: {score(ref):.4f}  folds {per_fold(ref)}")
        for w in (6.0, 12.0, 20.0):
            P = run(svc[wtd], w, stack)
            g = paired_bootstrap(y, P.argmax(1), ref.argmax(1), prompts)
            print(f"  pool-weighted SVC at {w}: {score(P):.4f} ({g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}])  folds {per_fold(P)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["fit"]:
        fit(sys.argv[2], sys.argv[3], int(os.environ.get("EXP047_WORKERS", "1")), int(os.environ.get("EXP047_POOL_N", "0")))
    else:
        compare(int(os.environ.get("EXP047_POOL_N", "0")))
