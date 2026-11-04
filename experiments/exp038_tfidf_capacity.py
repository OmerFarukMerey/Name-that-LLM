"""exp038: K's matched rows make infer.py's TF-IDF member much stronger alone (exp034: 0.418 -> 0.464), and weighing it
0.5 instead of 0.25 is the steadiest gain on K's replica (exp036: +0.0045). Its vectorisers are small for ~51k fit rows
(char_wb 2-5 capped at 60k features, word 1-2 at 30k; C=20). This refits it out of fold on exp015's folds with K's rows
(2x, extra_w 0.5) at larger vocabularies, scores it alone and in exp036's K replica (core + deb, SVC 12 and distort on
K's rows, Sinkhorn 0.30) at TF-IDF 0.25 / 0.5 / 0.75 / 1.0 against infer.py's member (exp034's m2_tfidf).

    python -I experiments/exp038_tfidf_capacity.py fit <ntl_open_pool.parquet> <nlm_matched_clean.parquet> [config ...]
    python -I experiments/exp038_tfidf_capacity.py                      # compare

-I because both parquet files are downloads; the repo's src/ and experiments/ are put on the path below.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
if "--plain" not in sys.argv:
    sys.argv.append("--plain")
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import exp015_nlm_core_oof as E  # noqa: E402  (reads --plain at import)
import exp034_matched_extra as X  # noqa: E402
import exp036_svc_matched as M  # noqa: E402
from exp029_atlas_ext import soft_sinkhorn  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402
from ntl.paths import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "outputs" / "exp038_tfidf_capacity"
WORD = r"\w+|[^\w\s]"
# name: ([(analyzer, ngram_range, max_features)], C); infer.py's member is char_wb (2, 5) 60k + word (1, 2) 30k, C 20
CONFIGS = {"big": ([("char_wb", (2, 5), 200_000), ("word", (1, 2), 100_000)], 20),
           "big3": ([("char_wb", (1, 6), 300_000), ("word", (1, 3), 150_000)], 20)}


def fit_fold(args):
    """infer.py's TF-IDF member (min_df 3, sublinear tf, no lowercasing, LR) with other vocabularies."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    name, tr_t, tr_y, ex_t, ex_y, va_t = args
    views, C = CONFIGS[name]
    t0 = time.time()
    fit_t, fit_y = np.concatenate([tr_t, ex_t]), np.concatenate([tr_y, ex_y])
    fit_w = np.concatenate([np.ones(len(tr_t), dtype=np.float32), np.full(len(ex_t), 0.5, dtype=np.float32)])
    mats_tr, mats_te = [], []
    for an, ngr, mf in views:
        kw = dict(analyzer="char_wb") if an == "char_wb" else dict(token_pattern=WORD)
        vec = TfidfVectorizer(sublinear_tf=True, lowercase=False, ngram_range=ngr, min_df=3, max_features=mf, **kw).fit(fit_t)
        mats_tr.append(vec.transform(fit_t)); mats_te.append(vec.transform(va_t))
    lp = LogisticRegression(C=C, max_iter=3000).fit(sp.hstack(mats_tr).tocsr(), fit_y, sample_weight=fit_w) \
        .predict_log_proba(sp.hstack(mats_te).tocsr())
    assert lp.shape[1] == 18
    return lp, time.time() - t0


def fit(pool_path, mc_path, names, workers):
    (OUT / "folds").mkdir(parents=True, exist_ok=True)
    train = load_train()
    folds = np.load(E.NLM_OOF)["folds"]
    l2i = {l: i for i, l in enumerate(E.NLM_LABELS)}
    texts, y = train.text.fillna("").astype(str).to_numpy(), train.model.map(l2i).to_numpy()
    ex = X.extra_rows("m2", pool_path, mc_path, texts)
    ex_t, ex_y = ex.text.to_numpy(), ex.model.map(l2i).to_numpy()
    part = lambda n, k: OUT / "folds" / f"{n}_{k}.npy"
    jobs, slots = [], []
    for n in names:
        for k in np.unique(folds):
            if not part(n, k).exists():
                jobs.append((n, texts[folds != k], y[folds != k], ex_t, ex_y, texts[folds == k]))
                slots.append((n, k))
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs) or 1))) as pool:
        for (n, k), (lp, secs) in zip(slots, pool.map(fit_fold, jobs)):
            np.save(part(n, k), lp)
            print(f"{n}: fold {k} done in {secs:.0f}s", flush=True)
    for n in names:
        lp = np.zeros((len(train), 18))
        for k in np.unique(folds):
            lp[folds == k] = np.load(part(n, k))
        np.save(OUT / f"{n}.npy", lp)
        print(f"{n} alone: {balanced_accuracy(train.model, np.array(E.NLM_LABELS)[lp.argmax(1)]):.4f}", flush=True)


def compare():
    train = load_train()
    nlm = np.load(E.NLM_OOF)
    folds, labels = nlm["folds"], np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    members["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    members["svc4"] = np.load(M.OUT / "m2_svc4.npy").astype(np.float64)
    members["distort"] = np.load(X.OUT / "m2_distort.npy")
    E.TERMS.update({"tfidf": 1, "distort": 1})
    score = lambda Z: balanced_accuracy(train.model, labels[Z.argmax(1)])
    per_fold = lambda Z: " ".join(f"{balanced_accuracy(train.model[folds == k], labels[Z[folds == k].argmax(1)]):.3f}"
                                  for k in np.unique(folds))
    tf = {"infer.py": np.load(X.OUT / "m2_tfidf.npy")}
    tf.update({n: np.load(OUT / f"{n}.npy") for n in CONFIGS if (OUT / f"{n}.npy").exists()})
    run = lambda t, w: soft_sinkhorn(E.blend({**members, "tfidf": tf[t]}, {**E.CORE, "svc4": 12.0, "deb": 1.0, "tfidf": w,
                                                                          "distort": 0.10}), alpha=0.30)
    ref = run("infer.py", 0.25)
    print(f"K replica (infer.py's TF-IDF member at 0.25): {score(ref):.4f}  folds {per_fold(ref)}")
    for t, lp in tf.items():
        print(f"\n{t} alone: {score(lp):.4f}")
        for w in (0.25, 0.5, 0.75, 1.0):
            P = run(t, w)
            g = paired_bootstrap(y, P.argmax(1), ref.argmax(1), prompts)
            print(f"  TF-IDF {w}: {score(P):.4f} (vs K {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}])  folds {per_fold(P)}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["fit"]:
        fit(sys.argv[2], sys.argv[3], [a for a in sys.argv[4:] if a in CONFIGS] or list(CONFIGS),
            int(os.environ.get("EXP038_WORKERS", "2")))
    else:
        compare()
