"""exp036: exp034 refitted the TF-IDF and distort members with K's matched rows, but the third member that learns from
`--extra`, the 4-view SVC (weight 12), still came from exp030 (open pool at 0.35, no external rows). infer.py's SVC fits
the official rows (1.0), the external rows (`--extra_w`: 0.5, or 1.0 for exp034's m2w1) and the pool (0.35), each
exactly de-duplicated against the rows before it. So K's change moves the SVC from official + nlm_matched_clean (24k)
+ open pool (190k) to official + the matched draw (49k, from the pool, so those passages go from 0.35 to 0.5) + the
rest of the pool. This fits that SVC out of fold on exp015's folds for nlm_matched_clean (their C), the 2x draw (K) and
4x, scores it alone, and scores the full replica (core + deb, Sinkhorn 0.30, SVC 12) with all three n-gram members
from the same external rows (exp034's TF-IDF and distort OOF) at TF-IDF 0.25-1.5.

    python -I experiments/exp036_svc_matched.py fit <ntl_open_pool.parquet> <nlm_matched_clean.parquet> [mc m2 m4 m2w1 ...]
    python -I experiments/exp036_svc_matched.py                  # compare

One fold of this SVC holds ~200k rows in four TF-IDF views (several GB), so it runs EXP036_WORKERS (default 1) at a time.
-I because both parquet files are downloads; the repo's src/ and experiments/ are put on the path below.
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
import re  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp015_nlm_core_oof as E  # noqa: E402  (reads --plain at import)
import exp034_matched_extra as X  # noqa: E402
from exp029_atlas_ext import soft_sinkhorn  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402
from ntl.paths import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "outputs" / "exp036_svc_matched"
FW = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())


def shape(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def func(t):
    return " ".join(w.lower() if w.lower() in FW or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))


def ident(t):
    return t


def fit_fold(args):
    """infer.py's svc4 block on one fold: official rows of the other folds (1.0), external rows (0.5), pool (0.35), each
    exactly de-duplicated against everything before it; decision values for the held-out rows."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    tr_t, tr_y, ex_t, ex_y, extra_w, pool_t, pool_y, va_t = args
    t0 = time.time()
    seen, texts, ys, sw = {X.key(t) for t in tr_t}, list(tr_t), list(tr_y), [1.0] * len(tr_t)
    for src_t, src_y, w in ((ex_t, ex_y, extra_w), (pool_t, pool_y, 0.35)):
        for t, l in zip(src_t, src_y):
            k = X.key(t)
            if k and k not in seen:
                seen.add(k); texts.append(t); ys.append(l); sw.append(w)
    A, B = [], []
    for view, lo, hi, mf, wv in [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5),
                                 ("function", 1, 3, 50000, 0.5)]:
        prep = shape if view == "shape" else func if view == "function" else ident
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


def fit(pool_path, mc_path, names, workers):
    (OUT / "folds").mkdir(parents=True, exist_ok=True)
    train = load_train()
    folds = np.load(E.NLM_OOF)["folds"]
    l2i = {l: i for i, l in enumerate(E.NLM_LABELS)}
    texts, y = train.text.fillna("").astype(str).to_numpy(), train.model.map(l2i).to_numpy()
    pool = pd.read_parquet(pool_path, columns=["text", "model", "flag_train_prompt_regen"])
    pool = pool[~pool.flag_train_prompt_regen.astype(bool) & pool.model.isin(E.NLM_LABELS)]
    pool_t, pool_y = pool.text.fillna("").astype(str).to_numpy(), pool.model.map(l2i).to_numpy()
    part = lambda v, k: OUT / "folds" / f"{v}_svc4_{k}.npy"
    jobs, slots = [], []
    for v in names:
        ex = X.extra_rows(v, pool_path, mc_path, texts)   # exp034's draw and its de-duplication against the official rows
        ex_t, ex_y = ex.text.to_numpy(), ex.model.map(l2i).to_numpy()
        for k in np.unique(folds):
            if not part(v, k).exists():
                jobs.append((texts[folds != k], y[folds != k], ex_t, ex_y, X.VARIANTS[v][1], pool_t, pool_y, texts[folds == k]))
                slots.append((v, k))
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs) or 1))) as ex_pool:
        for (v, k), (lp, n, secs) in zip(slots, ex_pool.map(fit_fold, jobs)):
            np.save(part(v, k), lp.astype(np.float32))
            print(f"{v} svc4 fold {k}: {n} fit rows, {secs:.0f}s", flush=True)
    for v in names:
        lp = np.zeros((len(train), 18), np.float32)
        for k in np.unique(folds):
            lp[folds == k] = np.load(part(v, k))
        np.save(OUT / f"{v}_svc4.npy", lp)
        print(f"{v} svc4 alone: {balanced_accuracy(train.model, np.array(E.NLM_LABELS)[lp.argmax(1)]):.4f}", flush=True)


def compare():
    train = load_train()
    nlm = np.load(E.NLM_OOF)
    labels = np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    members["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    E.TERMS.update({"tfidf": 1, "distort": 1})
    score = lambda P: balanced_accuracy(train.model, labels[P.argmax(1)])
    have = [v for v in X.VARIANTS if (OUT / f"{v}_svc4.npy").exists()]
    svc = {v: np.load(OUT / f"{v}_svc4.npy").astype(np.float64) for v in have}
    svc["exp030"] = np.load(REPO_ROOT / "outputs" / "exp030_open_pool_svc" / "svc4_open_oof.npy").astype(np.float64)
    for v, s in svc.items():
        print(f"svc4 alone, {v}: {score(s):.4f}")

    def run(v, s, w):
        lp = {m: np.load(X.OUT / f"{v}_{m}.npy") for m in ("tfidf", "distort")}
        ws = {**E.CORE, "svc4": 12.0, "deb": 1.0, "tfidf": w, "distort": 0.10}
        return soft_sinkhorn(E.blend({**members, "svc4": svc[s], **lp}, ws), alpha=0.30)

    folds = nlm["folds"]
    per_fold = lambda P: " ".join(f"{balanced_accuracy(train.model[folds == k], labels[P[folds == k].argmax(1)]):.3f}"
                                  for k in np.unique(folds))
    ref = run("mc", "mc" if "mc" in svc else "exp030", 0.25)
    print(f"\nreplica, all three n-gram members on nlm_matched_clean (their C): {score(ref):.4f}  folds {per_fold(ref)}")
    for v in have:
        for w in (0.25, 0.5, 0.75, 1.0, 1.5):
            P = run(v, v, w)
            g = paired_bootstrap(y, P.argmax(1), ref.argmax(1), prompts)
            print(f"  {v} (TF-IDF, distort and SVC), TF-IDF {w}: {score(P):.4f} (gain {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}])"
                  f"  folds {per_fold(P)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["fit"]:
        fit(sys.argv[2], sys.argv[3], [a for a in sys.argv[4:] if a in X.VARIANTS] or ["mc", "m2", "m4"],
            int(os.environ.get("EXP036_WORKERS", "1")))
    else:
        compare()
