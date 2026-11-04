"""exp034: areebbb's public 0.666 (version K of their ladder on our 0.663 pipeline) changes the external rows behind
infer.py's TF-IDF, distort and SVC members (--extra, weight --extra_w 0.5): instead of dhruvtibarewal's
nlm_matched_clean (24k), twice as many rows per label from crtproduction's Open Pool release v11 (Kaggle version 13),
drawn with the pool's typography `match_weight`. Their test: the TF-IDF member alone, prompt-grouped 5-fold CV on train,
nlm_matched_clean 0.413, 1x 0.429, 2x 0.447. This fits infer.py's TF-IDF and distort members out of fold on exp015's
folds for nlm_matched_clean and the matched draws at 1x-4x (and 2x at --extra_w 1.0), scores them alone, and inside
exp015's --plain replica (core + DeBERTa stand-in, exp030's open-pool SVC at weight 12, V31's soft Sinkhorn 0.30) at
infer.py's weights (TF-IDF 0.25, distort 0.10). The SVC's own share of the external rows is not refitted here.

    python -I experiments/exp034_matched_extra.py fit <ntl_open_pool.parquet> <nlm_matched_clean.parquet> [variant ...]
    python -I experiments/exp034_matched_extra.py                      # compare

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
import collections  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp015_nlm_core_oof as E  # noqa: E402  (reads --plain at import)
from exp029_atlas_ext import soft_sinkhorn  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402
from ntl.paths import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "outputs" / "exp034_matched_extra"
# name: (rows per label as a multiple of nlm_matched_clean's, extra_w); scale None = nlm_matched_clean itself
VARIANTS = {"mc": (None, 0.5), "m1": (1, 0.5), "m2": (2, 0.5), "m3": (3, 0.5), "m4": (4, 0.5), "m2w1": (2, 1.0), "m2k": (2, 0.5)}
# variants that append another public corpus to the draw, read from the env var NTL_ADD_<NAME>: kilupy's 18-model
# prompt-paired corpus (2026-11-02, CC BY 4.0; 16,603 Claude passages from the Anthropic API + 300 per other model)
ADD = {"m2k": "kilupy"}
MEMBERS = {"tfidf": 0.25, "distort": 0.10}   # infer.py's run weights
SVC_W, ALPHA = 12.0, 0.30   # K's settings


def key(t):
    """infer.py's _norm_text_key."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(t or ""))).strip()


def weighted_draw(g, n):
    """g.sample(n, weights='match_weight', random_state=0) as pandas 2.2 (Kaggle's image) runs it. pandas 3 refuses a
    draw without replacement when n x the largest normalised weight exceeds 1, which K's draw does (weights up to 50)."""
    w = np.nan_to_num(g.match_weight.to_numpy(dtype=np.float64))
    return g.take(np.random.RandomState(0).choice(len(g), size=n, replace=False, p=w / w.sum()).astype(np.intp))


def extra_rows(variant, pool_path, mc_path, train_texts):
    """K's draw (per label min(n, scale x nlm_matched_clean's count), weights match_weight, random_state 0), then infer.py's
    exact de-duplication against the official rows and earlier external rows."""
    scale, _ = VARIANTS[variant]
    mc = pd.read_parquet(mc_path, columns=["text", "model"])
    if scale is None:
        e = mc
    else:
        op = pd.read_parquet(pool_path, columns=["text", "model", "match_weight"])
        mix = mc.model.value_counts()
        e = pd.concat([weighted_draw(g, min(len(g), int(scale * mix.get(m, 0)))) for m, g in op.groupby("model")])
    if variant in ADD:
        e = pd.concat([e, pd.read_parquet(os.environ[f"NTL_ADD_{ADD[variant].upper()}"], columns=["text", "model"])])
    e = e[e.model.isin(E.NLM_LABELS)].copy()
    e["text"] = e.text.fillna("").astype(str)
    seen, keep = {key(t) for t in train_texts}, []
    for t in e.text.values:
        k = key(t)
        keep.append(bool(k) and k not in seen)
        if keep[-1]:
            seen.add(k)
    return e.loc[keep, ["text", "model"]].reset_index(drop=True)


def fit_fold(args):
    """One member on one fold: infer.py's code with FIT_T = the other folds' official rows + the external rows."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    member, tr_t, tr_y, ex_t, ex_y, extra_w, va_t = args
    t0 = time.time()
    fit_t = np.concatenate([tr_t, ex_t])
    fit_y = np.concatenate([tr_y, ex_y])
    fit_w = np.concatenate([np.ones(len(tr_t), dtype=np.float32), np.full(len(ex_t), extra_w, dtype=np.float32)])
    if member == "tfidf":
        mats_tr, mats_te = [], []
        for kw in [dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=60000),
                   dict(token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), min_df=3, max_features=30000)]:
            vec = TfidfVectorizer(sublinear_tf=True, lowercase=False, **kw).fit(fit_t)
            mats_tr.append(vec.transform(fit_t)); mats_te.append(vec.transform(va_t))
        lp = LogisticRegression(C=20, max_iter=3000).fit(sp.hstack(mats_tr).tocsr(), fit_y, sample_weight=fit_w) \
            .predict_log_proba(sp.hstack(mats_te).tocsr())
    else:
        wre = re.compile(r"[A-Za-z’']+")
        cnt = collections.Counter(w.lower() for t in fit_t for w in wre.findall(t))
        keep = {w for w, _ in cnt.most_common(100)}
        dist = lambda arr: [wre.sub(lambda m_: m_.group(0) if m_.group(0).lower() in keep else "*", t) for t in arr]
        d_tr, d_te = dist(fit_t), dist(va_t)
        vs = [TfidfVectorizer(sublinear_tf=True, lowercase=False, analyzer="char", ngram_range=(2, 6), min_df=2),
              TfidfVectorizer(sublinear_tf=True, lowercase=True, token_pattern=r"[^\s]+", ngram_range=(1, 3), min_df=2)]
        xa = sp.hstack([v_.fit_transform(d_tr) for v_ in vs]).tocsr()
        xb = sp.hstack([v_.transform(d_te) for v_ in vs]).tocsr()
        lp = LogisticRegression(C=10, max_iter=3000).fit(xa, fit_y, sample_weight=fit_w).predict_log_proba(xb)
    assert lp.shape[1] == 18, "every label must be in each fit"
    return lp, time.time() - t0


def fit(pool_path, mc_path, names, workers):
    """Each (variant, member, fold) is saved as it finishes (OUT/folds) and skipped on a rerun. A TF-IDF fit on 4x's 97k
    rows peaks at several GB, so keep the workers few (exp034's first run, 12 workers, ran the Mac into swap)."""
    (OUT / "folds").mkdir(parents=True, exist_ok=True)
    train = load_train()
    folds = np.load(E.NLM_OOF)["folds"]
    l2i = {l: i for i, l in enumerate(E.NLM_LABELS)}
    texts, y = train.text.fillna("").astype(str).to_numpy(), train.model.map(l2i).to_numpy()
    part = lambda v, m, k: OUT / "folds" / f"{v}_{m}_{k}.npy"
    jobs, slots = [], []
    for v in names:
        ex = extra_rows(v, pool_path, mc_path, texts)
        print(f"{v}: {len(ex)} external rows after de-duplication; per label "
              f"{ex.model.value_counts().min()}-{ex.model.value_counts().max()}", flush=True)
        ex_t, ex_y = ex.text.to_numpy(), ex.model.map(l2i).to_numpy()
        for m in MEMBERS:
            for k in np.unique(folds):
                if not part(v, m, k).exists():
                    jobs.append((m, texts[folds != k], y[folds != k], ex_t, ex_y, VARIANTS[v][1], texts[folds == k]))
                    slots.append((v, m, k))
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        for (v, m, k), (lp, secs) in zip(slots, pool.map(fit_fold, jobs)):
            np.save(part(v, m, k), lp)
            print(f"{v} {m}: fold {k} done in {secs:.0f}s", flush=True)
    for v in names:
        for m in MEMBERS:
            lp = np.zeros((len(train), 18))
            for k in np.unique(folds):
                lp[folds == k] = np.load(part(v, m, k))
            np.save(OUT / f"{v}_{m}.npy", lp)
            print(f"{v} {m} alone: {balanced_accuracy(train.model, np.array(E.NLM_LABELS)[lp.argmax(1)]):.4f}", flush=True)


def compare():
    train = load_train()
    nlm = np.load(E.NLM_OOF)
    folds, labels = nlm["folds"], np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(REPO_ROOT / "outputs" / "exp030_open_pool_svc" / "svc4_open_oof.npy").astype(np.float64)
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    members["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    weights = {**E.CORE, "svc4": SVC_W, "deb": 1.0}
    have = sorted({p.stem.rsplit("_", 1)[0] for p in OUT.glob("*_*.npy")}, key=list(VARIANTS).index)
    lps = {(v, m): np.load(OUT / f"{v}_{m}.npy") for v in have for m in MEMBERS}
    score = lambda P: balanced_accuracy(train.model, labels[P.argmax(1)])
    for v in have:
        print(f"{v:5s} alone: " + "  ".join(f"{m} {score(lps[v, m]):.4f}" for m in MEMBERS))
    E.TERMS.update({m: 1 for m in MEMBERS})

    def run(v, w=None, dw=None):
        ws = {**weights, "tfidf": MEMBERS["tfidf"] if w is None else w, "distort": MEMBERS["distort"] if dw is None else dw}
        return soft_sinkhorn(E.blend({**members, "tfidf": lps[v, "tfidf"], "distort": lps[v, "distort"]}, ws), alpha=ALPHA)

    ref = run("mc")
    print(f"\nreplica (core + deb, open-pool SVC at {SVC_W}, Sinkhorn {ALPHA}) with nlm_matched_clean: {score(ref):.4f}")
    for v in have[1:]:
        for w in (0.25, 0.5, 1.0):
            P = run(v, w)
            g = paired_bootstrap(y, P.argmax(1), ref.argmax(1), prompts)
            print(f"  {v} tfidf {w}: {score(P):.4f} (gain {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}])")


if __name__ == "__main__":
    if sys.argv[1:2] == ["fit"]:
        fit(sys.argv[2], sys.argv[3], [a for a in sys.argv[4:] if a in VARIANTS] or list(VARIANTS),
            int(os.environ.get("EXP034_WORKERS", "4")))
    else:
        compare()
