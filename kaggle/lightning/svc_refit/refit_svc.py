"""Refit lightningv08's ntl627-svc-prefit (Nishant Dahal's part-two 4-view n-gram LinearSVC) offline,
optionally with per-row sample weights, and write it in exactly the published cache format.

Fit data and order exactly as upstream: train.csv rows (pd.read_csv, text.astype(str)) first, then
pool_v4.parquet rows unfiltered (75,783), labels encoded in the cache's label order. Four TF-IDF
views (min_df=2, max_features per view, float32, view weight multiplied after fit_transform),
hstack csr, LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").

Sample weights: train rows --train-weight; pool rows whose text is in the quality pool
--quality-weight; the other pool rows --other-weight. All 1 (the default) is the published fit
and passes sample_weight=None like upstream. The TF-IDF fit ignores sample weights, so the
vocabulary and idf do not depend on them.

    python kaggle/lightning/svc_refit/refit_svc.py --out <dir>             (the published fit, to check the recipe)
    python kaggle/lightning/svc_refit/refit_svc.py --out <dir> --quality-weight 0.3 --other-weight 0.1

The second command builds omerfarukmerey/ntl-svc627-quality-tiers (readout.npz sha256 5827d9da...; the dataset's
README.md was rewritten by hand before upload). The published-weights refit matches the published coef to 2e-7.

--features-from DIR transforms with DIR's vocabulary/idf instead of fitting them (fallback for
max_features tie-breaking differences). --x-cache DIR stores/reuses the fit feature matrix.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy.sparse import hstack, load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svc627_format import (LABELS, VIEWS, cached_svc, fit_options, fixed_vectorizer, prep_for,  # noqa: E402
                           write_json, write_manifest, write_readout, write_view)

REPO = Path(__file__).resolve().parents[3]
TRAIN = REPO / "data/raw/train.csv"
TEST = REPO / "data/raw/test.csv"
POOL = REPO / "data/external/handwriting-blend/pool_v4.parquet"
QUALITY = REPO / "data/external/quality/quality_pool_public.parquet"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def view_features(view, lo, hi, max_features, weight, texts, extra, fixed=None):
    """Fit (or, with fixed=(vocabulary, idf), only transform) one view; returns weighted matrices."""
    t0 = time.time()
    prep = prep_for(view)
    if fixed is None:
        vec = TfidfVectorizer(**fit_options(view, lo, hi, max_features))
        X = vec.fit_transform([prep(t) for t in texts]) * weight
        vocabulary = [t for t, _ in sorted(vec.vocabulary_.items(), key=lambda kv: kv[1])]
    else:
        vocabulary, idf = fixed
        vec = fixed_vectorizer(view, lo, hi, vocabulary, idf)
        X = vec.transform([prep(t) for t in texts]) * weight
    Xe = vec.transform([prep(t) for t in extra]) * weight
    return view, vocabulary, np.asarray(vec.idf_), X, Xe, time.time() - t0


def load_fit_data():
    train = pd.read_csv(TRAIN)
    pool = pd.read_parquet(POOL)
    lab = {l: i for i, l in enumerate(LABELS)}
    texts = train.text.astype(str).tolist() + pool.text.tolist()
    y = np.r_[train.model.map(lab).values, pool.model.map(lab).values]
    assert not pd.isna(y).any() and len(texts) == len(y)
    return train, pool, texts, y.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-weight", type=float, default=1.0)
    ap.add_argument("--quality-weight", type=float, default=1.0)
    ap.add_argument("--other-weight", type=float, default=1.0)
    ap.add_argument("--features-from", default=None, help="cache dir whose vocabulary/idf to reuse (transform only)")
    ap.add_argument("--x-cache", default=None, help="dir to store / reuse the fit feature matrix")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.time()

    train, pool, texts, y = load_fit_data()
    test = pd.read_csv(TEST)
    preview_ids = test.id.astype(str).tolist()
    preview_texts = test.text.fillna("").astype(str).tolist()
    in_quality = pool.text.isin(set(pd.read_parquet(QUALITY, columns=["text"]).text)).to_numpy()
    log(f"fit rows {len(texts)} = train {len(train)} + pool {len(pool)}; pool rows in quality pool {in_quality.sum()}")

    w = np.r_[np.full(len(train), a.train_weight), np.where(in_quality, a.quality_weight, a.other_weight)]
    sample_weight = None if np.all(w == 1.0) else w

    xc = Path(a.x_cache) if a.x_cache else None
    if xc is not None and (xc / "X.npz").exists():
        log(f"loading features from {xc}")
        X, Xp = load_npz(xc / "X.npz").tocsr(), load_npz(xc / "Xp.npz").tocsr()
        for view, *_ in VIEWS:
            for suffix in ("-vocabulary.json", "-idf.npy"):
                shutil.copyfile(xc / (view + suffix), out / (view + suffix))
        fs = xc / "feature_source.txt"
        feature_source = (fs.read_text() if fs.exists() else "fit here") + f" (via x-cache {xc.name})"
    else:
        fixed = {}
        if a.features_from:
            src = Path(a.features_from)
            for view, *_ in VIEWS:
                fixed[view] = (json.loads((src / (view + "-vocabulary.json")).read_text(encoding="utf-8")),
                               np.load(src / (view + "-idf.npy"), allow_pickle=False))
        res = Parallel(n_jobs=a.n_jobs, backend="loky", max_nbytes=None)(
            delayed(view_features)(view, lo, hi, mf, wt, texts, preview_texts, fixed.get(view)) for view, lo, hi, mf, wt in VIEWS)
        A, B = [], []
        for view, vocabulary, idf, Xv, Xe, secs in res:  # Parallel keeps the VIEWS order
            log(f"view {view}: {Xv.shape} nnz {Xv.nnz} idf {idf.dtype} in {secs:.0f}s")
            write_view(out, view, vocabulary, idf)
            A.append(Xv)
            B.append(Xe)
        X, Xp = hstack(A, format="csr"), hstack(B, format="csr")
        del A, B, res
        feature_source = (f"vocabulary/idf of {Path(a.features_from).resolve().name} (transform only)" if a.features_from
                          else "vocabulary/idf fit here")
        if xc is not None:
            xc.mkdir(parents=True, exist_ok=True)
            save_npz(xc / "X.npz", X, compressed=False)
            save_npz(xc / "Xp.npz", Xp, compressed=False)
            (xc / "feature_source.txt").write_text(feature_source)
            for view, *_ in VIEWS:
                for suffix in ("-vocabulary.json", "-idf.npy"):
                    shutil.copyfile(out / (view + suffix), xc / (view + suffix))
    log(f"features {X.shape} {X.dtype} nnz {X.nnz} after {time.time() - T0:.0f}s; fitting LinearSVC "
        f"(sample_weight {'None' if sample_weight is None else 'tiered'})")

    t1 = time.time()
    svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")
    svc.fit(X, y, sample_weight=sample_weight)
    fit_secs = time.time() - t1
    log(f"LinearSVC fit in {fit_secs:.0f}s, n_iter_ {svc.n_iter_}")
    assert np.array_equal(svc.classes_, np.arange(18))
    write_readout(out, svc.coef_, svc.intercept_)

    direct = svc.decision_function(Xp)
    weights_desc = dict(train=a.train_weight, pool_in_quality=a.quality_weight, pool_other=a.other_weight)
    meta = {
        "labels": LABELS,
        "views": [list(v) for v in VIEWS],
        "fit_rows": int(len(texts)),
        "pool_rows": int(len(pool)),
        "official_rows": int(len(train)),
        "source": "offline refit of lightningv08 ntl627-svc-prefit (mrpicklerick/ntl-0-627-encoder-ratio-6-to-1 v1 / "
                  "Nishant Dahal part two recipe)" + (f"; {a.note}" if a.note else ""),
        "upstream_scriptVersionId": 361459569,
        "inference_only": True,
        "sklearn_version": sklearn.__version__,
        "data_license": "CC-BY-SA-4.0",
        "fitted": True,
        "sample_weights": None if sample_weight is None else weights_desc,
        "pool_rows_in_quality_pool": int(in_quality.sum()),
        "quality_pool": "fleongg quality_pool_public.parquet (text match)",
        "logit_scale": 1.0,
        "feature_source": feature_source,
        "svc": dict(C=0.5, random_state=42, max_iter=5000, class_weight="balanced", n_iter=int(np.max(svc.n_iter_))),
        "platform": dict(numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__, machine=os.uname().machine),
    }
    write_json(out / "metadata.json", meta)
    portable = cached_svc(out, preview_texts)
    np.savez_compressed(out / "preview.npz", ids=np.asarray(preview_ids), direct=direct, portable=portable)
    drift = float(np.abs(direct - portable).max())
    write_json(out / "COMPLETE.json", {
        "status": "COMPLETE_PORTABLE_NEW_SVC",
        "max_logit_drift": drift,
        "fit_rows": int(len(texts)),
        "pool_rows": int(len(pool)),
        "GPU_used": False,
        "elapsed_seconds": time.time() - T0,
        "sample_weights": None if sample_weight is None else weights_desc,
    })
    wdesc = ("no sample weights (the published fit)" if sample_weight is None else
             f"sample weights: train rows {a.train_weight}, pool rows also in fleongg's quality pool {a.quality_weight} "
             f"({int(in_quality.sum())} rows), other pool rows {a.other_weight}")
    (out / "README.md").write_text(
        "Offline refit of the ntl627-svc-prefit four-view n-gram SVC, same file format and loader (cached_svc).\n"
        "Architecture and public training pool by Nishant Dahal,\n"
        "https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two\n"
        "https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend\n"
        "Pool version1, pool_v4.parquet; training-pool metadata license CC BY-SA4.0.\n"
        "Cache format and original full-data fit by lightningv08 (ntl-olmo2-native24-final-inference-assets).\n"
        f"This fit: train.csv + pool_v4.parquet = {len(texts)} rows; {wdesc}.\n"
        f"Features: {feature_source}. Logits are unscaled (logit_scale 1.0 in metadata.json).\n", encoding="utf-8")
    write_manifest(out, "ntl627-svc-prefit")
    log(f"done in {time.time() - T0:.0f}s; preview direct vs portable drift {drift:.3g}; wrote {out}")


if __name__ == "__main__":
    main()
