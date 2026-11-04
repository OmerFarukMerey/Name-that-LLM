"""exp046: should a pool-trained member weight the open pool's rows by their typography match_weight? The J / K draws
(rows drawn in proportion to match_weight) moved the LB, and the pool's own study (dataq4_paired.csv: typography-matched
sampling +0.026 balanced accuracy on 45k passages) points the same way, but our TPU members' stage 1 reads the pool
uniformly. A cheap proxy: infer.py's TF-IDF member fitted on one fixed pool sample (EXP046_N per label) with per-row
loss weights match_weight^p, normalised to mean 1 within each label (the class balance stays) and clipped to
[1/c, c], scored on every official row (the pool holds no train-prompt rows, so that is out of sample).

    python -I experiments/exp046_pool_weighting.py <ntl_open_pool.parquet>
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402

ARMS = [(0.0, 4.0), (0.5, 4.0), (1.0, 4.0), (2.0, 4.0), (1.0, 10.0)]   # (power p, clip c)


def row_weights(d, p, c):
    w = np.clip(d.match_weight.fillna(1.0).to_numpy(np.float64), 1e-3, None) ** p
    w = pd.Series(w).groupby(d.model.to_numpy()).transform(lambda s: s / s.mean()).to_numpy()
    w = np.clip(w, 1 / c, c)
    return pd.Series(w).groupby(d.model.to_numpy()).transform(lambda s: s / s.mean()).to_numpy()


def fit(args):
    """infer.py's TF-IDF member (char_wb 2-5 60k + word/punct 1-2 30k, LR C 20, balanced classes) with row weights."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    p, c, d, texts = args
    vs = [TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=60000, sublinear_tf=True, lowercase=False),
          TfidfVectorizer(token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True,
                          lowercase=False)]
    Xa = sp.hstack([v.fit_transform(d.text) for v in vs]).tocsr()
    Xt = sp.hstack([v.transform(texts) for v in vs]).tocsr()
    w = row_weights(d, p, c)
    ess = float(w.sum() ** 2 / (w ** 2).sum() / len(w))
    lr = LogisticRegression(C=20, max_iter=3000, class_weight="balanced").fit(Xa, d.model, sample_weight=w)
    return p, c, ess, lr.predict(Xt)


def main(pool_path):
    n = int(os.environ.get("EXP046_N", "4000"))
    op = pd.read_parquet(pool_path, columns=["text", "model", "match_weight"])
    d = op.sample(frac=1.0, random_state=0).groupby("model").head(n).reset_index(drop=True)
    train = load_train()
    print(f"pool {len(op)} rows; sample {len(d)} ({n} per label at most)", flush=True)
    texts = train.text.astype(str).tolist()
    fams = {f: [m for m in sorted(train.model.unique()) if m.startswith(f)] for f in ("claude", "gemini", "gpt", "glm")}
    preds = {}
    with ProcessPoolExecutor(max_workers=int(os.environ.get("EXP046_WORKERS", "4"))) as pe:
        for p, c, ess, pred in pe.map(fit, [(p, c, d, texts) for p, c in ARMS]):
            preds[(p, c)] = pred
            rec = pd.Series(pred == train.model.values).groupby(train.model.values).mean()
            print(f"match_weight^{p} clip {c:4.1f} (ESS {ess:.2f}): balanced acc {balanced_accuracy(train.model, pred):.4f} | "
                  + "  ".join(f"{f} {rec[ms].mean():.3f}" for f, ms in fams.items()), flush=True)
    from ntl.oofbank import paired_bootstrap
    y = train.model.map({l: i for i, l in enumerate(sorted(train.model.unique()))}).to_numpy()
    to_i = lambda pr: pd.Series(pr).map({l: i for i, l in enumerate(sorted(train.model.unique()))}).to_numpy()
    base = to_i(preds[ARMS[0]])
    for k in ARMS[1:]:
        gap = paired_bootstrap(y, to_i(preds[k]), base, train.prompt_id.to_numpy())
        print(f"  p {k[0]} clip {k[1]} vs uniform: {gap[0]:+.4f} [{gap[1]:+.4f}, {gap[2]:+.4f}]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
