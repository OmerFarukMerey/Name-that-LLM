"""exp003_svc_pools: which public passage pool makes the stack's n-gram SVC best on host text?

The public 0.639 stack fits its 4-view LinearSVC (weight 1.0, its strongest single
member) on train + pool_v4 (75.8k passages). Larger public pools exist. Grouped
5-fold CV on the host passages only; pool rows whose content-word TF-IDF cosine to any
train passage is >= 0.3 are dropped first, so no prompt leaks into validation.

    python experiments/exp003_svc_pools.py pool_v4 nuri122k nlm_v5 union
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer

from ntl.cv import iter_folds
from ntl.data import LABELS, load_train
from ntl.metrics import balanced_accuracy, per_class_recall
from ntl.paths import OUTPUT_DIR, REPO_ROOT

sys.path.insert(0, str(Path(__file__).parent))
from exp002_pseudo_label_sim import svc_proba  # noqa: E402  (the stack's exact 4-view SVC)

EXT = REPO_ROOT / "data" / "external"
POOLS = {
    "pool_v4": EXT / "handwriting-blend/pool_v4.parquet",
    "nuri122k": EXT / "nuri130k/ntl_encoder_pool_122k.parquet",
    "nlm_v5": EXT / "nlm/nlm_pool_v5.parquet",
    "curated": EXT / "curated/curated_pool.parquet",
}
OUT = OUTPUT_DIR / "exp003_svc_pools"


def load_pool(name, train):
    if name == "union":
        frames = [load_pool(n, train) for n in ("pool_v4", "nuri122k", "nlm_v5")]
        return pd.concat(frames).drop_duplicates("text").reset_index(drop=True)
    p = pd.read_parquet(POOLS[name])[["text", "model"]]
    p = p[p.model.isin(LABELS) & ~p.text.str.contains("\n")].drop_duplicates("text")
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit(train.text.tolist() + p.text.tolist())
    sim = (vec.transform(p.text) @ vec.transform(train.text).T).max(1).toarray().ravel()
    print(f"{name}: {len(p)} rows, dropping {(sim >= 0.3).sum()} similar to a train prompt", flush=True)
    return p[sim < 0.3].reset_index(drop=True)


def run_fold(train, pool, tr, va):
    texts = train.text.iloc[tr].tolist() + pool.text.tolist()
    y = np.r_[train.model.iloc[tr].map(LABELS.index).values, pool.model.map(LABELS.index).values]
    return va, svc_proba(texts, y, train.text.iloc[va].tolist())


def main(names):
    train = load_train()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in names:
        pool = load_pool(name, train)
        oof = np.zeros((len(train), len(LABELS)))
        for va, p in Parallel(n_jobs=3)(delayed(run_fold)(train, pool, tr, va) for _, tr, va in iter_folds(train.prompt_id)):
            oof[va] = p
        pred = np.array(LABELS)[oof.argmax(1)]
        results[name] = dict(rows=len(pool), cv=balanced_accuracy(train.model, pred))
        print(name, results[name], flush=True)
        print(per_class_recall(train.model, pred).round(3).to_dict(), flush=True)
        np.save(OUT / f"oof_{name}.npy", oof)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["pool_v4", "nuri122k", "nlm_v5", "union"])
