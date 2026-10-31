"""exp006_svc_quality_tiers: does a quality filter on the public pool add to the pool down-weighting of exp004?

fleongg's quality-selected pool (40.1k public passages; 87.5% of them are also in pool_v4) drops
passages flagged for label conflicts, paragraph breaks, code or thinking, refusals and format
issues. Same grouped 5-fold protocol and SVC as exp005 (C 0.5, balanced classes).

Variants:
  q_wNN        the quality pool alone, pool sample weight 0.NN
  tier_wNN_NN  pool_v4; rows also in the quality pool at the first weight, the others at the second
  utier_wNN_NN the same on pool_v4 plus the quality passages it lacks

    python experiments/exp006_svc_quality_tiers.py q_w03 q_w05 tier_w03_01
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC

from ntl.cv import iter_folds
from ntl.data import LABELS, load_train
from ntl.metrics import balanced_accuracy
from ntl.paths import OUTPUT_DIR

sys.path.insert(0, str(Path(__file__).parent))
from exp003_svc_pools import EXT, POOLS, load_pool  # noqa: E402
from exp004_svc_variants import features  # noqa: E402
from exp005_svc_regularisation import BASE_OOF, number, paired_bootstrap  # noqa: E402

POOLS["quality"] = EXT / "quality/quality_pool_public.parquet"
OUT = OUTPUT_DIR / "exp006_svc_quality_tiers"


def pool_and_weights(variant, train):
    kind, *ws = variant.split("_")
    ws = [number(w.lstrip("w")) for w in ws]
    if kind == "q":
        pool = load_pool("quality", train)
        return pool, np.full(len(pool), ws[0])
    pool = load_pool("pool_v4", train)
    if kind == "utier":  # also the quality passages that pool_v4 lacks
        pool = pd.concat([pool, load_pool("quality", train)[["text", "model"]]]).drop_duplicates("text").reset_index(drop=True)
    good = pool.text.isin(set(pd.read_parquet(POOLS["quality"], columns=["text"]).text))
    print(f"{variant}: {good.sum()} of {len(pool)} pool_v4 rows are in the quality pool", flush=True)
    return pool, np.where(good, ws[0], ws[1])


def run_fold(train, pool, pool_w, tr, va):
    texts = train.text.iloc[tr].tolist() + pool.text.tolist()
    y = np.r_[train.model.iloc[tr].map(LABELS.index).values, pool.model.map(LABELS.index).values]
    X, Xv = features(texts, train.text.iloc[va].tolist())
    svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")
    d = svc.fit(X, y, sample_weight=np.r_[np.ones(len(tr)), pool_w]).decision_function(Xv)
    e = np.exp(d - d.max(1, keepdims=True))
    return va, e / e.sum(1, keepdims=True)


def main(variants):
    train = load_train()
    OUT.mkdir(parents=True, exist_ok=True)
    y, groups = train.model.to_numpy(), train.prompt_id.to_numpy()
    base = np.array(LABELS)[np.load(BASE_OOF).argmax(1)]
    results = {}
    for v in variants:
        pool, pool_w = pool_and_weights(v, train)
        oof = np.zeros((len(train), len(LABELS)))
        for va, p in Parallel(n_jobs=3)(delayed(run_fold)(train, pool, pool_w, tr, va) for _, tr, va in iter_folds(train.prompt_id)):
            oof[va] = p
        np.save(OUT / f"oof_{v}.npy", oof)
        pred = np.array(LABELS)[oof.argmax(1)]
        results[v] = dict(rows=len(pool), cv=balanced_accuracy(y, pred), gain_vs_w1_ci95=paired_bootstrap(y, base, pred, groups))
        print(v, results[v], flush=True)
    (OUT / f"metrics_{'_'.join(variants)}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["q_w03", "q_w05", "tier_w03_01"])
