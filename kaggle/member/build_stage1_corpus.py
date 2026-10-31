"""Stage-1 corpus for the training track: fleongg's quality-selected public pool, minus prompts near a train prompt.

    PYTHONPATH=src python kaggle/member/build_stage1_corpus.py

Source: data/external/quality/quality_pool_public.parquet (40,129 rows from public Kaggle datasets, each row with its
CC BY 4.0 / CC BY-SA 4.0 license). fleongg's own flags (label conflict, overlap with official text, duplicates,
refusals, code or thinking, non-English) are already false on every row; this script asserts that.

A pool prompt about the same topic as a train prompt would leak into the out-of-fold check of a stage-1-trained
member, so a pool prompt group is dropped when its TF-IDF centroid is closer to some train prompt's centroid than
TOPIC_COSINE. 0.25 is the 90th percentile of the same distance between two different train prompts
(2026-10-25: drops 38 groups, 719 rows). Writes outputs/member_stage1/stage1_quality.parquet:
id, text, label (ntl.data.LABELS index), group (pool prompt group), source, license.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ntl.data import LABELS, load_train
from ntl.paths import OUTPUT_DIR, REPO_ROOT

POOL = REPO_ROOT / "data/external/quality/quality_pool_public.parquet"
OUT = OUTPUT_DIR / "member_stage1"
TOPIC_COSINE = 0.25
FLAGS = ["label_conflict", "exact_official_or_visible_overlap", "duplicate_response", "code_or_thinking",
         "refusal_surface_flag", "format_flag"]


def centroids(matrix, keys):
    names, inverse = np.unique(keys, return_inverse=True)
    members = sp.csr_matrix((np.ones(len(inverse)), (inverse, np.arange(len(inverse)))), shape=(len(names), len(inverse)))
    return names, normalize(members @ matrix)


def main():
    pool, train = pd.read_parquet(POOL), load_train()
    if pool[FLAGS].any().any() or not pool.model.isin(LABELS).all() or (pool.kind != "public").any():
        raise SystemExit("the pool has flagged, non-public or unknown-label rows; recheck before using it")
    if pool.text.str.strip().duplicated().any() or pool.text.str.strip().isin(set(train.text.str.strip())).any():
        raise SystemExit("duplicate text or official text in the pool")
    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, stop_words="english")
    vectorizer.fit(pd.concat([train.text, pool.text]))
    _, train_c = centroids(vectorizer.transform(train.text), train.prompt_id.to_numpy())
    groups, pool_c = centroids(vectorizer.transform(pool.text), pool.prompt_group_id.to_numpy())
    near = groups[(pool_c @ train_c.T).toarray().max(1) > TOPIC_COSINE]
    kept = pool[~pool.prompt_group_id.isin(near)]
    out = pd.DataFrame({"id": kept.id, "text": kept.text, "label": kept.model.map({m: i for i, m in enumerate(LABELS)}),
                        "group": kept.prompt_group_id, "source": kept.source, "license": kept.license})
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "stage1_quality.parquet", index=False)
    print(f"dropped {len(near)} prompt groups ({len(pool) - len(kept)} rows) near a train prompt; kept {len(out)} rows, "
          f"{out.group.nunique()} groups")
    print(out.label.map(dict(enumerate(LABELS))).value_counts().to_string())


if __name__ == "__main__":
    main()
