"""Build ntl-stack.ipynb from the upstream public notebook plus our changes.

    python kaggle/stack/build.py SVC_POOL_WEIGHT=0.3 SVC_LOW_WEIGHT=0.1 SVC_WEIGHT_SCALE=1.64

upstream.ipynb is kirilltym's "0.639 | Teach More, Keep the Signature" (version 1, the scored
0.639), kept byte-for-byte. It fails closed: a missing component raises instead of silently
falling back. (ntl-stack v1-v3 used maverickss26's restructured v5 instead, kept as
upstream_maverickss26_v5.ipynb; its fail-safe fallback scored 0.620 on our hidden rerun.)
Our changes are:
  1. a knob cell (DEFAULTS, overridable on the command line) just before the SVC cell,
  2. cells inserted next to an anchor cell (INSERTS),
  3. exact substitutions inside upstream cells (PATCHES).
Every patch must match exactly once, so an upstream edit fails the build loudly.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The n-gram SVC (Stage B, weight 1.0): public-pool rows at a lower sample weight than host rows. Pool rows that
# are also in fleongg's quality-selected pool get SVC_POOL_WEIGHT, the others SVC_LOW_WEIGHT. The member's blend
# weight is scaled by the ratio of calibrated temperatures so its influence stays as tuned upstream.
# Evidence: experiments/exp004-exp006. Defaults reproduce the upstream notebook exactly.
DEFAULTS = {"SVC_POOL_WEIGHT": "1.0", "SVC_LOW_WEIGHT": "1.0", "SVC_WEIGHT_SCALE": "1.0"}
INSERTS = []  # (anchor text in an upstream code cell, "before" / "after", our cell file)
PATCHES = [  # (old, new)
    ('class_weight="balanced").fit(hstack(A, format="csr"), fit_y)',
     'class_weight="balanced").fit(hstack(A, format="csr"), fit_y,\n'
     '        sample_weight=np.r_[np.ones(len(train)), np.where(pool.text.isin(QUALITY_TEXTS), SVC_POOL_WEIGHT, SVC_LOW_WEIGHT)])'),
    ('    pool = pd.read_parquet(find("pool_v4.parquet"))\n',
     '    pool = pd.read_parquet(find("pool_v4.parquet"))\n'
     '    QUALITY_TEXTS = set(pd.read_parquet(find("quality_pool_public.parquet"), columns=["text"]).text)\n'
     '    print(f"SVC pool: {pool.text.isin(QUALITY_TEXTS).sum()} of {len(pool)} rows in the quality pool")\n'),
    ('vote("svc", softmax(svc.decision_function(hstack(B, format="csr"))), W_SVC)',
     'vote("svc", softmax(svc.decision_function(hstack(B, format="csr"))), W_SVC * SVC_WEIGHT_SCALE)'),
]


def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def src(cell):
    return "".join(cell["source"])


def find_cell(cells, anchor):
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and anchor in src(c)]
    if len(hits) != 1:
        raise SystemExit(f"anchor matched {len(hits)} cells: {anchor!r}")
    return hits[0]


def main(overrides):
    knobs = dict(DEFAULTS, **dict(o.split("=", 1) for o in overrides))
    unknown = set(knobs) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f"unknown knobs: {sorted(unknown)}")

    nb = json.loads((HERE / "upstream.ipynb").read_text())
    cells = nb["cells"]
    for anchor, where, name in INSERTS:
        i = find_cell(cells, anchor)
        cells.insert(i if where == "before" else i + 1, code_cell((HERE / name).read_text()))
    for old, new in PATCHES:
        i = find_cell(cells, old)
        cells[i]["source"] = src(cells[i]).replace(old, new).splitlines(keepends=True)
    cells.insert(find_cell(cells, 'vote("svc"'), code_cell(
        "# ---- our knobs (kaggle/stack/build.py); all 1.0 reproduces the upstream notebook\n"
        + "".join(f"{k} = {v}\n" for k, v in knobs.items())))
    (HERE / "ntl-stack.ipynb").write_text(json.dumps(nb, indent=1))
    print("built ntl-stack.ipynb with", knobs)


if __name__ == "__main__":
    main(sys.argv[1:])
