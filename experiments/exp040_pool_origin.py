"""exp040: crtproduction's open pool flags 69,745 of its 190,588 passages `flag_kb_proxy` (every label but
gemini-3.7-flash and qwen3.8-max has thousands; about half of each GPT label). If the competition's own passages came
through the same proxy, those rows would be the closest to the hidden test. This fits infer.py's TF-IDF member on pool
rows only, the same count per label from each arm (proxy rows, non-proxy rows, and the match_weight draw K uses), and
scores it on every official row: the pool holds no train-prompt rows, so that is an honest out-of-sample test.

    python -I experiments/exp040_pool_origin.py <ntl_open_pool.parquet>

EXP040_N (default 1500) is the count per label; labels without that many rows in both arms (gemini-3.7-flash,
qwen3.8-max) put the same non-proxy rows in both arms. -I because the pool is a download; the repo's src/ and
experiments/ are put on the path below.
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

from exp039_external_corpora import fit  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402


def main(pool_path):
    n = int(os.environ.get("EXP040_N", "1500"))
    op = pd.read_parquet(pool_path, columns=["text", "model", "flag_kb_proxy", "match_weight"])
    train = load_train()
    arms, shared = {"proxy": [], "non-proxy": [], "match_weight": []}, []
    for m, g in op.groupby("model"):
        px, npx = g[g.flag_kb_proxy], g[~g.flag_kb_proxy]
        if len(px) >= n and len(npx) >= n:
            arms["proxy"].append(px.sample(n, random_state=0)); arms["non-proxy"].append(npx.sample(n, random_state=0))
        else:
            s = npx.sample(min(n, len(npx)), random_state=0); shared.append(m)
            arms["proxy"].append(s); arms["non-proxy"].append(s)
        w = g.match_weight.to_numpy(np.float64)   # numpy draw: pandas 3 refuses this weighted sample without replacement
        arms["match_weight"].append(g.iloc[np.random.default_rng(0).choice(len(g), min(n, len(g)), replace=False, p=w / w.sum())])
    print("labels with the same non-proxy rows in both arms:", shared, flush=True)
    jobs = [(a, pd.concat(v)[["text", "model"]], train.text.astype(str).tolist()) for a, v in arms.items()]
    for a, v in arms.items():
        print(f"{a}: proxy share {pd.concat(v).flag_kb_proxy.mean():.3f}", flush=True)
    fams = {f: [m for m in sorted(train.model.unique()) if m.startswith(f)] for f in ("claude", "gemini", "gpt", "glm")}
    with ProcessPoolExecutor(max_workers=3) as pe:
        for name, rows, p in pe.map(fit, jobs):
            rec = pd.Series(p == train.model.values).groupby(train.model.values).mean()
            both = [m for m in rec.index if m not in shared]
            print(f"{name:13s} {rows:6d} rows: balanced acc {balanced_accuracy(train.model, p):.4f} | mean recall on the {len(both)} "
                  f"labels with both arms {rec[both].mean():.4f} | " + "  ".join(f"{f} {rec[ms].mean():.3f}" for f, ms in fams.items()),
                  flush=True)
            print("   " + " ".join(f"{k}={v:.2f}" for k, v in rec.items()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
