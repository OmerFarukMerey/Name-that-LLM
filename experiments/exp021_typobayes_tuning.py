"""exp021: tuning Roman Tamrazov's typography step (exp020, public 0.654) out of fold: its knobs, more markers, and
class rates helped by fleongg's public quality pool.

Base: exp015's `--plain` replica (core + DeBERTa stand-in, Sinkhorn) with no post step, since the hidden test has only a
few answers per prompt (exp020); optionally harshitakumari256's Qwen3-4B ATLAS OOF at 10% on low-margin rows (exp019) as
a second base. The step: evidence e = per-row-centred mean Bernoulli log-likelihood of binary markers under class rates
(hits + k p0) / (n + k) (Roman: k = 4, p0 = 0.5, i.e. (hits + 2) / (n + 4)), clipped to [0.015, 0.985]; prediction
argmax(log p + gate * clip(scale * e, -c, c)) with gate = clip((g - top-2 margin) / g, 0, 1). Roman: scale 2, c 0.35,
g 0.8. Rates come from the other folds of train (prompt-grouped). Knobs are compared on a grid, and also tuned on the
other folds and applied to the held-out fold (cross-fitted), which is the honest estimate of what tuning buys.

    PYTHONPATH=src python experiments/exp021_typobayes_tuning.py [path/to/atlas4b/oof_all.parquet]
"""

import itertools
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax

import exp020_typobayes_oof as T  # noqa: E402  (appends --plain and imports exp015 as T.E)
from ntl.data import load_train
from ntl.metrics import balanced_accuracy
from ntl.oofbank import paired_bootstrap
from ntl.paths import REPO_ROOT

E = T.E
POOL = REPO_ROOT / "data/external/quality/quality_pool_public.parquet"
# typography and discourse markers absent from Roman's list, chosen before looking at any result
EXTRA = ["'", '...', '  ', '--', '\n\n\n', ':\n', '\n* ', '\n#', '##', '|', '!', '?', '%', '$', '→', '×', '≈',
         'Additionally,', 'Furthermore,', 'In summary', 'Ultimately,', 'Notably,', 'Importantly,', 'Finally,',
         'For example,', 'In short,', 'Essentially,', "Here's", 'Here’s', "Let's", 'Let’s', "it's", 'it’s',
         "isn't", 'isn’t', "don't", 'don’t']
SCALES, CLIPS, GATES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0), (0.2, 0.35, 0.5, 0.75, 1.0), (0.5, 0.8, 1.0, 1.5)
DEFAULT = (2.0, 0.35, 0.8)


def markers(texts, marks):
    return np.array([[m in t for m in marks] for t in texts], dtype=np.float64)


def rates(X, y, k=4.0, p0=0.5):
    n = np.bincount(y, minlength=len(E.NLM_LABELS)).astype(float)
    hits = np.stack([X[y == c].sum(0) for c in range(len(E.NLM_LABELS))])
    return np.clip((hits + k * p0) / (n[:, None] + k), 0.015, 0.985)


def evidence(X, rate):
    e = (X @ np.log(rate).T + (1 - X) @ np.log(1 - rate).T) / X.shape[1]
    return e - e.mean(1, keepdims=True)


def oof_evidence(X, y, folds, k=4.0, prior=None):
    """centred evidence on each fold from rates fitted on the other folds; prior: (18, M) pool rates as p0."""
    ev = np.zeros((len(X), len(E.NLM_LABELS)))
    for f in np.unique(folds):
        fit = folds != f
        ev[~fit] = evidence(X[~fit], rates(X[fit], y[fit], k, 0.5 if prior is None else prior))
    return ev


def apply(raw, margin, ev, scale, clip, g):
    gate = np.clip((g - margin) / g, 0., 1.)
    return (raw + gate[:, None] * np.clip(scale * ev, -clip, clip)).argmax(1)


def main():
    train = load_train()
    labels = np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts, texts = train.prompt_id.to_numpy(), train.text.to_numpy()
    nlm = np.load(E.NLM_OOF); folds = nlm["folds"]
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(E.OUT / "svc4_oof.npy").astype(np.float64)
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    assert np.array_equal(z["ids"], train.id.to_numpy(dtype=str))
    members["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    score = lambda pred: balanced_accuracy(train.model, labels[pred])

    P0 = E.blend(members, {**E.CORE, "deb": 1.0})
    bases = {"replica": P0}
    if len(sys.argv) > 1 and sys.argv[1].endswith(".parquet"):
        a = pd.read_parquet(sys.argv[1]).set_index("id").loc[train.id.astype(str)]
        LL = a[[f"LL_{l}" for l in E.NLM_LABELS]].to_numpy(np.float64)
        A = np.exp(14.820207057988586 * (LL - LL.max(1, keepdims=True))); A /= A.sum(1, keepdims=True)
        s = np.sort(P0, 1); rows = (s[:, -1] - s[:, -2]) < 0.3
        P = P0.copy(); P[rows] = 0.9 * P0[rows] + 0.1 * A[rows]
        bases["replica + atlas4b 10%"] = P

    pool = pd.read_parquet(POOL, columns=["model", "text"])
    pool = pool[pool.model.isin(E.NLM_LABELS)]
    yp = pool.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    print(f"pool: {len(pool)} rows, {pool.model.nunique()} classes")
    X34, Xx = markers(texts, T.MARKS), markers(texts, T.MARKS + EXTRA)
    P34, Pxx = markers(pool.text.to_numpy(), T.MARKS), markers(pool.text.to_numpy(), T.MARKS + EXTRA)
    evs = {"Roman 34, train rates (v38a)": oof_evidence(X34, y, folds),
           "Roman 34, pool rates only": evidence(X34, rates(P34, yp)),
           "Roman 34, train + pool prior k=20": oof_evidence(X34, y, folds, 20.0, rates(P34, yp)),
           "Roman 34, train + pool prior k=100": oof_evidence(X34, y, folds, 100.0, rates(P34, yp)),
           f"Roman 34 + {len(EXTRA)} extra, train rates": oof_evidence(Xx, y, folds),
           f"Roman 34 + {len(EXTRA)} extra, train + pool prior k=20": oof_evidence(Xx, y, folds, 20.0, rates(Pxx, yp))}
    for name, ev in evs.items():
        print(f"evidence alone, {name:45s} {score(ev.argmax(1)):.4f}")

    grid = list(itertools.product(SCALES, CLIPS, GATES))
    for bname, P in bases.items():
        raw = np.log(E.sinkhorn(P) + 1e-12)
        q = softmax(raw, 1); s = np.sort(q, 1); margin = s[:, -1] - s[:, -2]
        none = raw.argmax(1)
        ref = apply(raw, margin, evs["Roman 34, train rates (v38a)"], *DEFAULT)
        print(f"\n== base: {bname}; no step {score(none):.4f}, v38a default {score(ref):.4f} "
              f"(vs no step {paired_bootstrap(y, ref, none, prompts)[0]:+.4f})")
        for ename, ev in evs.items():
            preds = {cfg: apply(raw, margin, ev, *cfg) for cfg in grid}
            sc = {cfg: score(p) for cfg, p in preds.items()}
            best = sorted(sc, key=sc.get, reverse=True)
            # cross-fitted tuning: knobs chosen on the other folds, applied to the held-out fold
            cf = np.empty_like(none); picks = []
            for f in np.unique(folds):
                tr = folds != f
                pick = max(grid, key=lambda cfg: balanced_accuracy(y[tr], preds[cfg][tr]))
                cf[~tr] = preds[pick][~tr]; picks.append(pick)
            d = paired_bootstrap(y, preds[DEFAULT], ref, prompts)
            c = paired_bootstrap(y, cf, ref, prompts)
            print(f"  {ename:45s} default knobs {sc[DEFAULT]:.4f} ({d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}])  "
                  f"cross-fitted tuning {score(cf):.4f} ({c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]) picks {picks}")
            print(f"    grid best: " + "; ".join(f"{cfg} {sc[cfg]:.4f}" for cfg in best[:5])
                  + f"; default rank {best.index(DEFAULT) + 1}/{len(grid)}")
            if ename.endswith("(v38a)"):
                for g in GATES:   # the plateau around Roman's knobs, one gate at a time
                    print(f"    gate {g}: " + " ".join(f"s{sv:g}/c{cv:g} {sc[(sv, cv, g)]:.4f}"
                                                   for sv in SCALES for cv in CLIPS if cv in (0.35, 0.75)))


if __name__ == "__main__":
    main()
