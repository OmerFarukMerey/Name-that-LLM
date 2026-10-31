"""exp022: more weak style evidence in the place of Roman Tamrazov's typography step (exp020/021), out of fold.

His step is the one public change whose replica gain (+0.003) matched the LB (+0.003): a train-only likelihood of a
weak signal (34 binary markers, 0.170 alone), clipped and added to the ensemble's log-probabilities on uncertain rows.
Candidates of the same kind, all fitted on the other folds (prompt-grouped) and added the same way
(argmax(log p + gate * clip(scale * e, -c, c)), e centred per row):
  nb34      Roman's Bernoulli likelihood (v38a; scale 2, clip 0.35, gate 0.8)
  shape     a diagonal Gaussian likelihood of 16 length / layout / rate features (log words, paragraphs, sentence
            length, commas per sentence, ...), averaged over the features like his
  lr        a multinomial logistic regression on his 34 markers + the 16 features (centred log-probabilities)
Scale is reported on a grid and tuned on the other folds (cross-fitted); clip and gate are his.

    PYTHONPATH=src python experiments/exp022_style_evidence.py [path/to/atlas4b/oof_all.parquet]
"""

import re
import sys

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import exp020_typobayes_oof as T  # noqa: E402  (appends --plain and imports exp015 as T.E)
import exp021_typobayes_tuning as U  # noqa: E402
from ntl.data import load_train
from ntl.metrics import balanced_accuracy
from ntl.oofbank import paired_bootstrap

E = T.E
SCALES = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
SENT = re.compile(r"[.!?]+(?:\s|$)")


def shape_features(texts):
    rows = []
    for t in texts:
        words = t.split()
        nw = max(len(words), 1)
        sents = max(len(SENT.findall(t)), 1)
        paras = [p for p in t.split("\n\n") if p.strip()]
        lines = [l for l in t.split("\n") if l.strip()]
        rows.append([np.log(nw), np.log(len(t)), len(paras), len(lines), nw / sents, np.mean([len(w) for w in words]) if words else 0.0,
                     len({w.lower() for w in words}) / nw, t.count(",") / sents, t.count(",") / nw * 100,
                     sum(w[:1].isupper() for w in words) / nw, sum(c.isdigit() for c in t) / len(t) * 100,
                     t.count("(") / nw * 100, t.count(":") / nw * 100, (t.count("—") + t.count("–")) / nw * 100,
                     np.mean([len(p.split()) for p in paras]) if paras else nw, sum(l.lstrip()[:1] in "-*•" or l.lstrip()[:2].rstrip(".").isdigit() for l in lines) / max(len(lines), 1)])
    return np.array(rows, dtype=np.float64)


def gauss_evidence(F, y, folds):
    ev = np.zeros((len(F), len(E.NLM_LABELS)))
    for f in np.unique(folds):
        fit = folds != f
        sc = StandardScaler().fit(F[fit]); Z, Zv = sc.transform(F[fit]), sc.transform(F[~fit])
        mu = np.stack([Z[y[fit] == c].mean(0) for c in range(len(E.NLM_LABELS))])
        sd = np.stack([Z[y[fit] == c].std(0) + 0.1 for c in range(len(E.NLM_LABELS))])
        ll = -0.5 * (((Zv[:, None, :] - mu[None]) / sd[None]) ** 2).mean(2) - np.log(sd).mean(1)[None]
        ev[~fit] = ll - ll.mean(1, keepdims=True)
    return ev


def lr_evidence(X, y, folds, C=0.05):
    ev = np.zeros((len(X), len(E.NLM_LABELS)))
    for f in np.unique(folds):
        fit = folds != f
        sc = StandardScaler().fit(X[fit])
        m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[fit]), y[fit])
        lp = log_softmax(m.decision_function(sc.transform(X[~fit])), 1)
        ev[~fit] = lp - lp.mean(1, keepdims=True)
    return ev


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

    X34 = U.markers(texts, T.MARKS); F = shape_features(texts)
    evs = {"nb34": U.oof_evidence(X34, y, folds), "shape": gauss_evidence(F, y, folds),
           "lr": lr_evidence(np.hstack([X34, F]), y, folds)}
    for k, ev in evs.items():
        print(f"evidence alone {k:6s} {score(ev.argmax(1)):.4f}")
    print(f"evidence alone nb34+shape {score((evs['nb34'] + evs['shape']).argmax(1)):.4f}")

    for bname, P in bases.items():
        raw = np.log(E.sinkhorn(P) + 1e-12)
        q = softmax(raw, 1); qs = np.sort(q, 1); margin = qs[:, -1] - qs[:, -2]
        ref = U.apply(raw, margin, evs["nb34"], *U.DEFAULT)
        print(f"\n== {bname}: no step {score(raw.argmax(1)):.4f}, v38a {score(ref):.4f}")
        cands = {"shape alone": lambda s: U.apply(raw, margin, evs["shape"], s, 0.35, 0.8),
                 "v38a + shape": lambda s: (raw + np.clip((0.8 - margin) / 0.8, 0, 1)[:, None]
                                            * (np.clip(2.0 * evs["nb34"], -.35, .35)
                                               + np.clip(s * evs["shape"], -.35, .35))).argmax(1),
                 "lr in place of nb34": lambda s: U.apply(raw, margin, evs["lr"], s, 0.35, 0.8),
                 "v38a + lr": lambda s: (raw + np.clip((0.8 - margin) / 0.8, 0, 1)[:, None]
                                         * (np.clip(2.0 * evs["nb34"], -.35, .35)
                                            + np.clip(s * evs["lr"], -.35, .35))).argmax(1)}
        for cname, fn in cands.items():
            preds = {s: fn(s) for s in SCALES}
            cf = np.empty_like(ref); picks = []
            for f in np.unique(folds):
                tr = folds != f
                pick = max(SCALES, key=lambda s: balanced_accuracy(y[tr], preds[s][tr]))
                cf[~tr] = preds[pick][~tr]; picks.append(pick)
            c = paired_bootstrap(y, cf, ref, prompts)
            print(f"  {cname:22s} " + " ".join(f"s{s:g} {score(p):.4f}" for s, p in preds.items())
                  + f" | cross-fitted {score(cf):.4f} vs v38a {c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}] picks {picks}")


if __name__ == "__main__":
    main()
