"""exp043: a per-class bias on the replica's final log-scores, learnt on four folds (coordinate ascent on balanced
accuracy, or shifts that even out the predicted counts) and applied to the fifth, with and without the stylegram stand-in
for the stack (exp041).

    python -I experiments/exp043_class_bias.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
sys.argv.append("--plain")
import numpy as np, pandas as pd
import exp015_nlm_core_oof as E
import exp034_matched_extra as X
import exp036_svc_matched as M
from exp029_atlas_ext import soft_sinkhorn
from ntl.data import load_train
from ntl.metrics import balanced_accuracy
from ntl.oofbank import paired_bootstrap

train = load_train()
nlm = np.load(E.NLM_OOF)
folds, labels = nlm["folds"], np.array(E.NLM_LABELS)
y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
prompts = train.prompt_id.to_numpy()
mem = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
mem["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
mem["svc4"] = np.load(M.OUT / "m2_svc4.npy").astype(np.float64)
mem["distort"] = np.load(X.OUT / "m2_distort.npy")
mem["tfidf"] = np.load(X.OUT / "m2_tfidf.npy")
E.TERMS.update({"tfidf": 1, "distort": 1})
sty = np.load(f"{ROOT}/outputs/exp035_comp_extra/ntl-v31ac-a20-sk30-osvc12-xm2-cx-p30_stylegram.npy")

def final(comp_c=None):
    P = E.blend(mem, {**E.CORE, "svc4": 12.0, "deb": 1.0, "tfidf": 0.25, "distort": 0.10})
    if comp_c:
        C = np.exp(comp_c * E.lsm(sty)); C /= C.sum(1, keepdims=True)
        P = 0.85 * P + 0.15 * C; P /= P.sum(1, keepdims=True)
    return soft_sinkhorn(P, alpha=0.30)

def ba(L, idx, b):
    return balanced_accuracy(train.model.values[idx], labels[(L[idx] + b).argmax(1)])

def fit_bias(L, idx, steps=(0.4, 0.2, 0.1, 0.05), rounds=3):
    b = np.zeros(18)
    best = ba(L, idx, b)
    for st in steps:
        for _ in range(rounds):
            moved = False
            for c in range(18):
                for d in (st, -st):
                    b2 = b.copy(); b2[c] += d
                    s = ba(L, idx, b2)
                    if s > best + 1e-9:
                        best, b, moved = s, b2, True
            if not moved:
                break
    return b - b.mean()

def fit_counts(L, idx, iters=200, lr=0.05):
    """Shift b until each class is predicted about equally often on idx (no labels used beyond the even target)."""
    b = np.zeros(18)
    for _ in range(iters):
        cnt = np.bincount((L[idx] + b).argmax(1), minlength=18)
        b -= lr * (np.log(cnt + 1) - np.log(len(idx) / 18 + 1))
    return b - b.mean()

for name, c in (("replica", None), ("replica + stylegram x3", 3.0)):
    L = final(c)
    base = L.argmax(1)
    for how, fitter in (("BA ascent", fit_bias), ("even counts", fit_counts)):
        out = base.copy(); bs = []
        for k in np.unique(folds):
            tr = np.flatnonzero(folds != k); va = folds == k
            b = fitter(L, tr); bs.append(b)
            out[va] = (L[va] + b).argmax(1)
        g = paired_bootstrap(y, out, base, prompts)
        bm = np.mean(bs, 0)
        top = ", ".join(f"{labels[i].split('-', 1)[1][:10]} {bm[i]:+.2f}" for i in np.argsort(-np.abs(bm))[:6])
        print(f"{name:24s} {how:11s}: {balanced_accuracy(train.model, labels[base]):.4f} -> {balanced_accuracy(train.model, labels[out]):.4f} "
              f"({g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]) | mean bias: {top}", flush=True)
