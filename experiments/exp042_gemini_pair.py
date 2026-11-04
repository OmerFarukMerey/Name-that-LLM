"""exp042: the replica's largest confusion is gemini-3.7-flash -> 3.8-flash (crtproduction found AI Studio's "3.7" rows
indistinguishable from 3.8 and kept only Vertex AI ones from pool v11). Out of fold on K's replica: the Gemini confusion,
predicted counts, each member's 3.7-vs-3.8 AUC and offset on true 3.7 / 3.8 rows, and a shift on the 3.7 logit chosen on
the other folds.

    python -I experiments/exp042_gemini_pair.py
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
P = E.blend(mem, {**E.CORE, "svc4": 12.0, "deb": 1.0, "tfidf": 0.25, "distort": 0.10})
L = soft_sinkhorn(P, alpha=0.30)
pred = L.argmax(1)
score = lambda p: balanced_accuracy(train.model, labels[p])
print(f"K replica {score(pred):.4f}")
g = [i for i, l in enumerate(labels) if l.startswith("gemini")]
cm = pd.crosstab(pd.Series(labels[y], name="true"), pd.Series(labels[pred], name="pred"))
print(cm.loc[labels[g], [c for c in labels[g] if c in cm.columns]].to_string())
print("predicted counts (all 18):", dict(zip([l.split('-', 1)[1][:10] for l in labels], np.bincount(pred, minlength=18))))
# per-member view of 3.7 vs 3.8 on true 3.7 / 3.8 rows
i37, i38 = list(labels).index("gemini-3.7-flash"), list(labels).index("gemini-3.8-flash")
for k, v in list(mem.items()) + [("blend+sinkhorn", L)]:
    d = v[:, i37] - v[:, i38]
    auc = (d[y == i37][:, None] > d[y == i38][None, :]).mean()
    print(f"{k:15s} 3.7-vs-3.8 AUC on true 3.7/3.8 rows {auc:.3f}; mean d (3.7 rows) {d[y == i37].mean():+.3f} (3.8 rows) {d[y == i38].mean():+.3f}")
# bias shift on the 3.7 logit, chosen on the other folds, applied to each fold
deltas = np.round(np.arange(-0.5, 2.01, 0.05), 3)
out = pred.copy()
for k in np.unique(folds):
    tr = folds != k
    best = max(deltas, key=lambda dl: balanced_accuracy(train.model[tr], labels[(L[tr] + np.eye(18)[i37] * dl).argmax(1)]))
    out[folds == k] = (L[folds == k] + np.eye(18)[i37] * best).argmax(1)
    print(f"fold {k}: delta {best}", end="; ")
gap = paired_bootstrap(y, out, pred, prompts)
print(f"\ncross-fitted 3.7 shift: {score(out):.4f} ({gap[0]:+.4f} [{gap[1]:+.4f}, {gap[2]:+.4f}])")
for dl in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
    p2 = (L + np.eye(18)[i37] * dl).argmax(1)
    rec = pd.Series(p2 == y).groupby(y).mean()
    print(f"delta {dl}: BA {score(p2):.4f}  3.7 recall {rec[i37]:.3f}  3.8 recall {rec[i38]:.3f}  3.7 predicted {np.sum(p2 == i37)}")
