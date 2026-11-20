"""Head-alone OOF quality of his recipe at L256 vs C (and L160 C 0.05 for reference): 18-class accuracy, GLM-vs-rest
AUC, within-GLM accuracy on GLM rows, log loss; plus the stage result (exp058 revert, routing, cover 0.15, W 1)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv.append("revert")
import exp058_stage_w_fable as S
import numpy as np
from sklearn.metrics import roc_auc_score
X = S.X
from ntl.cv import prompt_folds
from ntl.data import load_train
from ntl.metrics import balanced_accuracy
from ntl.oofbank import paired_bootstrap

train = load_train()
ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
prompts = train.prompt_id.to_numpy()
sf, _ = X.load_scores("flash", ids); sg, _ = X.load_scores("glm53", ids)
names = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
folds = prompt_folds(prompts)
nlm = np.load(X.E.NLM_OOF)
members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
members["svc4"] = np.load(X.E.OUT / "svc4_oof.npy").astype(np.float64)
zl = np.log(np.clip(X.E.sinkhorn(X.E.blend(members, X.E.CORE)), 1e-12, None))
score = lambda p: balanced_accuracy(train.model, np.asarray(X.LABELS)[p])
h = X.HIS["entropy_ref"]
glm = np.isin(y, X.GLM)
Xc = {c: X.design(X.features(sf, texts, c), X.features(sg, texts, c), names, X.HIS["diff_features"])[0] for c in (160, 256)}
ref = S.blend(zl, X.oof_head(Xc[160], y, folds, 0.05), 0.15, 1.0, h, True)
for cut, C in [(160, 0.05), (256, 0.01), (256, 0.02), (256, 0.05), (256, 0.1), (256, 0.2), (256, 0.5)]:
    ll = X.oof_head(Xc[cut], y, folds, C)
    p = np.exp(ll)
    auc = roc_auc_score(glm, p[:, X.GLM].sum(1))
    within = (ll[glm][:, X.GLM].argmax(1) == (y[glm] == X.GLM[1]).astype(int)).mean()
    nll = -ll[np.arange(len(y)), y].mean()
    st = S.blend(zl, ll, 0.15, 1.0, h, True)
    g = paired_bootstrap(y, st, ref, prompts, n=1000)
    print(f"L{cut} C {C:<5g} acc {(ll.argmax(1) == y).mean():.4f}  GLM AUC {auc:.4f}  within-GLM {within:.3f}  "
          f"nll {nll:.4f}  stage {score(st):.4f} vs L160: {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]", flush=True)
